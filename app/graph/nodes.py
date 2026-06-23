"""LangGraph 工作流节点实现。

每个节点对应 PRD 中的一个处理阶段：
1. planner_node — 研究规划
2. retriever_node — 信息检索
3. content_extractor_node — 网页抓取清洗
4. source_evaluator_node — 信源评估
5. evidence_builder_node — RAG 证据构建
6. analyst_node — 分析
7. fact_checker_node — 事实核查
8. report_writer_node — 报告生成
"""

import asyncio
import json
import logging
from typing import Any

from app.core.config import settings
from app.graph.state import ResearchState
from app.services.crawler_service import CrawlerService
from app.services.rag_service import RAGService
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)

# 提示词模板路径
PROMPTS_DIR = "app/prompts"

# ── Analyst 策略感知指南 ───────────────────────────────────────────────────
_DEPTH_GUIDE: dict[str, str] = {
    "shallow": "严格 300 字以内，仅核心要点，不展开技术细节",
    "medium":  "严格 500 字以内，涵盖主要方面，可引用数据和示例",
    "deep":    "严格 800 字以内（超出视为违规），覆盖技术机制、多维视角和具体证据",
}

_INTENT_GUIDE: dict[str, str] = {
    "quick_overview":     "直接给出核心答案，避免过度引申",
    "deep_investigation": "从多角度分析，提供具体机制和证据",
    "comparison":         "识别对比维度，量化差异，做出明确判断",
    "how_to":             "按逻辑步骤组织，每步骤给出可操作指导",
    "trend_tracking":     "识别当前状态、演进方向和关键驱动力",
}

_REPORT_TYPE_GUIDE: dict[str, str] = {
    "summary": (
        "报告类型=summary（快速摘要）：控制在 600 字以内，"
        "仅包含「标题 → 执行摘要（3–5 条要点）→ 核心结论 → 参考来源」，不展开子问题章节。"
    ),
    "comparison": (
        "报告类型=comparison（对比分析）：必须包含 Markdown 对比表格（各维度 × 各对象），"
        "随后展开各维度差异分析，最后给出综合建议。"
    ),
    "deep": (
        "报告类型=deep（深度报告）：完整七节结构——"
        "标题 → 摘要 → 研究问题说明 → 核心结论 → 分析（各子问题独立成节）→ 风险与不确定性 → 参考来源。"
    ),
}

# benchmark 可在运行前覆写此值以降低 API 限速风险；生产代码保持默认 5
_ANALYST_LLM_CONCURRENCY: int = 5

# ── Short Output 检测阈值（实测最小有效内容 × 1.2）────────────────────────────
_SHORT_THRESHOLDS: dict[str, int] = {
    "planner":                   720,
    "analyst":                   276,
    "fact_checker":               50,
    "report_writer_summary":     300,   # summary 格式本身 ≤600 字，956 会误触发
    "report_writer_comparison": 3514,
    "report_writer_deep":       5345,
}

# 各节点重试时告知模型"必须包含哪些内容"的描述
_SHORT_REQUIRED_CONTENT: dict[str, str] = {
    "planner": (
        "完整的研究计划 JSON，必须包含：\n"
        "  · question_analysis（intent / domain / depth / dimensions / reasoning）\n"
        "  · research_goal\n"
        "  · 至少 2 个 sub_questions，每题含 angle / question / search_queries（≥3 条）"
    ),
    "analyst": (
        "完整的分析 JSON，必须包含：\n"
        "  · sub_question_id\n"
        "  · answer（针对子问题的实质性回答，150–800 字，取决于深度要求）\n"
        "  · citations（引用来源编号列表）\n"
        "  · confidence（0.0–1.0 的浮点数）\n"
        "  · evidence_gap（true / false）"
    ),
    "fact_checker": (
        "完整的核查结果 JSON，必须包含：\n"
        "  · passed（true / false）\n"
        "  · issues（问题列表，无问题时为空数组 []）\n"
        "  · follow_up_queries（补充查询列表，无时为空数组 []）"
    ),
    "report_writer": (
        "完整的 Markdown 研究报告，必须包含上方「报告类型」所要求的全部章节结构和字数，"
        "不得省略任何部分。"
    ),
}


async def _llm_with_short_retry(
    llm: Any,
    messages: list[dict],
    node_name: str,
    threshold: int,
    **llm_kwargs: Any,
) -> str:
    """调用 LLM；若输出低于 short 阈值则注入诊断信息重试一次。

    重试消息完整告知模型：错误类型、实际与期望长度、上次输出内容、
    可能原因、以及必须包含的内容清单——让模型"知道发生了什么"再重试。
    """
    output = await llm.chat(messages, **llm_kwargs)

    if len(output) >= threshold:
        return output

    # ── 首次输出过短：构建诊断重试消息 ──────────────────────────────────────
    actual_len = len(output)
    pct        = actual_len * 100 // threshold if threshold else 0
    required   = _SHORT_REQUIRED_CONTENT.get(node_name, "完整的节点输出内容")

    error_msg = (
        f"[SHORT OUTPUT ERROR — 请重新生成完整内容]\n\n"
        f"错误类型    : 输出过短 / 内容截断\n"
        f"节点名称    : {node_name}\n"
        f"实际输出    : {actual_len} 字符\n"
        f"最低质量阈值: {threshold} 字符（当前仅达到要求的 {pct}%）\n\n"
        f"你的上一次输出内容如下：\n"
        f"---\n"
        f"{output.strip() if output.strip() else '（空输出，无任何内容）'}\n"
        f"---\n\n"
        f"失败可能原因：\n"
        f"  · 思维链（reasoning）占用了过多 token，导致正文输出空间不足被截断\n"
        f"  · 回复在生成中途中断，内容不完整\n"
        f"  · JSON 括号/引号未正确闭合，导致主体内容丢失\n"
        f"  · 误将思维过程当作最终输出直接返回\n\n"
        f"本次重新生成必须包含以下完整内容：\n"
        f"{required}\n\n"
        f"要求：直接输出完整内容，不要任何前言、解释或省略，"
        f"输出长度不得少于 {threshold} 字符。"
    )

    retry_messages = messages + [
        {"role": "assistant", "content": output},
        {"role": "user",      "content": error_msg},
    ]

    logger.warning(
        "[SHORT] %s 首次输出 %d 字符 < 阈值 %d，注入诊断信息重试",
        node_name, actual_len, threshold,
    )

    retry_output = await llm.chat(retry_messages, **llm_kwargs)

    if len(retry_output) < threshold:
        logger.error(
            "[SHORT RETRY FAILED] %s 重试后仍输出 %d 字符 < 阈值 %d，放行继续流程",
            node_name, len(retry_output), threshold,
        )
    else:
        logger.info(
            "[SHORT RETRY OK] %s 重试后输出 %d 字符 >= 阈值 %d",
            node_name, len(retry_output), threshold,
        )

    return retry_output


def _load_prompt(name: str) -> str:
    """加载提示词模板。"""
    import os
    path = os.path.join(PROMPTS_DIR, f"{name}.md")
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning("提示词文件未找到: %s", path)
        return ""  # 使用内置默认提示词


def _read_prompt(name: str) -> str:
    """读取提示词模板，若文件不存在则使用内置默认值。"""
    prompt = _load_prompt(name)
    if prompt:
        return prompt

    defaults = {
        "analyst": (
            "你是一个数据分析专家。请基于提供的证据，对研究子问题进行回答。\n"
            "请保持客观，仅使用提供的证据，不要在证据之外编造信息。\n"
            "如果证据不足，请明确说明。\n\n"
            "返回 JSON 格式：\n"
            '{\n'
            '  "sub_question_id": "...",\n'
            '  "answer": "...",\n'
            '  "citations": ["S1", "S2"],\n'
            '  "confidence": 0.8,\n'
            '  "evidence_gap": false\n'
            '}'
        ),
        "fact_checker": (
            "你是一个严谨的事实核查专家。请检查分析结果是否存在以下问题：\n"
            "1. 关键结论是否有证据支持\n"
            "2. 引用是否对应原文\n"
            "3. 不同来源是否存在冲突\n"
            "4. 是否出现过度推断\n\n"
            "返回 JSON 格式：\n"
            '{\n'
            '  "passed": true,\n'
            '  "issues": [{"type": "...", "claim": "...", "reason": "..."}],\n'
            '  "follow_up_queries": ["..."]\n'
            '}'
        ),
        "report_writer": (
            "你是一个专业的研究报告撰写专家。请基于研究计划、分析结果"
            "和引用来源，生成一份结构化的 Markdown 研究报告。\n\n"
            "报告结构：\n"
            "1. 标题\n"
            "2. 摘要\n"
            "3. 研究问题说明\n"
            "4. 核心结论\n"
            "5. 分章节分析（对应各个子问题）\n"
            "6. 风险与不确定性\n"
            "7. 参考来源列表\n\n"
            "请使用清晰的 Markdown 格式。在每部分标注引用编号，如 [S1]。"
        ),
    }
    return defaults.get(name, "")


async def planner_node(state: ResearchState) -> dict[str, Any]:
    """研究规划节点 — 自动分析问题策略并生成研究计划。"""
    logger.info("Planner 节点开始执行")

    # 多轮补充研究（round > 1）：sub_questions 已由上轮传入，跳过重新规划
    existing_sqs = state.get("sub_questions", [])
    if existing_sqs and state.get("current_round", 1) > 1:
        follow_ups = state.get("follow_up_queries", [])
        logger.info(
            "Planner 跳过（第 %d 轮），沿用第 1 轮计划（%d 个子问题）",
            state["current_round"], len(existing_sqs),
        )
        return {
            "research_strategy": state.get("research_strategy", {}),
            "research_plan":     state.get("research_plan", {}),
            "sub_questions":     existing_sqs,
            "search_queries":    follow_ups or state.get("search_queries", []),
            "progress":          10,
            "progress_message":  f"沿用第 1 轮研究计划（{len(existing_sqs)} 个子问题）",
        }

    from app.services.llm_service import LLMService

    llm = LLMService()
    prompt = _read_prompt("planner")

    # 构建用户消息：问题 + 语言 + 可选的用户偏好提示
    hints = state.get("user_hints") or {}
    hint_lines: list[str] = []
    if hints.get("report_type"):
        hint_lines.append(f"用户偏好报告类型: {hints['report_type']}（summary=简要概览 / deep=深入分析 / comparison=对比报告）")
    if hints.get("search_depth"):
        hint_lines.append(f"用户偏好搜索深度: {hints['search_depth']}（basic=轻量 / advanced=全面）")
    if hints.get("note"):
        hint_lines.append(f"用户备注: {hints['note']}")

    hint_block = ("\n\n用户可选偏好（仅供参考，可根据问题实际情况调整）：\n" + "\n".join(hint_lines)) if hint_lines else ""

    user_message = (
        f"研究问题: {state['query']}\n"
        f"输出语言: {state.get('language', 'zh-CN')}"
        f"{hint_block}\n\n"
        f"请先分析问题（question_analysis），再生成研究计划。"
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_message},
    ]

    content = await _llm_with_short_retry(
        llm, messages, "planner", _SHORT_THRESHOLDS["planner"],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    plan = json.loads(content)

    strategy = plan.get("question_analysis", {})
    sub_questions = plan.get("sub_questions", [])

    depth = strategy.get("depth", "medium")
    domain = strategy.get("domain", "general")
    intent = strategy.get("intent", "unknown")

    logger.info(
        "Planner 策略: intent=%s, domain=%s, depth=%s, sub_questions=%d",
        intent, domain, depth, len(sub_questions),
    )

    return {
        "research_strategy": strategy,
        "research_plan": plan,
        "sub_questions": sub_questions,
        "search_queries": [
            q for sq in sub_questions
            for q in sq.get("search_queries", [sq.get("question", "")])
        ],
        "progress": 15,
        "progress_message": (
            f"研究计划已生成: {len(sub_questions)} 个子问题"
            f"（{depth}深度 · {domain}领域 · {intent}意图）"
        ),
    }


async def retriever_node(state: ResearchState) -> dict[str, Any]:
    """信息检索节点 — 所有查询完全并发，结果携带 sub_question_id。"""
    logger.info("Retriever 节点开始执行")

    search_service = SearchService()
    sub_questions = state.get("sub_questions", [])

    all_results: list[dict[str, Any]] = []

    if sub_questions:
        follow_ups = state.get("follow_up_queries", [])
        current_round = state.get("current_round", 1)

        if follow_ups and current_round > 1:
            # 第 2 轮：仅搜索 follow_up 查询（避免重复第 1 轮），结果归属为空由 analyst 走 accepted_docs 回退
            tagged: list[tuple[str, str]] = [(q, "") for q in follow_ups[:5]]
        else:
            # 展平：[(query, sub_question_id), ...] — 在展平时打标，保留归属
            tagged = [
                (q, sq.get("id", ""))
                for sq in sub_questions
                for q in sq.get("search_queries", [])
            ]

        async def _search_tagged(query: str, sq_id: str) -> list[dict[str, Any]]:
            results = await search_service.search(query, max_results=settings.max_search_results)
            return [{**r.model_dump(), "sub_question_id": sq_id} for r in results]

        batches = await asyncio.gather(
            *[_search_tagged(q, sq_id) for q, sq_id in tagged],
            return_exceptions=True,
        )
        for i, batch in enumerate(batches):
            if isinstance(batch, Exception):
                logger.warning("搜索失败 [%s]: %s", tagged[i][0][:40], batch)
                continue
            all_results.extend(batch)

        logger.info("并发搜索完成: %d 个查询，%d 条原始结果", len(tagged), len(all_results))
    else:
        # 兜底：sub_questions 为空时（如多轮补充研究传入 follow_up_queries）
        queries = state.get("search_queries", [state["query"]])
        results = await search_service.multi_search(
            queries=queries,
            max_results_per_query=settings.max_search_results,
        )
        all_results = [r.model_dump() for r in results]

    # 全局 URL 去重，保留首次命中的 sub_question_id
    seen_urls: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for r in all_results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            deduped.append(r)

    # 收集 Tavily 摘要答案：每个 sq_id 取首条不重复 answer
    summaries: list[dict[str, Any]] = []
    seen_answers: set[str] = set()
    for r in deduped:
        answer = r.get("query_answer")
        sq_id = r.get("sub_question_id", "")
        if answer and answer not in seen_answers:
            seen_answers.add(answer)
            summaries.append({"sq_id": sq_id, "answer": answer})

    n_raw = sum(1 for r in deduped if r.get("raw_content"))
    return {
        "search_results": deduped,
        "search_summaries": summaries,
        "progress": 30,
        "progress_message": (
            f"搜索完成: {len(deduped)} 条结果，"
            f"{n_raw} 条含 raw_content，{len(summaries)} 条摘要"
        ),
    }


async def content_extractor_node(state: ResearchState) -> dict[str, Any]:
    """内容提取节点 — Tavily raw_content 直接使用，其余 URL 爬取补充。"""
    logger.info("Content Extractor 节点开始执行")

    search_results = state.get("search_results", [])[:settings.max_sources_per_round]
    if not search_results:
        return {"crawled_documents": [], "progress": 45, "progress_message": "没有可处理的搜索结果"}

    documents: list[dict[str, Any]] = []
    urls_to_crawl: list[str] = []

    for r in search_results:
        raw = r.get("raw_content")
        if raw:
            # Tavily 已返回完整正文，无需 HTTP 请求
            documents.append({
                "url": r["url"],
                "title": r.get("title", ""),
                "content": raw,
                "text_length": len(raw),
                "fetch_time": 0.0,
                "error": None,
                "tavily_score": r.get("tavily_score"),
            })
        elif r.get("url"):
            urls_to_crawl.append(r["url"])

    # 补充爬取没有 raw_content 的 URL（DuckDuckGo 结果或 Tavily 未返回正文的情况）
    if urls_to_crawl:
        crawler = CrawlerService()
        crawled = await crawler.batch_fetch(urls_to_crawl)
        documents.extend([doc.model_dump() for doc in crawled])

    n_raw = len(search_results) - len(urls_to_crawl)
    n_crawled = len(urls_to_crawl)
    valid = sum(1 for d in documents if d.get("content") and not d.get("error"))

    logger.info("内容提取完成: raw_content=%d, 爬取=%d, 有效=%d", n_raw, n_crawled, valid)
    return {
        "crawled_documents": documents,
        "progress": 45,
        "progress_message": f"内容提取完成: {valid} 有效（{n_raw} 来自 Tavily，{n_crawled} 爬取）",
    }


async def source_evaluator_node(state: ResearchState) -> dict[str, Any]:
    """信源评估节点 — 评估抓取内容的质量。"""
    logger.info("Source Evaluator 节点开始执行")

    documents = state.get("crawled_documents", [])
    evaluated: list[dict[str, Any]] = []

    for doc in documents:
        if not doc.get("content") or doc.get("error"):
            evaluated.append({
                "url": doc.get("url", ""),
                "title": doc.get("title", ""),
                "final_score": 0.0,
                "accepted": False,
                "reason": "内容为空或抓取失败",
            })
            continue

        content_length = len(doc.get("content", ""))
        title = doc.get("title", "")
        tavily_score = doc.get("tavily_score")

        if tavily_score is not None:
            # Tavily 路径：直接使用查询相关度分数，阈值 0.5
            final_score = round(tavily_score, 3)
            accepted = final_score >= 0.5 and content_length > 50
            reason = f"Tavily score={final_score}" if accepted else f"Tavily score 过低 ({final_score})"
        else:
            # DuckDuckGo 路径：内容长度代理
            final_score = round(min(1.0, content_length / 3000), 3)
            accepted = content_length > 300
            reason = "内容充足" if accepted else f"内容过短 ({content_length} chars)"

        evaluated.append({
            "url": doc.get("url", ""),
            "title": title,
            "content_length": content_length,
            "final_score": final_score,
            "accepted": accepted,
            "reason": reason,
        })

    return {
        "evaluated_sources": evaluated,
        "progress": 55,
        "progress_message": f"信源评估完成: {sum(1 for e in evaluated if e['accepted'])} 个合格来源",
    }


async def evidence_builder_node(state: ResearchState) -> dict[str, Any]:
    """证据构建节点 — 切片、向量化、入库。"""
    logger.info("Evidence Builder 节点开始执行")

    accepted_urls = {
        e["url"] for e in state.get("evaluated_sources", [])
        if e.get("accepted")
    }

    if not accepted_urls:
        logger.warning("没有合格的信源可用于证据构建")
        return {
            "evidence_chunks": [],
            "progress": 60,
            "progress_message": "没有合格的信源",
        }

    # 导入 RAG 服务
    from app.models.source import CrawledDocument

    documents = [
        CrawledDocument(**doc) for doc in state.get("crawled_documents", [])
        if doc.get("url") in accepted_urls
    ]

    try:
        rag_service = RAGService()
        chunks = await rag_service.build_evidence(
            documents=documents,
            task_id=state["task_id"],
        )
        return {
            "evidence_chunks": [c.model_dump() for c in chunks],
            "progress": 65,
            "progress_message": f"证据构建完成: {len(chunks)} 个 Chunk",
        }
    except Exception as exc:
        logger.warning("Qdrant 不可用，跳过证据构建: %s", exc)
        return {
            "evidence_chunks": [],
            "progress": 65,
            "progress_message": "Qdrant 不可用，跳过向量入库",
        }


async def _analyze_single_question(
    sq: dict[str, Any],
    shared_system_content: str,
    sq_top_cids: dict[str, list[str]],
    search_summaries: list[dict[str, Any]] | None = None,
    research_strategy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """分析单个子问题（可并发调用）。

    接收在 analyst_node 中预构建的共享系统消息（包含全部证据），
    仅在 user 消息中追加子问题特定内容，使共享前缀可被后端 KV cache 复用。
    """
    import re
    import json as json_mod
    from app.services.llm_service import LLMService

    question = sq.get("question", "")
    qid = sq.get("id", "")

    strategy = research_strategy or {}
    depth = strategy.get("depth", "medium")

    # Part 3：子问题特定内容（per-question suffix）

    # 3a. 当前子问题的 Tavily 搜索摘要（若有）
    tavily_block = ""
    if search_summaries:
        sq_answers = [s["answer"] for s in search_summaries if s.get("sq_id") == qid]
        if sq_answers:
            bullets = "\n".join(f"- {a}" for a in sq_answers[:3])
            tavily_block = f"[Tavily 搜索摘要]\n{bullets}\n\n"

    # 3b. 本题最相关来源 hint（来自 analyst_node 中的 per-question RAG ranking）
    top_cids = sq_top_cids.get(qid, [])
    hint_block = ""
    if top_cids:
        hint_block = f"本问题最相关来源（优先参考）: {', '.join(top_cids)}\n\n"

    depth_guide = _DEPTH_GUIDE.get(depth, "请给出清晰全面的回答")

    per_question_content = (
        f"{tavily_block}"
        f"{hint_block}"
        f"当前子问题 ID: {qid}\n"
        f"问题: {question}\n\n"
        f"字数要求: {depth_guide}\n\n"
        f"请仅针对当前子问题作答，引用使用上方 Citation Registry 中的 ID（如 [C01]）。\n"
        f"返回 JSON:\n"
        f'{{"sub_question_id": "{qid}", "answer": "...", "citations": ["C01"], '
        f'"confidence": 0.8, "evidence_gap": false}}'
    )

    # OpenAI 协议格式：system content 为纯字符串。
    # 所有子问题共享相同 shared_system_content（前缀），MiMo/OpenAI 后端
    # 自动对重复前缀建立 KV cache，后续调用直接命中。
    messages = [
        {"role": "system", "content": shared_system_content},
        {"role": "user", "content": per_question_content},
    ]

    llm = LLMService()
    result_str = await _llm_with_short_retry(
        llm, messages, "analyst", _SHORT_THRESHOLDS["analyst"],
        temperature=0.3,
    )

    try:
        result = json_mod.loads(result_str)
    except json_mod.JSONDecodeError:
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_str, re.DOTALL)
        if not match:
            raise ValueError(f"LLM 返回格式无法解析 [{qid}]: {result_str[:200]}")
        result = json_mod.loads(match.group(1))

    return {
        "sub_question_id": qid,
        "question": question,
        "answer": result.get("answer", ""),
        "citations": result.get("citations", top_cids[:3]),
        "confidence": round(result.get("confidence", 0.5), 2),
        "evidence_gap": result.get("evidence_gap", not bool(top_cids)),
    }


async def analyst_node(state: ResearchState) -> dict[str, Any]:
    """分析节点 — 并发对所有子问题进行 RAG 检索 + LLM 分析。"""
    logger.info("Analyst 节点开始执行")

    sub_questions = state.get("sub_questions", [])
    task_id = state.get("task_id", "")
    crawled_docs = state.get("crawled_documents", [])
    evaluated = state.get("evaluated_sources", [])
    search_summaries = state.get("search_summaries", [])

    accepted_urls = {e["url"] for e in evaluated if e.get("accepted")}
    accepted_docs = [d for d in crawled_docs if d.get("url") in accepted_urls]
    research_strategy = state.get("research_strategy", {})

    # 初始化共享 RAG 服务（AsyncQdrantClient 支持并发访问）
    rag_service: RAGService | None = None
    qdrant_ok = False
    try:
        rag_service = RAGService()
        qdrant_ok = await rag_service.vector_store.health_check()
    except Exception:
        qdrant_ok = False

    # 构建全局 Citation Registry（每篇通过评估的来源分配唯一稳定 ID）
    citation_registry: list[dict[str, str]] = []
    url_to_cid: dict[str, str] = {}
    _cid_n = 1
    for doc in accepted_docs:
        url = doc.get("url", "")
        if url and url not in url_to_cid:
            cid = f"C{_cid_n:02d}"
            url_to_cid[url] = cid
            citation_registry.append({"id": cid, "title": doc.get("title", url), "url": url})
            _cid_n += 1

    # ── 共享 RAG 池：并发查询所有子问题，去重后作为统一证据前缀 ──────────────
    # 目的：所有子问题共享同一份证据文本，使 LLM 调用的 system message 完全相同，
    # 命中后端 KV prefix cache，降低 prompt 处理开销。
    shared_chunks: list[dict[str, Any]] = []
    sq_top_cids: dict[str, list[str]] = {}  # qid → 本题 top-3 CID（用于 user suffix hint）

    if qdrant_ok and rag_service:
        async def _rag_for_sq(sq: dict) -> tuple[str, list[dict]]:
            try:
                chunks = await rag_service.retrieve_evidence(
                    query=sq["question"], task_id=task_id, top_k=6,
                )
                return sq.get("id", ""), chunks
            except Exception as exc:
                logger.warning("共享 RAG 检索失败 [%s]: %s", sq.get("id"), exc)
                return sq.get("id", ""), []

        rag_per_sq = await asyncio.gather(*[_rag_for_sq(sq) for sq in sub_questions])

        seen_chunk_keys: set[tuple[str, str]] = set()
        for qid, chunks in rag_per_sq:
            q_top: list[str] = []
            for chunk in chunks:
                url = chunk.get("url", "")
                key = (url, chunk.get("text", "")[:80])
                cid = url_to_cid.get(url)
                # 记录本题 top-3 CID（不受去重影响）
                if cid and cid not in q_top and len(q_top) < 3:
                    q_top.append(cid)
                # 全局去重后加入共享池
                if key not in seen_chunk_keys and chunk.get("score", 0) > 0.3:
                    seen_chunk_keys.add(key)
                    shared_chunks.append(chunk)
            sq_top_cids[qid] = q_top

        logger.info("共享 RAG 池: %d 条 chunk（来自 %d 个子问题查询）", len(shared_chunks), len(sub_questions))

    # ── 构建共享证据文本（Part 2）：无 RAG 时回退到原始文档 ───────────────────
    evidence_parts: list[str] = []
    if shared_chunks:
        for chunk in shared_chunks:
            url   = chunk.get("url", "")
            cid   = url_to_cid.get(url, "C??")
            text  = chunk.get("text", "")[:800]
            score = chunk.get("score", 0)
            evidence_parts.append(
                f"[{cid}] {chunk.get('title', '来源')} (相关度: {score:.2f})\n"
                f"URL: {url}\n{text}"
            )
    else:
        for doc in accepted_docs:
            url     = doc.get("url", "")
            cid     = url_to_cid.get(url, "C??")
            content = doc.get("content", "")[:1000]
            if content:
                evidence_parts.append(
                    f"[{cid}] {doc.get('title', '来源')}\nURL: {url}\n{content}"
                )

    evidence_text = "\n\n---\n\n".join(evidence_parts) if evidence_parts else "无可用来源。"

    # ── 构建共享 system message（Part 1 + Part 2）────────────────────────────
    # Part 1：系统规则（analyst prompt）
    # Part 2：研究背景 + 完整子问题列表 + Citation Registry + 全部证据
    # 所有并发 LLM 调用持有完全相同的 system message → 命中 prefix cache
    analyst_prompt = _read_prompt("analyst")
    strategy = research_strategy
    depth  = strategy.get("depth",  "medium")
    intent = strategy.get("intent", "deep_investigation")
    domain = strategy.get("domain", "general")

    sub_questions_list = "\n".join(
        f"{i + 1}. [{sq.get('id', '')}] {sq.get('question', '')}"
        for i, sq in enumerate(sub_questions)
    )
    citation_index = "\n".join(
        f"[{c['id']}] {c['title']} — {c['url']}" for c in citation_registry
    )

    shared_system_content = (
        f"{analyst_prompt}\n\n"
        f"---\n\n"
        f"## 研究背景\n"
        f"- 原始问题: {state['query']}\n"
        f"- 研究意图: {intent} — {_INTENT_GUIDE.get(intent, '')}\n"
        f"- 分析深度: {depth} — {_DEPTH_GUIDE.get(depth, '')}\n"
        f"- 研究领域: {domain}\n\n"
        f"## 完整子问题列表（共 {len(sub_questions)} 题）\n"
        f"{sub_questions_list}\n\n"
        f"## Citation Registry\n"
        f"{citation_index}\n\n"
        f"## 全部可用证据（{len(evidence_parts)} 条，跨子问题合并去重）\n"
        f"{evidence_text}"
    )

    # 最多同时发起 _ANALYST_LLM_CONCURRENCY 个 LLM 并发调用，避免触发 API 限速
    sem = asyncio.Semaphore(_ANALYST_LLM_CONCURRENCY)

    async def bounded_analyze(sq: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            return await _analyze_single_question(
                sq=sq,
                shared_system_content=shared_system_content,
                sq_top_cids=sq_top_cids,
                search_summaries=search_summaries,
                research_strategy=research_strategy,
            )

    # 所有子问题并发执行，return_exceptions=True 保证单个失败不影响其他结果
    results = await asyncio.gather(
        *[bounded_analyze(sq) for sq in sub_questions],
        return_exceptions=True,
    )

    sub_answers: list[dict[str, Any]] = []
    for sq, result in zip(sub_questions, results):
        if isinstance(result, Exception):
            logger.warning("子问题分析异常 [%s]: %s", sq.get("id"), result)
            sub_answers.append({
                "sub_question_id": sq.get("id", ""),
                "question": sq.get("question", ""),
                "answer": "分析失败，请重试。",
                "citations": [],
                "confidence": 0.0,
                "evidence_gap": True,
            })
        else:
            sub_answers.append(result)

    if rag_service:
        await rag_service.vector_store.close()

    logger.info(
        "Analyst 并发完成: %d 个子问题, %d 个成功",
        len(sub_questions),
        sum(1 for a in sub_answers if a["confidence"] > 0),
    )

    return {
        "sub_answers":       sub_answers,
        "citation_registry": citation_registry,
        "progress":          75,
        "progress_message":  f"分析完成: {len(sub_answers)} 个子问题（并发执行）",
    }


async def _check_single_answer(
    sub_answer: dict[str, Any],
    accepted_docs: list[dict[str, Any]],
    llm: Any,
    prompt: str,
    sem: asyncio.Semaphore,
    citation_registry: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """对单个子问题答案进行事实核查（可并发调用）。"""
    import json as json_mod
    import re

    async with sem:
        question   = sub_answer.get("question", "")
        answer     = sub_answer.get("answer", "")          # 完整答案，不截断
        citations  = sub_answer.get("citations", [])
        confidence = sub_answer.get("confidence", 0.5)

        # 用 Citation Registry 构建 URL→CID 映射，让核查来源标签与行内引用一致
        _url_cid = {c["url"]: c["id"] for c in (citation_registry or [])}

        # 提取前 5 条接受来源的内容片段（每条 800 字），使用全局 CID 标签
        source_parts: list[str] = []
        for i, doc in enumerate(accepted_docs[:5]):
            content = (doc.get("content") or "")[:800]
            if content:
                url = doc.get("url", "")
                cid = _url_cid.get(url, f"C{i+1:02d}")
                source_parts.append(
                    f"[{cid}] {doc.get('title', url)}\n{content}"
                )
        source_text = "\n\n".join(source_parts) if source_parts else "（无可用来源）"

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": (
                f"子问题: {question}\n"
                f"置信度: {confidence:.0%}  引用: {', '.join(citations) or '无'}\n\n"
                f"分析结果:\n{answer}\n\n"
                f"## 可用来源内容\n{source_text}\n\n"
                "请对以上分析结果进行事实核查，返回 JSON。"
            )},
        ]
        result_str = await _llm_with_short_retry(
            llm, messages, "fact_checker", _SHORT_THRESHOLDS["fact_checker"],
            temperature=0.1,
        )

        try:
            result = json_mod.loads(result_str)
        except json_mod.JSONDecodeError:
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_str, re.DOTALL)
            if not match:
                return {
                    "sub_question_id": sub_answer.get("sub_question_id", ""),
                    "passed": True, "issues": [], "follow_up_queries": [],
                }
            result = json_mod.loads(match.group(1))

        return {
            "sub_question_id":   sub_answer.get("sub_question_id", ""),
            "passed":            result.get("passed", True),
            "issues":            result.get("issues", []),
            "follow_up_queries": result.get("follow_up_queries", []),
        }


async def fact_checker_node(state: ResearchState) -> dict[str, Any]:
    """事实核查节点 — 按子问题并发核查，每次可见完整答案和来源内容。"""
    logger.info("Fact Checker 节点开始执行")

    from app.services.llm_service import LLMService

    sub_answers  = state.get("sub_answers", [])
    crawled_docs = state.get("crawled_documents", [])
    evaluated    = state.get("evaluated_sources", [])

    if not sub_answers:
        return {
            "fact_check_result":  {"passed": True, "issues": [], "follow_up_queries": []},
            "fact_check_passed":  True,
            "follow_up_queries":  [],
            "progress":           85,
            "progress_message":   "无子问题，跳过事实核查",
        }

    accepted_urls = {e["url"] for e in evaluated if e.get("accepted")}
    accepted_docs = [d for d in crawled_docs if d.get("url") in accepted_urls]

    llm    = LLMService()
    prompt = _read_prompt("fact_checker")
    sem    = asyncio.Semaphore(3)   # 最多 3 个并发核查，避免 API 限速

    citation_registry = state.get("citation_registry", [])
    check_results = await asyncio.gather(
        *[_check_single_answer(sa, accepted_docs, llm, prompt, sem, citation_registry)
          for sa in sub_answers],
        return_exceptions=True,
    )

    all_issues:      list[dict[str, Any]] = []
    all_follow_ups:  list[str]            = []
    seen_follow_ups: set[str]             = set()
    any_failed = False

    for cr in check_results:
        if isinstance(cr, Exception):
            logger.warning("单问题事实核查异常: %s", cr)
            continue
        if not cr.get("passed", True):
            any_failed = True
        for issue in cr.get("issues", []):
            all_issues.append(issue)
        for fq in cr.get("follow_up_queries", []):
            if fq not in seen_follow_ups:
                seen_follow_ups.add(fq)
                all_follow_ups.append(fq)

    fact_check_result = {
        "passed":            not any_failed,
        "issues":            all_issues,
        "follow_up_queries": all_follow_ups[:5],
    }

    logger.info(
        "Fact Checker 并发完成: %d 个子问题, %d 个 issue, %d 条 follow_up",
        len(sub_answers), len(all_issues), len(all_follow_ups),
    )

    return {
        "fact_check_result":  fact_check_result,
        "fact_check_passed":  fact_check_result["passed"],
        "follow_up_queries":  all_follow_ups[:5],
        "progress":           85,
        "progress_message": (
            f"事实核查完成: {len(all_issues)} 个问题"
            if all_issues else "事实核查通过"
        ),
    }


async def report_writer_node(state: ResearchState) -> dict[str, Any]:
    """报告生成节点 — 使用 LLM 生成最终 Markdown 报告。"""
    logger.info("Report Writer 节点开始执行")

    from app.services.llm_service import LLMService

    query             = state["query"]
    research_plan     = state.get("research_plan", {})
    research_strategy = state.get("research_strategy", {})
    user_hints        = state.get("user_hints", {})
    sub_answers       = state.get("sub_answers", [])
    crawled_docs      = state.get("crawled_documents", [])
    evaluated         = state.get("evaluated_sources", [])
    fact_check        = state.get("fact_check_result", {})

    intent      = research_strategy.get("intent", "deep_investigation")
    domain      = research_strategy.get("domain", "general")
    depth       = research_strategy.get("depth", "medium")
    report_type = user_hints.get("report_type", "deep")
    report_type_guide = _REPORT_TYPE_GUIDE.get(report_type, _REPORT_TYPE_GUIDE["deep"])

    sub_answers_text = "\n\n".join([
        f"### {a.get('question', '')}\n"
        f"置信度: {a.get('confidence', 0):.0%}\n"
        f"回答: {a.get('answer', '')}\n"
        f"引用: {', '.join(a.get('citations', []))}"
        for a in sub_answers
    ])

    citation_registry = state.get("citation_registry", [])
    if citation_registry:
        source_lines = [
            f"[{c['id']}] [{c['title']}]({c['url']})"
            for c in citation_registry
        ]
    else:
        seen_urls: set[str] = set()
        source_lines = []
        for doc in crawled_docs:
            url = doc.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                source_lines.append(f"- [{doc.get('title', url)}]({url})")
    sources_text = "\n".join(source_lines) if source_lines else "（无）"

    issues = fact_check.get("issues", [])
    issues_text = "\n".join([
        f"- [{i.get('type')}] {i.get('claim')}: {i.get('reason')}"
        for i in issues
    ]) if issues else "无"

    user_message = (
        f"研究问题: {query}\n"
        f"研究目标: {research_plan.get('research_goal', query)}\n"
        f"问题意图: {intent}  研究领域: {domain}  研究深度: {depth}\n"
        f"报告类型: {report_type_guide}\n"
        f"研究轮次: {state.get('current_round', 1)}/{state.get('max_rounds', 2)}\n"
        f"通过评估的来源数: {sum(1 for e in evaluated if e.get('accepted'))}/{len(crawled_docs)}\n\n"
        f"## 各子问题分析结果\n\n{sub_answers_text}\n\n"
        f"## 事实核查结果\n{issues_text}\n\n"
        f"## 参考来源\n{sources_text}\n\n"
        f"请严格按照上方「报告类型」要求生成 Markdown 研究报告。"
    )

    llm = LLMService()
    prompt = _read_prompt("report_writer")
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_message},
    ]

    rw_threshold = _SHORT_THRESHOLDS.get(
        f"report_writer_{report_type}",
        _SHORT_THRESHOLDS["report_writer_deep"],   # 未知类型按最严格标准
    )
    report = await _llm_with_short_retry(
        llm, messages, "report_writer", rw_threshold,
        temperature=0.3,
    )

    return {
        "final_report": report,
        "progress": 100,
        "progress_message": "研究报告已生成",
    }


def should_continue_research(state: ResearchState) -> str:
    """判断是否需要进行下一轮研究。"""
    current_round = state.get("current_round", 1)
    max_rounds = state.get("max_rounds", 2)
    fact_check_passed = state.get("fact_check_passed", True)
    follow_up = state.get("follow_up_queries", [])

    if current_round >= max_rounds:
        logger.info("达到最大研究轮数 (%d)，结束研究", max_rounds)
        return "report_writer"

    if not fact_check_passed and follow_up:
        logger.info(
            "事实核查未通过，进入第 %d 轮补充研究",
            current_round + 1,
        )
        return "retriever"

    logger.info("事实核查通过，进入报告生成")
    return "report_writer"
