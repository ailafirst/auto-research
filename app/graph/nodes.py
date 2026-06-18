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
from datetime import datetime
from typing import Any

from app.core.config import settings
from app.graph.state import ResearchState
from app.services.crawler_service import CrawlerService
from app.services.rag_service import RAGService
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)

# 提示词模板路径
PROMPTS_DIR = "app/prompts"


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
    """研究规划节点 — 将用户问题拆解为研究计划。"""
    logger.info("Planner 节点开始执行")

    from app.services.llm_service import LLMService

    llm = LLMService()
    prompt = _read_prompt("planner")

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": (
            f"研究问题: {state['query']}\n"
            f"语言: {state.get('language', 'zh-CN')}\n"
            f"请生成研究计划。"
        )},
    ]

    content = await llm.chat(messages, temperature=0.3)
    plan = json.loads(content)

    sub_questions = plan.get("sub_questions", [])
    return {
        "research_plan": plan,
        "sub_questions": sub_questions,
        "search_queries": [
            q for sq in sub_questions
            for q in sq.get("search_queries", [sq.get("question", "")])
        ],
        "progress": 15,
        "progress_message": f"研究计划已生成: {len(sub_questions)} 个子问题",
    }


async def retriever_node(state: ResearchState) -> dict[str, Any]:
    """信息检索节点 — 搜索相关网页。"""
    logger.info("Retriever 节点开始执行")

    search_service = SearchService()
    queries = state.get("search_queries", [state["query"]])

    results = await search_service.multi_search(
        queries=queries,
        max_results_per_query=settings.max_search_results,
    )

    return {
        "search_results": [r.model_dump() for r in results],
        "progress": 30,
        "progress_message": f"搜索完成: {len(results)} 个结果",
    }


async def content_extractor_node(state: ResearchState) -> dict[str, Any]:
    """网页抓取节点 — 抓取并清洗网页内容。"""
    logger.info("Content Extractor 节点开始执行")

    crawler = CrawlerService()
    urls = [
        r["url"] for r in state.get("search_results", [])
        if r.get("url")
    ]

    # 限制抓取数量
    max_urls = settings.max_sources_per_round
    urls = urls[:max_urls]

    if not urls:
        return {
            "crawled_documents": [],
            "progress": 40,
            "progress_message": "没有可抓取的 URL",
        }

    documents = await crawler.batch_fetch(urls)
    valid_docs = [doc for doc in documents if doc.content and not doc.error]

    return {
        "crawled_documents": [doc.model_dump() for doc in documents],
        "progress": 45,
        "progress_message": f"抓取完成: {len(valid_docs)}/{len(urls)} 个页面成功",
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

        # 评分逻辑
        relevance = min(1.0, content_length / 3000) if content_length > 100 else 0.1
        credibility = 0.7  # 基础分，后续可基于域名扩展
        freshness = 0.8    # 基础分
        final_score = relevance * 0.5 + credibility * 0.3 + freshness * 0.2

        accepted = final_score > 0.3 and content_length > 50

        evaluated.append({
            "url": doc.get("url", ""),
            "title": title,
            "content_length": content_length,
            "relevance_score": round(relevance, 2),
            "credibility_score": round(credibility, 2),
            "freshness_score": round(freshness, 2),
            "final_score": round(final_score, 2),
            "accepted": accepted,
            "reason": "通过评估" if accepted else f"评分过低 ({final_score:.2f})",
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
    rag_service: RAGService | None,
    qdrant_ok: bool,
    accepted_docs: list[dict[str, Any]],
    task_id: str,
    use_llm: bool,
) -> dict[str, Any]:
    """分析单个子问题（可并发调用）。"""
    import re
    import json as json_mod
    from app.services.llm_service import LLMService

    question = sq.get("question", "")
    qid = sq.get("id", "")

    # 1. RAG 检索证据（优先）
    rag_chunks: list[dict[str, Any]] = []
    if qdrant_ok and rag_service:
        try:
            rag_chunks = await rag_service.retrieve_evidence(
                query=question, task_id=task_id, top_k=6,
            )
        except Exception as exc:
            logger.warning("RAG 检索失败 [%s]: %s", qid, exc)

    # 2. 构建分析上下文
    context_material: list[str] = []

    if rag_chunks:
        for i, chunk in enumerate(rag_chunks):
            text = chunk.get("text", "")
            score = chunk.get("score", 0)
            if text and score > 0.3:
                context_material.append(
                    f"[来源 R{i+1}] 标题: {chunk.get('title', '来源')} (相关度: {score:.2f})\n"
                    f"URL: {chunk.get('url', '')}\n内容: {text[:800]}"
                )
        logger.info("子问题 '%s' RAG 命中 %d 个", question[:30], len(context_material))

    # RAG 无结果时回退到原始文档
    if not context_material:
        for i, doc in enumerate(accepted_docs):
            content = doc.get("content", "")
            if content:
                context_material.append(
                    f"[来源 S{i+1}] 标题: {doc.get('title', '来源')}\n"
                    f"URL: {doc.get('url', '')}\n内容: {content[:1000]}"
                )

    context_text = "\n\n---\n\n".join(context_material) if context_material else "无可用的来源内容。"

    # 3. LLM 分析
    answer_text = ""
    confidence = 0.1
    evidence_gap = True
    citations: list[str] = []

    if use_llm:
        try:
            llm = LLMService()
            prompt = _read_prompt("analyst")
            messages = [
                {"role": "system",
                 "content": f"{prompt}\n\n你是一个专业研究分析师。请基于提供的来源内容回答。"},
                {"role": "user",
                 "content": f"子问题: {question}\n\n可用来源:\n{context_text}\n\n请返回 JSON。"},
            ]
            result_str = await llm.chat(messages, temperature=0.3)

            try:
                result = json_mod.loads(result_str)
            except json_mod.JSONDecodeError:
                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_str, re.DOTALL)
                result = json_mod.loads(match.group(1)) if match else None

            if result:
                answer_text = result.get("answer", "")
                default_cites = [
                    f"R{i+1}" if rag_chunks else f"S{i+1}"
                    for i in range(min(3, len(context_material)))
                ]
                citations = result.get("citations", default_cites)
                confidence = result.get("confidence", 0.5)
                evidence_gap = result.get("evidence_gap", len(context_material) == 0)

        except Exception as exc:
            logger.warning("LLM 分析失败 [%s]: %s", qid, exc)

    # 4. 降级方案
    if not answer_text:
        if context_material:
            answer_text = f"检索到 {len(context_material)} 个相关来源，以下是主要内容：\n\n"
            for m in context_material[:5]:
                answer_text += f"{m[:400]}\n\n"
        else:
            answer_text = "未找到相关内容。"
        confidence = min(0.5, 0.2 + len(context_material) * 0.05)
        evidence_gap = len(context_material) == 0
        citations = [
            f"R{i+1}" if rag_chunks else f"S{i+1}"
            for i in range(min(3, len(context_material)))
        ]

    return {
        "sub_question_id": qid,
        "question": question,
        "answer": answer_text,
        "citations": citations,
        "confidence": round(confidence, 2),
        "evidence_gap": evidence_gap,
    }


async def analyst_node(state: ResearchState) -> dict[str, Any]:
    """分析节点 — 并发对所有子问题进行 RAG 检索 + LLM 分析。"""
    logger.info("Analyst 节点开始执行")

    sub_questions = state.get("sub_questions", [])
    task_id = state.get("task_id", "")
    crawled_docs = state.get("crawled_documents", [])
    evaluated = state.get("evaluated_sources", [])

    accepted_urls = {e["url"] for e in evaluated if e.get("accepted")}
    accepted_docs = [d for d in crawled_docs if d.get("url") in accepted_urls]

    # 初始化共享 RAG 服务（AsyncQdrantClient 支持并发访问）
    rag_service: RAGService | None = None
    qdrant_ok = False
    if settings.llm_api_key:
        try:
            rag_service = RAGService()
            qdrant_ok = await rag_service.vector_store.health_check()
        except Exception:
            qdrant_ok = False

    use_llm = bool(settings.llm_api_key)

    # 最多同时发起 5 个 LLM 并发调用，避免触发 API 限速
    sem = asyncio.Semaphore(5)

    async def bounded_analyze(sq: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            return await _analyze_single_question(
                sq=sq,
                rag_service=rag_service,
                qdrant_ok=qdrant_ok,
                accepted_docs=accepted_docs,
                task_id=task_id,
                use_llm=use_llm,
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
        "sub_answers": sub_answers,
        "progress": 75,
        "progress_message": f"分析完成: {len(sub_answers)} 个子问题（并发执行）",
    }


async def fact_checker_node(state: ResearchState) -> dict[str, Any]:
    """事实核查节点 — 使用 LLM 检查分析结果的可信度。"""
    logger.info("Fact Checker 节点开始执行")

    sub_answers = state.get("sub_answers", [])
    issues: list[dict[str, Any]] = []
    follow_up: list[str] = []

    # 尝试使用 LLM 进行真实的事实核查
    if settings.llm_api_key:
        try:
            from app.services.llm_service import LLMService

            answers_text = "\n\n".join([
                f"子问题 {a.get('sub_question_id', '?')}: {a.get('question', '')}\n"
                f"分析: {a.get('answer', '')[:500]}\n"
                f"置信度: {a.get('confidence', 0):.0%}\n"
                f"引用: {', '.join(a.get('citations', []))}"
                for a in sub_answers
            ])

            llm = LLMService()
            prompt = _read_prompt("fact_checker")

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": (
                    f"请对以下分析结果进行事实核查。\n\n{answers_text}\n\n"
                    f"返回 JSON。"
                )},
            ]

            result_str = await llm.chat(messages, temperature=0.3)

            import json as json_mod
            import re
            try:
                result = json_mod.loads(result_str)
            except json_mod.JSONDecodeError:
                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_str, re.DOTALL)
                if match:
                    result = json_mod.loads(match.group(1))
                else:
                    raise

            issues = result.get("issues", [])
            follow_up = result.get("follow_up_queries", [])

            fact_check_result = {
                "passed": result.get("passed", len(issues) == 0),
                "issues": issues,
                "follow_up_queries": follow_up[:5],
            }

            return {
                "fact_check_result": fact_check_result,
                "fact_check_passed": fact_check_result["passed"],
                "follow_up_queries": follow_up[:5],
                "progress": 85,
                "progress_message": (
                    f"事实核查完成: {len(issues)} 个问题"
                    if issues else "事实核查通过"
                ),
            }

        except Exception as exc:
            logger.warning("LLM 事实核查失败，降级处理: %s", exc)

    # 降级方案：基于 evidence_gap 和 confidence 的规则检查
    for answer in sub_answers:
        if answer.get("evidence_gap"):
            issues.append({
                "type": "insufficient_evidence",
                "claim": answer.get("question", ""),
                "reason": "当前检索到的证据不足以完整回答此子问题",
            })
            for kw in answer.get("question", "").split()[:3]:
                if len(kw) > 2:
                    follow_up.append(f"{kw} 详细资料")

        if answer.get("confidence", 1.0) < 0.3:
            issues.append({
                "type": "low_confidence",
                "claim": answer.get("question", ""),
                "reason": f"置信度仅 {answer.get('confidence', 0):.0%}，需要更多证据",
            })

    passed = len(issues) == 0
    fact_check_result = {
        "passed": passed,
        "issues": issues,
        "follow_up_queries": follow_up[:5],
    }

    return {
        "fact_check_result": fact_check_result,
        "fact_check_passed": passed,
        "follow_up_queries": follow_up[:5],
        "progress": 85,
        "progress_message": (
            f"事实核查完成: {len(issues)} 个问题"
            if issues else "事实核查通过"
        ),
    }


async def report_writer_node(state: ResearchState) -> dict[str, Any]:
    """报告生成节点 — 整合所有结果为最终报告。"""
    logger.info("Report Writer 节点开始执行")

    query = state["query"]
    research_plan = state.get("research_plan", {})
    sub_answers = state.get("sub_answers", [])
    crawled_docs = state.get("crawled_documents", [])
    evaluated = state.get("evaluated_sources", [])
    fact_check = state.get("fact_check_result", {})

    # 收集引用来源
    sources: list[str] = []
    seen_urls: set[str] = set()
    for doc in crawled_docs:
        url = doc.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            sources.append(f"- [{doc.get('title', '来源')}]({url})")

    # 构建报告
    report = f"""# {research_plan.get('research_goal', query)}

> **自动生成的研究报告**
> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> 研究轮次: {state.get('current_round', 1)}/{state.get('max_rounds', 2)}

---

## 摘要

本研究针对「{query}」进行了多轮搜索与分析。
共检索 {len(crawled_docs)} 个来源，其中 {sum(1 for e in evaluated if e.get('accepted'))} 个通过质量评估。

---

## 研究问题说明

**原始问题**: {query}

**问题类型**: {research_plan.get('question_type', 'general')}

**研究计划包含 {len(research_plan.get('sub_questions', []))} 个子问题**:

"""

    for sq in research_plan.get("sub_questions", []):
        report += f"- {sq.get('question', '')}\n"

    report += "\n---\n\n## 核心结论\n\n"

    for answer in sub_answers:
        report += f"""### {answer.get('question', '')}

**置信度**: {answer.get('confidence', 0):.0%}

{answer.get('answer', '*(无内容)*')}

**引用**: {', '.join(answer.get('citations', [])) if answer.get('citations') else '无直接引用'}

---

"""

    # 事实核查结果
    if fact_check:
        issues = fact_check.get("issues", [])
        if issues:
            report += "## ⚠️ 事实核查发现的问题\n\n"
            for issue in issues:
                report += f"- **[{issue.get('type', 'issue')}]** {issue.get('claim', '')}: {issue.get('reason', '')}\n"
            report += "\n"

    # 风险与不确定性
    evidence_gaps = [a for a in sub_answers if a.get("evidence_gap")]
    if evidence_gaps:
        report += "## 风险与不确定性\n\n"
        report += "以下子问题的证据可能不够充分：\n\n"
        for gap in evidence_gaps:
            report += f"- ⚠️ {gap.get('question', '')}\n"
        report += "\n"

    # 参考来源
    report += "## 参考来源\n\n"
    if sources:
        report += "\n".join(sources)
    else:
        report += "*(无可用来源)*"

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
