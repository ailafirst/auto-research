"""Query Rewriter —— 实验版，不依赖生产代码。

从 app.services.llm_service 仅复用 LLMService 做 LLM 调用，
所有改写/检索逻辑完全独立于 app/services/rag_service.py。

支持的改写模式：
  - keyword:    子问题 → 多组英文关键词短语（Variant A）
  - hyde:       子问题 → 英文假设答案段落（Variant B）
  - multi_query: 子问题 → 多个不同措辞的变体（Variant C）
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_KEYWORD_PROMPT = """\
You are a search query optimizer for a technical document retrieval system. Given a research sub-question, extract {num_groups} distinct keyword groups that would help retrieve relevant technical documents.

Rules:
- Each keyword group is a short string of 3-8 domain-specific English terms/phrases separated by spaces
- Groups should cover DIFFERENT facets of the question
- Terms should be concrete, technical, and likely to appear in English technical documents
- Prioritize: technical terms, methodology names, evaluation metrics, tool names, domain concepts

Return a JSON object with a single key "keywords" containing an array of strings.

Example for "RAG系统如何评估检索质量":
{{"keywords": [
  "RAG retrieval evaluation metrics precision recall",
  "retrieval quality assessment MRR NDCG hit rate",
  "faithfulness context_precision RAGAS benchmark"
]}}

Now process this research sub-question:
Question: {question}
"""

_HYDE_PROMPT = """\
You are a technical document retrieval assistant. Given a research sub-question, write a 2-3 sentence hypothetical passage that WOULD contain the answer to this question.

Rules:
- Write in factual, neutral English as if it's an excerpt from a real technical document or research paper
- Include concrete technical terms, numbers, methodology names, and domain concepts
- DO NOT include meta-commentary like "this passage would discuss..."; just write the passage itself
- The passage is fictional — it is only used as an embedding vector for document retrieval

Return a JSON object with a single key "passage" containing the passage string.

Example for "什么是检索增强生成RAG":
{{"passage": "Retrieval-Augmented Generation (RAG) is a technique that combines a retrieval system with a generative language model. In RAG, a query is first used to retrieve relevant documents from a knowledge corpus, and then these retrieved documents are provided as context to a language model to generate a grounded response. This approach reduces hallucination and enables access to up-to-date information without retraining the model."}}

Now process this research sub-question:
Question: {question}
"""

_MULTI_QUERY_PROMPT = """\
Generate {num_groups} different English phrasings of the given research question. Each variant should express the same information need using different vocabulary and sentence structure. Keep each variant concise (one sentence).

Return a JSON object with a single key "variants" containing an array of strings.

Question: {question}
"""


# ── 通用 JSON 解析 ──

def _parse_json_response(result: str) -> dict | None:
    """从 LLM 响应中提取第一个完整 JSON 对象。"""
    text = result.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        logger.warning("JSON 未找到: %s", result[:80])
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        logger.warning("JSON 解析失败: %s — %s", exc, result[:80])
        return None


async def _call_llm(prompt: str) -> str:
    """调用 LLM 并返回原始文本。"""
    from app.services.llm_service import LLMService
    llm = LLMService()
    return await llm.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
        response_format={"type": "json_object"},
    )


# ── Variant A: Keyword Expansion ──

async def rewrite_keywords(question: str, num_groups: int = 4) -> list[str]:
    """将单个问题扩展为多组英文关键词。"""
    try:
        prompt = _KEYWORD_PROMPT.format(question=question, num_groups=num_groups)
        result = await _call_llm(prompt)
        data = _parse_json_response(result)
        if data is None:
            return []
        keywords = data.get("keywords", [])
        logger.debug("Keyword: %s → %d groups", question[:40], len(keywords))
        return keywords[:num_groups]
    except Exception as exc:
        logger.warning("Keyword 失败（降级）: %s", exc)
        return []


# ── Variant B: HyDE ──

async def rewrite_hyde(question: str) -> list[str]:
    """生成假设答案段落用于 HyDE 检索。

    返回单元素列表（包含假设段落）或空列表（降级）。
    HyDE 段落仅用作检索 query 向量，不进入 LLM 证据上下文。
    """
    try:
        prompt = _HYDE_PROMPT.format(question=question)
        result = await _call_llm(prompt)
        data = _parse_json_response(result)
        if data is None:
            return []
        passage = data.get("passage", "")
        if not passage:
            return []
        logger.debug("HyDE: %s → 段落 %d chars", question[:40], len(passage))
        return [passage]
    except Exception as exc:
        logger.warning("HyDE 失败（降级）: %s", exc)
        return []


# ── Variant C: Multi-Query ──

async def rewrite_multi_query(question: str, num_groups: int = 4) -> list[str]:
    """生成多个不同措辞的查询变体。"""
    try:
        prompt = _MULTI_QUERY_PROMPT.format(question=question, num_groups=num_groups)
        result = await _call_llm(prompt)
        data = _parse_json_response(result)
        if data is None:
            return []
        variants = data.get("variants", [])
        logger.debug("MultiQ: %s → %d variants", question[:40], len(variants))
        return variants[:num_groups]
    except Exception as exc:
        logger.warning("MultiQ 失败（降级）: %s", exc)
        return []


# ── Variant D: HyDE + Keyword 融合 ──

async def rewrite_hyde_keyword_fusion(question: str, num_groups: int = 4) -> list[str]:
    """HyDE 段落 + Keyword 多组关键词融合为 extra_queries。

    同时调用 HyDE 和 Keyword，拼接结果列表，互相补充检索视角。
    """
    import asyncio
    try:
        hyde_task = rewrite_hyde(question)
        kw_task = rewrite_keywords(question, num_groups=num_groups)
        hyde_result, kw_result = await asyncio.gather(hyde_task, kw_task)
        combined = list(hyde_result) + list(kw_result)  # [passage] + [kw_group1, kw_group2, ...]
        combined = [q for q in combined if q]
        logger.debug("Fusion: %s → %d extra queries (hyde=%d, kw=%d)",
                      question[:40], len(combined), len(hyde_result), len(kw_result))
        return combined
    except Exception as exc:
        logger.warning("Fusion 失败（降级）: %s", exc)
        return []


# ── 批量调度 ──

async def batch_rewrite(
    qa_list: list[dict[str, Any]], mode: str = "keyword", num_groups: int = 4,
) -> dict[int, list[str]]:
    """批量改写（mode=keyword|hyde|multi_query），并发 LLM 调用。

    返回 {index_in_qa_list → [query_string, ...]}。
    """
    import asyncio

    fn = {"keyword": rewrite_keywords, "hyde": rewrite_hyde, "multi_query": rewrite_multi_query}
    rewriter = fn.get(mode, rewrite_keywords)

    async def _one(i: int, qa: dict) -> tuple[int, list[str]]:
        extra = await rewriter(qa["question"], num_groups=num_groups) if mode != "hyde" else await rewriter(qa["question"])
        return i, extra

    results = await asyncio.gather(*[_one(i, qa) for i, qa in enumerate(qa_list)])
    return dict(results)


# ── 检索辅助（xling 修复版，不依赖 app/services/rag_service 的行为）─────────────

_CJK_RE = __import__("re").compile("[一-鿿]")


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


async def multi_query_retrieve(
    rag: Any,
    query: str,
    task_id: str,
    top_k: int,
    extra_queries: list[str] | None = None,
) -> list[dict[str, Any]]:
    """多 query 检索：原始 query + extra_queries，逐个做 xling → embed → search → merge。

    这是 retrieve_evidence() 的改进版——对每个含中文的 query 单独附加英文翻译，
    而非原版的"只翻译主 query、有 extra_queries 就跳过 xling"。
    """
    # 1. 收集所有 query 字符串
    q_strings: list[str] = [query]
    if extra_queries:
        q_strings.extend(q for q in extra_queries if q)

    # 2. xling：每个含中文的 query 附加英文翻译
    xling_additions: list[str] = []
    from app.services.translation_service import get_translation_service
    ts = get_translation_service()
    for qs in q_strings:
        if _has_cjk(qs):
            en = await ts.translate(qs)
            if en and en != qs:
                xling_additions.append(en)

    all_q = q_strings + xling_additions

    # 3. 批量 Embed
    vectors = await rag._embed(all_q)
    if not vectors or not vectors[0]:
        return []

    # 4. 单 query 快路径
    if len(vectors) == 1:
        return await rag.vector_store.search(
            query_vector=vectors[0], task_id=task_id, top_k=top_k,
        )

    # 5. 多 query 合并取最高分（不截断——reranker 负责从大池中精排）
    merged: dict[str, dict[str, Any]] = {}
    for vec in vectors:
        if not vec:
            continue
        hits = await rag.vector_store.search(
            query_vector=vec, task_id=task_id, top_k=top_k,
        )
        for h in hits:
            key = f"{h.get('source_id', '')}#{h.get('chunk_index', 0)}"
            if key not in merged or h["score"] > merged[key]["score"]:
                merged[key] = h
    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)
