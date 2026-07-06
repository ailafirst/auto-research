#!/usr/bin/env python3
"""Phase 2 实验：按社区组织证据后 analyst 输出质量 vs 基线。

流程：
  跑完整流水线 (planner → … → evidence_builder) → 建图 → 跑两次 analyst
    ├─ baseline: 标准 shared_system_content（flat 证据列表）
    └─ graph_v:  证据按社区分组 + 标注社区名和成员数
  → RAGAS 对比 faithfulness / context_precision

用法:
  python rag_experiments/phase2_test.py 04
  python rag_experiments/phase2_test.py 04 07
"""

from __future__ import annotations

# ── RAGAS 依赖垫片 ──
import sys as _sys, types as _types
_vs = _types.ModuleType("langchain_community.chat_models.vertexai")
_vs.ChatVertexAI = type("ChatVertexAI", (), {})
_sys.modules.setdefault("langchain_community.chat_models.vertexai", _vs)

import argparse
import asyncio
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

TASKS_FILE = ROOT / "benchmark" / "tasks.json"
GREEN = "\033[32m"; YELLOW = "\033[33m"; RED = "\033[31m"
BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"


async def _warmup():
    from app.core.config import settings as _s
    from app.services.rag_service import _get_reranker, get_rag_service
    t0 = time.perf_counter()
    print(f"  {DIM}预热...{RESET}", flush=True)
    rag = get_rag_service()
    if _s.embedding_provider == "st":
        await rag._get_st_model()
    await _get_reranker()
    if _s.xling_enabled:
        from app.services.translation_service import get_translation_service
        await get_translation_service().translate("预热")
    print(f"  {DIM}预热完成 ({time.perf_counter() - t0:.1f}s){RESET}", flush=True)
    return rag


async def run_full_pipeline(task_id: str, task: dict, max_sources: int = 20) -> dict:
    """跑 planner → retriever → content_extractor → source_evaluator → evidence_builder。"""
    from app.core.config import settings as _s
    from app.graph.nodes import (
        content_extractor_node, evidence_builder_node, planner_node,
        retriever_node, source_evaluator_node,
    )
    _s.max_sources_per_round = max_sources
    now = datetime.now().isoformat()
    state: dict[str, Any] = {
        "task_id": task_id, "query": task["query"],
        "language": task.get("language", "zh-CN"), "max_rounds": 1,
        "current_round": 1, "status": "planning", "user_hints": {},
        "research_strategy": {}, "research_plan": {}, "sub_questions": [],
        "search_queries": [], "search_results": [], "search_summaries": [],
        "crawled_documents": [], "evaluated_sources": [], "evidence_chunks": [],
        "sub_answers": [], "fact_check_result": {}, "fact_check_passed": True,
        "follow_up_queries": [], "citation_registry": [], "citation_mismatches": [],
        "analyst_revision_done": False, "final_report": "", "errors": [],
        "progress": 0, "progress_message": "", "created_at": now, "updated_at": now,
    }
    steps = [
        ("planner", planner_node), ("retriever", retriever_node),
        ("content_extractor", content_extractor_node),
        ("source_evaluator", source_evaluator_node),
        ("evidence_builder", evidence_builder_node),
    ]
    for name, fn in steps:
        t0 = time.perf_counter()
        result = await fn(state)
        state.update(result)
        print(f"    {name}: {time.perf_counter() - t0:.1f}s", flush=True)
    return state


def _build_evidence_text(
    shared_chunks: list[dict],
    citation_registry: list[dict],
    url_to_cid: dict[str, str],
    accepted_docs: list[dict],
    graph=None,
) -> str:
    """构建 evidence_text——可选按社区组织。"""
    if graph:
        # ── 按社区分组 ──
        from collections import OrderedDict
        comm_groups: dict[int, list[dict]] = {}
        no_comm: list[dict] = []
        for ch in shared_chunks:
            key = f"{ch.get('source_id','')}#{ch.get('chunk_index',0)}"
            cid = graph.key_to_community.get(key)
            if cid is not None:
                comm_groups.setdefault(cid, []).append(ch)
            else:
                no_comm.append(ch)

        # 找每个社区的关键词——用频率最高的几个 chunk title 词
        parts: list[str] = []
        for comm in sorted(graph.communities, key=lambda c: c["id"]):
            chs = comm_groups.get(comm["id"], [])
            if not chs:
                continue
            # 取 title 的词频作为"社区名"
            from collections import Counter
            words: list[str] = []
            for ch in chs:
                title = ch.get("title", "")
                words.extend([w for w in title.replace("-", " ").split() if len(w) > 1])
            top_words = [w for w, _ in Counter(words).most_common(3)] if words else []
            topic = " / ".join(top_words) if top_words else "相关主题"
            parts.append(f"[社区 {comm['id']}: 「{topic}」({comm['size']} 篇])")
            for ch in chs:
                url = ch.get("url", "")
                cid_str = url_to_cid.get(url, "C??")
                text = ch.get("text", "")[:800]
                score = ch.get("score", 0)
                parts.append(
                    f"[{cid_str}] {ch.get('title','来源')} (相关度: {score:.2f})\n"
                    f"URL: {url}\n{text}"
                )
            parts.append("")
        # 未归属的 chunks
        if no_comm:
            parts.append("[其他]")
            for ch in no_comm:
                url = ch.get("url", "")
                cid_str = url_to_cid.get(url, "C??")
                text = ch.get("text", "")[:800]
                parts.append(f"[{cid_str}] {ch.get('title','来源')}\nURL: {url}\n{text}")
        return "\n\n".join(parts)

    else:
        # ── 标准 flat 列表 ──
        parts: list[str] = []
        for ch in shared_chunks:
            url = ch.get("url", "")
            cid_str = url_to_cid.get(url, "C??")
            text = ch.get("text", "")[:800]
            score = ch.get("score", 0)
            parts.append(
                f"[{cid_str}] {ch.get('title','来源')} (相关度: {score:.2f})\n"
                f"URL: {url}\n{text}"
            )
        return "\n\n---\n\n".join(parts)


async def run_analyst(
    state: dict, graph=None, label: str = "baseline",
) -> list[dict]:
    """启动 analyst（并发分析所有子问题），返回 sub_answers。"""
    from app.graph.nodes import _analyze_single_question, _read_prompt, _DEPTH_GUIDE, _INTENT_GUIDE, _ANALYST_LLM_CONCURRENCY
    import asyncio

    sub_questions = state.get("sub_questions", [])
    task_id = state.get("task_id", "")
    crawled_docs = state.get("crawled_documents", [])
    evaluated = state.get("evaluated_sources", [])
    search_summaries = state.get("search_summaries", [])
    research_strategy = state.get("research_strategy", {})
    accepted_urls = {e["url"] for e in evaluated if e.get("accepted")}
    accepted_docs = [d for d in crawled_docs if d.get("url") in accepted_urls]

    # 从 Qdrant 检索 + rerank → 共享证据池
    from app.services.rag_service import get_rag_service, rerank_chunks
    from app.core.config import settings
    rag = get_rag_service()
    qdrant_ok = await rag.vector_store.health_check()

    shared_chunks: list[dict] = []
    sq_top_cids: dict[str, list[str]] = {}
    citation_registry: list[dict] = []
    url_to_cid: dict[str, str] = {}

    _cid_n = 1
    for doc in accepted_docs:
        url = doc.get("url", "")
        if url and url not in url_to_cid:
            cid = f"C{_cid_n:02d}"
            url_to_cid[url] = cid
            citation_registry.append({"id": cid, "title": doc.get("title", url), "url": url})
            _cid_n += 1

    if qdrant_ok:
        retrieve_k = settings.reranker_retrieve_k if settings.reranker_enabled else 6

        async def _rag_for_sq(sq):
            try:
                chunks = await rag.retrieve_evidence(
                    query=sq["question"], task_id=task_id, top_k=retrieve_k,
                )
                if settings.reranker_enabled and chunks:
                    chunks = await rerank_chunks(sq["question"], chunks, top_k=settings.reranker_top_k)
                return sq.get("id", ""), chunks
            except Exception as exc:
                return sq.get("id", ""), []

        rag_per_sq = await asyncio.gather(*[_rag_for_sq(sq) for sq in sub_questions])
        seen_keys: set[tuple] = set()
        for qid, chunks in rag_per_sq:
            q_top: list[str] = []
            for ch in chunks:
                url = ch.get("url", "")
                key = (url, ch.get("text", "")[:80])
                cid = url_to_cid.get(url)
                if cid and cid not in q_top and len(q_top) < 3:
                    q_top.append(cid)
                if key not in seen_keys and ch.get("score", 0) > 0.45:
                    seen_keys.add(key)
                    shared_chunks.append(ch)
            sq_top_cids[qid] = q_top

    # 回退：无 RAG 时用原始文档
    if not shared_chunks:
        for doc in accepted_docs:
            url = doc.get("url", "")
            cid = url_to_cid.get(url, "C??")
            content = doc.get("content", "")[:1000]
            if content:
                shared_chunks.append({
                    "score": 0.0, "text": content, "url": url,
                    "title": doc.get("title", ""),
                })

    # 构建 evidence_text（按社区或 flat）
    evidence_text = _build_evidence_text(
        shared_chunks, citation_registry, url_to_cid, accepted_docs, graph,
    )

    # 构建 shared_system_content
    analyst_prompt = _read_prompt("analyst")
    strategy = research_strategy
    depth = strategy.get("depth", "medium")
    intent = strategy.get("intent", "deep_investigation")
    domain = strategy.get("domain", "general")

    sub_questions_list = "\n".join(
        f"{i+1}. [{sq.get('id','')}] {sq.get('question','')}"
        for i, sq in enumerate(sub_questions)
    )
    citation_index = "\n".join(
        f"[{c['id']}] {c['title']} — {c['url']}" for c in citation_registry
    )

    shared_system_content = (
        f"{analyst_prompt}\n\n---\n\n"
        f"## 研究背景\n"
        f"- 原始问题: {state['query']}\n"
        f"- 研究意图: {intent} — {_INTENT_GUIDE.get(intent, '')}\n"
        f"- 分析深度: {depth} — {_DEPTH_GUIDE.get(depth, '')}\n"
        f"- 研究领域: {domain}\n\n"
        f"## 完整子问题列表（共 {len(sub_questions)} 题）\n"
        f"{sub_questions_list}\n\n"
        f"## Citation Registry\n{citation_index}\n\n"
        f"## 全部可用证据（{len(shared_chunks)} 条，跨子问题合并去重）\n"
        f"{evidence_text}"
    )

    sem = asyncio.Semaphore(_ANALYST_LLM_CONCURRENCY)

    async def bounded(sq):
        async with sem:
            return await _analyze_single_question(
                sq=sq,
                shared_system_content=shared_system_content,
                sq_top_cids=sq_top_cids,
                search_summaries=search_summaries,
                research_strategy=research_strategy,
            )

    results = await asyncio.gather(
        *[bounded(sq) for sq in sub_questions],
        return_exceptions=True,
    )
    sub_answers = []
    for sq, r in zip(sub_questions, results):
        if isinstance(r, Exception):
            print(f"    {RED}✗{RESET} {sq.get('id','')}: {r}", file=sys.stderr)
            sub_answers.append({
                "sub_question_id": sq.get("id", ""),
                "question": sq.get("question", ""),
                "answer": "分析失败。", "citations": [],
                "confidence": 0.0, "evidence_gap": True,
            })
        else:
            sub_answers.append(r)

    n_ok = sum(1 for a in sub_answers if a["confidence"] > 0)
    print(f"    {label}: {n_ok}/{len(sub_answers)} sub_answers")

    # 收集每子问题的 chunk 文本（用于 RAGAS）
    sq_chunks: dict[str, list[str]] = {}
    if qdrant_ok:
        for qid, chs in rag_per_sq:
            sq_chunks[qid] = [c.get("text", "") for c in chs if c.get("text")]

    return sub_answers, citation_registry, sq_chunks


async def _compute_ragas(sub_questions: list[dict], sub_answers: list[dict],
                          sq_chunks: dict[str, list[str]]) -> dict:
    """对指定子问题/答案/上下文算 RAGAS faith + ctx_prec，返回均值。"""
    from ragas.metrics.collections import Faithfulness, ContextPrecisionWithoutReference
    import app.services.llm_service as _llm_mod
    from app.core.config import settings as _settings

    # 基准测试 LLM 并发预算
    _llm_mod._llm_semaphore = asyncio.Semaphore(_settings.llm_benchmark_concurrency)
    _RAGAS_SEM = _llm_mod._llm_semaphore

    def _make_ragas_llm():
        import openai as _oa
        model = _settings.llm_model
        if model.startswith("openai/"):
            model = model[len("openai/"):]
        client = _oa.AsyncOpenAI(
            api_key=_settings.llm_api_key,
            base_url=_settings.llm_base_url or None,
            timeout=90, max_retries=1,
        )
        from ragas.llms import llm_factory
        return llm_factory(model=model, provider="openai", client=client, max_tokens=4096)

    ragas_llm = _make_ragas_llm()
    faith = Faithfulness(llm=ragas_llm)
    ctx = ContextPrecisionWithoutReference(llm=ragas_llm)

    async def _score(metric, **kw):
        async with _RAGAS_SEM:
            try:
                r = await asyncio.wait_for(metric.ascore(**kw), timeout=180)
                v = r.value
                return float(v) if v is not None else None
            except Exception:
                return None

    targets = []
    for sq in sub_questions:
        qid = sq.get("id", "")
        question = sq.get("question", "")
        ans_rec = next((a for a in sub_answers if a.get("sub_question_id") == qid), {})
        answer = ans_rec.get("answer", "")
        contexts = sq_chunks.get(qid, [])
        if answer and contexts:
            targets.append((question, answer, contexts))

    if not targets:
        return {"faithfulness": None, "context_precision": None}

    tasks = []
    for q, a, ctx in targets:
        tasks.append(_score(faith, user_input=q, response=a, retrieved_contexts=ctx))
        tasks.append(_score(ctx, user_input=q, response=a, retrieved_contexts=ctx))

    scores = await asyncio.gather(*tasks, return_exceptions=True)
    faith_scores = [s for s in scores[0::2] if isinstance(s, (int, float))]
    ctx_scores = [s for s in scores[1::2] if isinstance(s, (int, float))]

    return {
        "faithfulness": sum(faith_scores) / len(faith_scores) if faith_scores else None,
        "context_precision": sum(ctx_scores) / len(ctx_scores) if ctx_scores else None,
    }


async def eval_task(task: dict, args) -> None:
    """跑一个任务：流水线 → 建图 → baseline analyst → graph_v analyst → 对比。"""
    task_id = f"phase2_{task['id']}"
    print(f"\n{'='*60}")
    print(f"  [{task['id']}] {task['name']}")
    print(f"  query: {task['query']}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    state = await run_full_pipeline(task_id, task)
    print(f"  pipeline: {time.perf_counter()-t0:.1f}s", flush=True)

    # Baseline analyst
    t1 = time.perf_counter()
    base_answers, _, base_sq_chunks = await run_analyst(state, graph=None, label="baseline")
    base_time = time.perf_counter() - t1

    # Build graph
    from rag_experiments.graph_builder import build_graph
    from app.services.rag_service import get_rag_service
    rag = get_rag_service()
    t2 = time.perf_counter()
    graph = await build_graph(rag.vector_store, task_id, threshold=0.70)
    if graph:
        print(f"    graph: {graph.total_chunks} chunks → {len(graph.communities)} communities"
              f"  ({time.perf_counter()-t2:.2f}s)")
    else:
        print(f"    {YELLOW}graph: 建图失败（跳过 graph_v analyst）{RESET}")
        return

    # Graph-V analyst
    t3 = time.perf_counter()
    graph_answers, _, graph_sq_chunks = await run_analyst(state, graph=graph, label="graph_v")
    graph_time = time.perf_counter() - t3

    # ── 输出对比 ──
    print(f"\n  {BOLD}sub_answers 对比{RESET}")
    for sq, ba, ga in zip(state.get("sub_questions", []), base_answers, graph_answers):
        qid = sq.get("id", "")
        qtext = sq.get("question", "")
        # 关键：看 answer 长度、citations 是否有差异
        same_answer = ba.get("answer", "") == ga.get("answer", "")
        same_cites = set(ba.get("citations", [])) == set(ga.get("citations", []))
        same_conf = abs(ba.get("confidence", 0) - ga.get("confidence", 0)) < 0.01
        diff = f"{GREEN}identical{RESET}" if (same_answer and same_cites and same_conf) else f"{YELLOW}differs{RESET}"
        print(f"  [{qid}] {diff}  {qtext[:50]}")
        if not same_answer:
            ba_len = len(ba.get("answer", ""))
            ga_len = len(ga.get("answer", ""))
            print(f"        baseline({ba_len}ch) vs graph_v({ga_len}ch)"
                  f"  cites: {ba.get('citations',[])} → {ga.get('citations',[])}"
                  f"  conf: {ba.get('confidence',0):.2f} → {ga.get('confidence',0):.2f}")

    total_time = base_time + graph_time
    print(f"\n  baseline: {base_time:.1f}s  graph_v: {graph_time:.1f}s  total: {total_time:.1f}s")

    # ── RAGAS 评分 ──
    print(f"\n  {BOLD}RAGAS 评分{RESET}")
    base_ragas = await _compute_ragas(state["sub_questions"], base_answers, base_sq_chunks)
    graph_ragas = await _compute_ragas(state["sub_questions"], graph_answers, graph_sq_chunks)

    print(f"  {'指标':<20}  {'Baseline':>10}  {'GraphRAG-V':>10}  {'Δ':>8}")
    print(f"  {'─' * 52}")
    for key, label in [("faithfulness", "Faithfulness"), ("context_precision", "CtxPrecision")]:
        b = base_ragas.get(key)
        g = graph_ragas.get(key)
        if b is None and g is None:
            print(f"  {label:<20}  {'N/A':>10}  {'N/A':>10}  {'N/A':>8}")
            continue
        b = b or 0.0
        g = g or 0.0
        d = g - b
        ds = f"  {GREEN}{d:+.3f}{RESET}" if d > 0.01 else (f"  {RED}{d:+.3f}{RESET}" if d < -0.01 else f"  {DIM}{d:+.3f}{RESET}")
        bc = GREEN if b > 0.7 else (YELLOW if b > 0.5 else RED)
        gc = GREEN if g > 0.7 else (YELLOW if g > 0.5 else RED)
        print(f"  {label:<20}  {bc}{b:.3f}{RESET:>7}  {gc}{g:.3f}{RESET:>7}  {ds}")

    return base_answers, graph_answers, state, graph


async def main(task_ids: list[str]) -> None:
    all_tasks = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    task_map = {t["id"]: t for t in all_tasks}
    selected = [task_map[t] for t in task_ids if t in task_map]
    if not selected:
        print("无有效任务", file=sys.stderr)
        sys.exit(1)

    await _warmup()

    for task in selected:
        await eval_task(task, sys.argv[1:])
        # 内存 Qdrant 在每个任务后释放（in-memory），无需清理


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 实验：按社区组织证据")
    parser.add_argument("task_ids", nargs="+", help="任务 ID（如 04 07）")
    args = parser.parse_args()
    asyncio.run(main(args.task_ids))
