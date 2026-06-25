"""RAG 多任务基准测试

对指定任务集运行完整流水线，评测各 RAG 阶段质量，横向对比。
每个任务强制 max_rounds=1，跳过 fact_checker/report_writer 节省时间。

── 切片阶段（零成本，来自 evidence_chunks）──
  chunk_len_p50        中位 chunk 字符数（目标 400-700）
  short_chunk_ratio    < 150 字符的碎片 chunk 占比（↓ 越低越好）

── 检索阶段（零成本，来自 Qdrant 分数）──
  chunk_score_mean     Qdrant 检索分数均值（余弦相似度）
  score_std            每子问题分数标准差均值（↑ 高 = 检索有区分度）
  threshold_pass_rate  分数 > 0.45 的 chunk 比例
  source_diversity     每子问题平均不同来源数（↑ 高 = 检索多样）
  multi_source_ratio   ≥3 个不同来源的子问题比例（↑ 高 = 不依赖单源）

── 质量评估（可选，需要 LLM）──
  faithfulness         RAGAS：答案声明可从 context 推导的比例
  context_precision    RAGAS：检索 chunk 对回答有用的比例

── 系统指标 ──
  evidence_gap_rate    analyst 报告证据缺口的子问题比例（0 最优）

用法：
  python benchmark/benchmark_rag.py                     # 全部 12 个任务
  python benchmark/benchmark_rag.py 01 04 08            # 指定任务
  python benchmark/benchmark_rag.py --save              # 结果写入 benchmark/results/
  python benchmark/benchmark_rag.py --skip-ragas        # 跳过 RAGAS，只看运营指标（快速）
"""
from __future__ import annotations

# ── RAGAS 依赖垫片
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

import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

TASKS_FILE    = Path(__file__).parent / "tasks.json"
RESULTS_DIR   = Path(__file__).parent / "results"
DEFAULT_TASKS = ["01","02","03","04","05","06","07","08","09","10","11","12"]
THRESHOLD     = 0.45

# 基准测试速度限制（防止单任务耗时失控）
# 注意：子问题数量不裁剪——由 planner 自身设计决定，裁剪会扭曲检索/RAGAS 指标。
# 锁修复 + bounded batch 后 OOM 不再是约束（实测 960 chunk encode 7.6s/峰值<6GB），
# 故放宽来源/URL；截断改为按 tavily_score 排序的"质量优先"，不再盲目取前 N。
MAX_BENCH_URLS    = 24   # 爬取上限 → 按相关度排序取 top-24（喂满 16 来源+爬取失败冗余）
MAX_BENCH_SOURCES = 16   # 来源上限 → source_evaluator 已按 final_score 取 top-N（质量优先）

GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def _g(t): return f"{GREEN}{t}{RESET}"
def _y(t): return f"{YELLOW}{t}{RESET}"
def _r(t): return f"{RED}{t}{RESET}"
def _b(t): return f"{BOLD}{t}{RESET}"
def _d(t): return f"{DIM}{t}{RESET}"

# ── RAG 拦截 ──────────────────────────────────────────────────────────────────
_rag_trace: dict[str, dict] = {}
_patched = False


def _patch_rag() -> None:
    global _patched
    if _patched:
        return
    import app.services.rag_service as _mod
    _orig = _mod.RAGService.retrieve_evidence

    async def _traced(self, query: str, task_id: str | None = None, top_k: int | None = None):
        result = await _orig(self, query=query, task_id=task_id, top_k=top_k)
        scores = [c.get("score", 0.0) for c in result]
        _rag_trace[query] = {
            "chunks":  result,
            "scores":  scores,
            "n_above": sum(1 for s in scores if s > THRESHOLD),
            "n_total": len(result),
        }
        return result

    _mod.RAGService.retrieve_evidence = _traced
    _patched = True


# ── Benchmark 速度限流 ────────────────────────────────────────────────────────

def _bench_cap(state: dict, after: str) -> None:
    """各节点之后裁剪中间状态，防止 embedding/LLM/RAGAS 耗时爆炸。"""
    if after == "retriever":
        results = state.get("search_results", [])
        if results:
            # 质量优先：始终按 Tavily 相关度降序排序，使下游 content_extractor 的
            # [:max_sources_per_round] 截断也取到高分结果（DDG 降级结果无分排末尾）
            results = sorted(results, key=lambda r: r.get("tavily_score") or 0.0, reverse=True)
            state["search_results"] = results[:MAX_BENCH_URLS]

    elif after == "source_evaluator":
        sources = state.get("evaluated_sources", [])
        if len(sources) > MAX_BENCH_SOURCES:
            top = sorted(sources, key=lambda x: x.get("final_score", 0), reverse=True)[:MAX_BENCH_SOURCES]
            keep_urls = {s.get("url", "") for s in top}
            state["evaluated_sources"] = top
            # crawled_documents 是 dict 列表，用 .get() 而非 getattr
            state["crawled_documents"] = [
                d for d in state.get("crawled_documents", [])
                if d.get("url") in keep_urls
            ]


# ── Benchmark LLM 并发预算 ────────────────────────────────────────────────────
# benchmark 进程中，pipeline（analyst/fact_checker/planner 等）和 RAGAS 评估的 LLM
# 调用共享同一个信号量（llm_benchmark_concurrency=10），而非各自独立。
# 实现方式：替换 llm_service 模块级变量 _llm_semaphore（LOAD_GLOBAL 运行时查
# __dict__，替换后所有经 LLMService.chat() 的调用自动走新信号量）；_RAGAS_SEM
# 指向同一对象，使 RAGAS 与 pipeline 共用同一预算。
import app.services.llm_service as _llm_mod
from app.core.config import settings as _settings

_llm_mod._llm_semaphore = asyncio.Semaphore(_settings.llm_benchmark_concurrency)
_RAGAS_SEM = _llm_mod._llm_semaphore


def _make_ragas_llm():
    from app.core.config import settings
    import openai as _oa
    from ragas.llms import llm_factory
    model = settings.llm_model
    if model.startswith("openai/"):
        model = model[len("openai/"):]
    client = _oa.AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or None,
        timeout=90,      # openai 默认 read=600s：并发压满端点时挂起的连接会堵 10 分钟
        max_retries=1,   # 失败快速重试 1 次后放弃，由 _ragas_score 兜底为 None
    )
    return llm_factory(model=model, provider="openai", client=client, max_tokens=4096)


async def _ragas_score(metric, **kwargs) -> float | None:
    async with _RAGAS_SEM:
        try:
            # 硬上限：单个 ascore 内部可能对多个 context 串行发多次 LLM 调用，仅靠客户端
            # 超时不足以封顶整体；wait_for 保证任一评分 ≤180s，超时记 None 而非堵死全局槽位。
            result = await asyncio.wait_for(metric.ascore(**kwargs), timeout=180)
            v = result.value
            return float(v) if v is not None else None
        except Exception:
            return None


# ── 指标计算 ──────────────────────────────────────────────────────────────────

async def compute_task_metrics(
    sub_questions:   list[dict],
    sub_answers:     list[dict],
    evidence_chunks: list[dict],
    skip_ragas:      bool = False,
) -> dict:
    if not skip_ragas:
        from ragas.metrics.collections import Faithfulness, ContextPrecisionWithoutReference
        ragas_llm    = _make_ragas_llm()
        faith_metric = Faithfulness(llm=ragas_llm)
        ctx_metric   = ContextPrecisionWithoutReference(llm=ragas_llm)

    # ── 切片阶段指标（来自已索引的 evidence_chunks）────────────────────────────
    chunk_lens = [len(c.get("text", "")) for c in evidence_chunks if c.get("text")]
    chunking: dict = {
        "chunk_count":       len(chunk_lens),
        "chunk_len_p50":     float(np.median(chunk_lens))     if chunk_lens else None,
        "chunk_len_p90":     float(np.percentile(chunk_lens, 90)) if chunk_lens else None,
        "short_chunk_ratio": sum(1 for l in chunk_lens if l < 150) / len(chunk_lens) if chunk_lens else None,
    }

    # ── 检索阶段 + RAGAS 指标（每子问题）─────────────────────────────────────
    per_q: list[dict] = []
    # (rec, question, answer, chunk_texts) — RAGAS 评分目标，循环后统一并发执行
    _ragas_targets: list[tuple[dict, str, str, list[str]]] = []

    for sq in sub_questions:
        qid      = sq.get("id", "")
        question = sq.get("question", "")
        trace    = _rag_trace.get(question)
        ans_rec  = next((a for a in sub_answers if a.get("sub_question_id") == qid), {})
        answer   = ans_rec.get("answer", "")

        rec: dict = {
            "qid":               qid,
            "evidence_gap":      ans_rec.get("evidence_gap", True),
            "confidence":        ans_rec.get("confidence", 0.0),
            "rag_available":     trace is not None,
            "n_total":           0,
            "n_above":           0,
            "score_mean":        None,
            "score_std":         None,
            "unique_sources":    None,
            "faithfulness":      None,
            "context_precision": None,
        }

        if not trace:
            per_q.append(rec)
            continue

        scores = trace["scores"]
        chunks = trace["chunks"]
        rec["n_total"]         = trace["n_total"]
        rec["n_above"]         = trace["n_above"]
        rec["score_mean"]      = float(np.mean(scores))   if scores else None
        rec["score_std"]       = float(np.std(scores))    if len(scores) > 1 else 0.0
        rec["unique_sources"]  = len({c.get("url", "") for c in chunks if c.get("url")})

        if not skip_ragas and chunks and question and answer:
            # 延迟到循环后统一并发评分，跑满全局 LLM 并发预算（见下方 gather）
            _ragas_targets.append((
                rec, question, answer[:1500],          # guard against very long answers
                [c.get("text", "")[:1200] for c in chunks],
            ))

        per_q.append(rec)

    # 所有子问题的 RAGAS 评分一次性并发提交：每个目标内部 faith+ctx 两调用，全部协程
    # 共抢全局 _RAGAS_SEM(=_llm_semaphore, 上限 8) → 跑满并发预算，不再逐题 2 路串行。
    if _ragas_targets:
        async def _score_one(rec: dict, q: str, ans: str, ctxs: list[str]) -> None:
            faith_val, ctx_val = await asyncio.gather(
                _ragas_score(faith_metric, user_input=q, response=ans, retrieved_contexts=ctxs),
                _ragas_score(ctx_metric,   user_input=q, response=ans, retrieved_contexts=ctxs),
            )
            rec["faithfulness"]      = faith_val
            rec["context_precision"] = ctx_val

        await asyncio.gather(*[_score_one(*t) for t in _ragas_targets])

    def _agg(lst, key):
        vals = [x[key] for x in lst if x.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    has_rag   = [x for x in per_q if x["rag_available"]]
    n         = len(per_q)
    n_total   = sum(x["n_total"] for x in has_rag)
    n_above   = sum(x["n_above"] for x in has_rag)

    multi_src = (
        sum(1 for x in has_rag if (x.get("unique_sources") or 0) >= 3) / len(has_rag)
        if has_rag else None
    )

    return {
        # 切片阶段
        **chunking,
        # 检索阶段
        "n_subq":              n,
        "n_chunks_total":      n_total,
        "n_above_threshold":   n_above,
        "chunk_score_mean":    _agg(has_rag, "score_mean"),
        "score_std_mean":      _agg(has_rag, "score_std"),
        "threshold_pass_rate": n_above / n_total if n_total else None,
        "source_diversity":    _agg(has_rag, "unique_sources"),
        "multi_source_ratio":  multi_src,
        # RAGAS 质量评估
        "faithfulness":        _agg(per_q, "faithfulness"),
        "context_precision":   _agg(per_q, "context_precision"),
        # 系统指标
        "evidence_gap_rate":   sum(1 for x in per_q if x["evidence_gap"]) / n if n else 1.0,
        "avg_confidence":      _agg(per_q, "confidence"),
        "per_question":        per_q,
    }


# ── 单任务运行 ────────────────────────────────────────────────────────────────

async def run_one_task(task: dict, skip_ragas: bool) -> dict:
    _rag_trace.clear()

    from app.graph.nodes import (
        analyst_node, content_extractor_node, evidence_builder_node,
        planner_node, retriever_node, source_evaluator_node,
    )
    import app.graph.nodes as _nodes
    _nodes._ANALYST_LLM_CONCURRENCY = 3
    # 放宽爬取上限到 MAX_BENCH_URLS（生产默认 20），与质量排序后的 top-N 对齐
    from app.core.config import settings as _settings
    _settings.max_sources_per_round = MAX_BENCH_URLS
    # Tavily 使用生产默认（advanced + raw_content=True）：
    # retriever_node 的 key probe 会在批量搜索前找到可用 key，消除级联失败，
    # raw_content 使 content_extractor 无需爬取（0s），与生产行为一致。

    now = datetime.now().isoformat()
    state: dict[str, Any] = {
        "task_id": f"bench_rag_{task['id']}", "query": task["query"],
        "language": task.get("language", "zh-CN"), "max_rounds": 1,
        "current_round": 1, "status": "planning", "user_hints": {},
        "research_strategy": {}, "research_plan": {}, "sub_questions": [],
        "search_queries": [], "search_results": [], "search_summaries": [],
        "crawled_documents": [], "evaluated_sources": [], "evidence_chunks": [],
        "sub_answers": [], "fact_check_result": {}, "fact_check_passed": True,
        "follow_up_queries": [], "final_report": "", "errors": [],
        "progress": 0, "progress_message": "", "created_at": now, "updated_at": now,
    }

    pipeline = [
        ("planner", planner_node), ("retriever", retriever_node),
        ("content_extractor", content_extractor_node),
        ("source_evaluator", source_evaluator_node),
        ("evidence_builder", evidence_builder_node), ("analyst", analyst_node),
    ]

    node_times: dict[str, float] = {}
    failed = False
    for name, fn in pipeline:
        t0 = time.perf_counter()
        try:
            result = await fn(state)
            state.update(result)
        except Exception as exc:
            print(f"    {name} FAILED: {exc}", file=sys.stderr)
            failed = True
            break
        node_times[name] = round(time.perf_counter() - t0, 2)
        _bench_cap(state, after=name)  # 裁剪中间状态，防止耗时失控

    if failed or not state.get("sub_answers"):
        return {"task_id": task["id"], "task_name": task["name"],
                "failed": True, "node_times": node_times, "metrics": {}}

    metrics = await compute_task_metrics(
        state.get("sub_questions",  []),
        state.get("sub_answers",    []),
        state.get("evidence_chunks", []),
        skip_ragas=skip_ragas,
    )

    return {"task_id": task["id"], "task_name": task["name"],
            "failed": False, "node_times": node_times, "metrics": metrics}


# ── 结果展示 ──────────────────────────────────────────────────────────────────

# 切片 + 检索阶段（无需 LLM，始终展示）
RETRIEVAL_COLS = [
    ("len_p50",    "chunk_len_p50",       "~", "切片P50字"),
    ("short_r",    "short_chunk_ratio",   "↓", "碎片率"),
    ("score",      "chunk_score_mean",    "↑", "Qdrant分"),
    ("score_std",  "score_std_mean",      "↑", "分数std"),
    ("pass_rate",  "threshold_pass_rate", "↑", "阈值通过"),
    ("diversity",  "source_diversity",    "↑", "来源多样"),
    ("multi_src",  "multi_source_ratio",  "↑", "多源率"),
    ("gap_rate",   "evidence_gap_rate",   "↓", "缺口率"),
]

# RAGAS 质量列（--skip-ragas 时隐藏）
RAGAS_COLS = [
    ("faithful",  "faithfulness",      "↑", "Faithfulness"),
    ("ctx_prec",  "context_precision", "↑", "CtxPrecision"),
]


def _cell(val: float | None, better: str) -> str:
    if val is None:
        return _d("  N/A ")
    if better == "~":   # chunk length: 400-700 is good
        s = f"{val:.0f}"
        colored = _g(s) if 300 <= val <= 800 else (_y(s) if 150 <= val <= 1000 else _r(s))
    elif better == "↓":
        s = f"{val:.3f}"
        colored = _g(s) if val < 0.2 else (_y(s) if val < 0.5 else _r(s))
    else:
        s = f"{val:.3f}"
        colored = _g(s) if val >= 0.7 else (_y(s) if val >= 0.5 else _r(s))
    return colored


def _print_section(results: list[dict], cols: list[tuple], title: str) -> None:
    success = [r for r in results if not r["failed"]]
    col_w = 13

    print(f"\n  {_b(title)}")
    header = f"  {'ID':<4}  {'任务名':<20}  {'子问题':>5}  {'Chunks':>6}"
    for _, _, arrow, label in cols:
        header += f"  {(label + arrow):>{col_w}}"
    header += f"  {'耗时':>7}"
    print(header)
    print(f"  {'-' * (44 + len(cols) * (col_w + 2) + 10)}")

    for r in results:
        m     = r.get("metrics", {})
        times = r.get("node_times", {})
        if r["failed"]:
            print(f"  {r['task_id']:<4}  {r['task_name'][:18]:<20}  {'FAILED':>5}")
            continue
        row = f"  {r['task_id']:<4}  {r['task_name'][:18]:<20}  {m.get('n_subq',0):>5}  {m.get('n_chunks_total',0):>6}"
        for _, key, arrow, _ in cols:
            row += f"  {_cell(m.get(key), arrow):>{col_w}}"
        row += f"  {sum(times.values()):>6.0f}s"
        print(row)

    if len(success) > 1:
        avg_row = f"  {'均值':<4}  {'─'*18:<20}  {'':>5}  {'':>6}"
        for _, key, arrow, _ in cols:
            vals = [r["metrics"].get(key) for r in success if r["metrics"].get(key) is not None]
            avg  = float(np.mean(vals)) if vals else None
            avg_row += f"  {_cell(avg, arrow):>{col_w}}"
        print(avg_row)


def print_results_table(results: list[dict], skip_ragas: bool) -> None:
    print(f"\n{'═' * 110}")
    print(f"  {_b('RAG 基准测试结果')}{'  ' + _y('（RAGAS 已跳过）') if skip_ragas else ''}")
    print(f"{'═' * 110}")

    _print_section(results, RETRIEVAL_COLS, "切片 & 检索阶段指标（确定性，无 LLM 依赖）")

    if not skip_ragas:
        _print_section(results, RAGAS_COLS, "质量评估指标（RAGAS LLM-judge）")
        print(f"\n  {_d('Faithfulness: 答案声明可从 chunk 推导的比例  CtxPrecision: 检索 chunk 对回答有用的比例')}")


# ── 主程序 ────────────────────────────────────────────────────────────────────

async def _warmup_models() -> None:
    """进程启动时预热 embedding / reranker 模型。
    消除每个任务 evidence_builder / analyst 的首次冷启动延迟，
    使节点耗时只反映实际计算，不含模型加载，且 12 任务共享同一次加载。
    """
    from app.services.rag_service import get_rag_service, _get_reranker
    from app.core.config import settings as _s
    t0 = time.perf_counter()
    print(f"  {DIM}预热模型中...{RESET}", flush=True)
    rag = get_rag_service()
    if _s.embedding_provider == "st":
        await rag._get_st_model()
    elif _s.embedding_provider == "fastembed":
        await rag._get_fastembed_model()
    if _s.reranker_enabled:
        await _get_reranker()
    print(f"  {DIM}模型预热完成（{time.perf_counter()-t0:.1f}s）{RESET}", flush=True)


async def main(task_ids: list[str], save: bool, skip_ragas: bool) -> None:
    _patch_rag()

    all_tasks = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    task_map  = {t["id"]: t for t in all_tasks}

    selected = [task_map[tid] for tid in task_ids if tid in task_map]
    missing  = [tid for tid in task_ids if tid not in task_map]
    if missing:
        print(f"[警告] 任务不存在，跳过: {missing}", file=sys.stderr)
    if not selected:
        print("无有效任务，退出", file=sys.stderr)
        sys.exit(1)

    print(f"{'═' * 72}")
    print(f"  {_b('benchmark_rag')} — RAG 多任务基准测试（RAGAS LLM-judge）")
    print(f"  任务: {[t['id'] for t in selected]}")
    if skip_ragas:
        print(f"  {_y('RAGAS 评测已跳过（--skip-ragas）')}")
    print(f"{'═' * 72}")

    await _warmup_models()

    results: list[dict] = []
    total_start = time.perf_counter()

    for i, task in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] {_b(task['id'])} — {task['name']}")
        t0     = time.perf_counter()
        result = await run_one_task(task, skip_ragas=skip_ragas)
        dt     = round(time.perf_counter() - t0, 1)
        results.append(result)

        m      = result.get("metrics", {})
        status = _r("FAILED") if result["failed"] else _g("OK")
        faith  = m.get("faithfulness")
        ctx    = m.get("context_precision")
        suffix = f"  faithfulness={faith:.3f}  ctx_prec={ctx:.3f}" if faith is not None else ""
        print(f"  → {status}  耗时={dt}s  gap={m.get('evidence_gap_rate', '?'):.2f}{suffix}"
              if not result["failed"] else f"  → {status}  耗时={dt}s")

    total_dt = round(time.perf_counter() - total_start, 1)
    print_results_table(results, skip_ragas)
    print(f"\n  总耗时: {total_dt}s")

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RESULTS_DIR / f"rag_benchmark_{ts}.json"
        payload = {
            "timestamp": datetime.now().isoformat(),
            "tasks":     [t["id"] for t in selected],
            "skip_ragas": skip_ragas,
            "results":   results,
        }
        for r in payload["results"]:
            for q in r.get("metrics", {}).get("per_question", []):
                q.pop("chunks", None)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  结果已保存: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG 多任务基准测试（RAGAS LLM-judge）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("task_ids", nargs="*", default=DEFAULT_TASKS,
                        help="任务 ID 列表（默认: 全部 12 个）")
    parser.add_argument("--save", action="store_true",
                        help="结果写入 benchmark/results/rag_YYYYMMDD.json")
    parser.add_argument("--skip-ragas", dest="skip_ragas", action="store_true",
                        help="跳过 RAGAS LLM 评测，只显示运营指标（快速模式）")
    args = parser.parse_args()

    asyncio.run(main(args.task_ids, args.save, args.skip_ragas))
