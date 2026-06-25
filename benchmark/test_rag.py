"""RAG 质量诊断工具

对单个任务运行完整流水线（planner → … → analyst），
拦截每次 retrieve_evidence 调用，使用 RAGAS 评测检索与生成质量。

指标（均为 RAGAS LLM-judge，不含 embedding-based 指标）：
  faithfulness          答案中每条声明是否能从检索内容推导（claim 级别核查）
  context_precision     检索到的 chunk 中对回答问题有用的比例（加权精度）
  chunk_score_mean      Qdrant 原始检索分数均值（向量层，仅供参考）
  threshold_pass_rate   检索 chunk 中分数 > 0.45 的比例（仅供参考）
  evidence_gap_rate     analyst 报告证据缺口的子问题比例（0 最优）

用法：
  python benchmark/test_rag.py 04
  python benchmark/test_rag.py 04 --no-report      # 跳过 fact_checker/report_writer
  python benchmark/test_rag.py 04 --verbose         # 打印每条 chunk 文本摘要
  python benchmark/test_rag.py 04 --skip-ragas      # 跳过 RAGAS 评测，只看运营指标
"""
from __future__ import annotations

# ── RAGAS 依赖垫片（ragas 0.4.x 依赖已被 langchain-community 0.4+ 移除的 vertexai 模块）
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

TASKS_FILE = Path(__file__).parent / "tasks.json"
THRESHOLD  = 0.45

GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RED    = "\033[31m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def _g(t): return f"{GREEN}{t}{RESET}"
def _y(t): return f"{YELLOW}{t}{RESET}"
def _c(t): return f"{CYAN}{t}{RESET}"
def _r(t): return f"{RED}{t}{RESET}"
def _b(t): return f"{BOLD}{t}{RESET}"
def _d(t): return f"{DIM}{t}{RESET}"

# ── RAG 调用拦截 ───────────────────────────────────────────────────────────────
_rag_trace: dict[str, dict] = {}


def _patch_rag() -> None:
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


# ── RAGAS 设置 ────────────────────────────────────────────────────────────────
_RAGAS_SEM = asyncio.Semaphore(3)   # 限制并发 RAGAS LLM 调用数


def _make_ragas_llm():
    """用项目 LLM 配置初始化 RAGAS InstructorLLM。"""
    from app.core.config import settings
    import openai as _oa
    from ragas.llms import llm_factory
    model = settings.llm_model
    if model.startswith("openai/"):
        model = model[len("openai/"):]
    client = _oa.AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or None,
    )
    return llm_factory(model=model, provider="openai", client=client, max_tokens=4096)


async def _ragas_score(metric, **kwargs) -> float | None:
    """调用 RAGAS 指标，限速并捕获异常。"""
    async with _RAGAS_SEM:
        try:
            result = await metric.ascore(**kwargs)
            v = result.value
            return float(v) if v is not None else None
        except Exception as exc:
            print(f"  {_y('[ragas]')} {type(metric).__name__} error: {exc}", file=sys.stderr)
            return None


# ── 指标计算 ──────────────────────────────────────────────────────────────────

async def compute_metrics(
    sub_questions: list[dict],
    sub_answers:   list[dict],
    skip_ragas:    bool = False,
) -> list[dict]:
    """为每个子问题计算 RAGAS 指标 + 检索运营指标。"""
    if not skip_ragas:
        from ragas.metrics.collections import Faithfulness, ContextPrecisionWithoutReference
        ragas_llm    = _make_ragas_llm()
        faith_metric = Faithfulness(llm=ragas_llm)
        ctx_metric   = ContextPrecisionWithoutReference(llm=ragas_llm)

    results: list[dict] = []

    for sq in sub_questions:
        qid      = sq.get("id", "")
        question = sq.get("question", "")
        trace    = _rag_trace.get(question)
        ans_rec  = next((a for a in sub_answers if a.get("sub_question_id") == qid), {})
        answer   = ans_rec.get("answer", "")

        base: dict = {
            "qid":               qid,
            "question":          question,
            "answer":            answer,
            "confidence":        ans_rec.get("confidence", 0.0),
            "evidence_gap":      ans_rec.get("evidence_gap", True),
            "citations":         ans_rec.get("citations", []),
            "n_total":           0,
            "n_above":           0,
            "score_mean":        0.0,
            "scores":            [],
            "threshold_pass_rate": 0.0,
            "faithfulness":      None,
            "context_precision": None,
            "rag_available":     trace is not None,
        }

        if not trace:
            results.append(base)
            continue

        chunks  = trace["chunks"]
        scores  = trace["scores"]
        n_above = trace["n_above"]
        n_total = trace["n_total"]

        base["n_total"]             = n_total
        base["n_above"]             = n_above
        base["scores"]              = scores
        base["score_mean"]          = float(np.mean(scores)) if scores else 0.0
        base["threshold_pass_rate"] = n_above / n_total      if n_total else 0.0

        if not skip_ragas and chunks and question and answer:
            chunk_texts   = [c.get("text", "")[:1200] for c in chunks]
            answer_trunc  = answer[:1500]   # guard against very long analyst answers
            faith_val, ctx_val = await asyncio.gather(
                _ragas_score(faith_metric, user_input=question, response=answer_trunc, retrieved_contexts=chunk_texts),
                _ragas_score(ctx_metric,   user_input=question, response=answer_trunc, retrieved_contexts=chunk_texts),
            )
            base["faithfulness"]      = faith_val
            base["context_precision"] = ctx_val

        results.append(base)

    return results


# ── 输出 ──────────────────────────────────────────────────────────────────────

def _bar(score: float | None, width: int = 20) -> str:
    if score is None:
        return _d("N/A".ljust(width + 5))
    filled = int(score * width)
    bar    = "█" * filled + "░" * (width - filled)
    color  = _g if score >= 0.7 else (_y if score >= 0.5 else _r)
    return f"{color(bar)} {score:.3f}"


def print_per_question(metrics: list[dict], verbose: bool = False) -> None:
    print(f"\n{'═' * 72}")
    print(f"  {_b('每题 RAG 指标')}")
    print(f"{'═' * 72}")

    for m in metrics:
        qid   = m["qid"]
        gap   = m["evidence_gap"]
        conf  = m["confidence"]
        gap_s = _r("缺口") if gap else _g("充足")
        conf_s = _g(f"{conf:.0%}") if conf >= 0.8 else _y(f"{conf:.0%}")

        print(f"\n  {_b(qid)} — {_d(m['question'][:60])}")
        print(f"  证据: {gap_s}  置信度: {conf_s}  引用: {', '.join(m['citations']) or '无'}")

        if not m["rag_available"]:
            print(f"  {_d('(未命中 RAG，已回退到原始文档)')}")
            continue

        n_t = m["n_total"]
        n_a = m["n_above"]
        print(f"  检索: {n_t} 条, 通过阈值({THRESHOLD}): {_g(str(n_a)) if n_a > 0 else _r('0')} 条")
        scores_str = "  " + " ".join(
            (_g if s > THRESHOLD else _r)(f"{s:.3f}") for s in m["scores"]
        )
        print(f"  分数: {scores_str}")

        # RAGAS 指标
        if m["faithfulness"] is not None or m["context_precision"] is not None:
            print(f"  faithfulness      : {_bar(m['faithfulness'])}")
            print(f"  context_precision : {_bar(m['context_precision'])}")
        else:
            print(f"  {_d('（RAGAS 指标已跳过或计算失败）')}")

        if verbose:
            chunks = _rag_trace.get(m["question"], {}).get("chunks", [])
            for i, (c, s) in enumerate(zip(chunks, m["scores"])):
                prefix = _g("✓") if s > THRESHOLD else _r("✗")
                title  = c.get("title", "")[:40]
                text   = c.get("text", "")[:80].replace("\n", " ")
                print(f"    {prefix} [{i+1}] {s:.3f}  {title}")
                print(f"         {_d(text)}…")


def print_summary(task: dict, metrics: list[dict], elapsed: dict[str, float]) -> None:
    print(f"\n{'═' * 72}")
    print(f"  {_b('RAG 质量汇总')} — [{task['id']}] {task['name']}")
    print(f"{'═' * 72}")

    n = len(metrics)
    if n == 0:
        print("  无数据")
        return

    has_rag   = [m for m in metrics if m["rag_available"]]
    has_ragas = [m for m in has_rag if m["faithfulness"] is not None]

    gap_rate = sum(1 for m in metrics if m["evidence_gap"]) / n
    avg_conf = float(np.mean([m["confidence"] for m in metrics]))

    def _agg(lst, key):
        vals = [x[key] for x in lst if x.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    n_above_total = sum(m["n_above"] for m in has_rag)
    n_total_total = sum(m["n_total"] for m in has_rag)
    tpr           = n_above_total / n_total_total if n_total_total else None

    rows = [
        # RAGAS 质量指标
        ("── RAGAS 质量指标 ──────────────────────────────────────────", None, ""),
        ("faithfulness",       _agg(has_ragas, "faithfulness"),      "↑ 答案声明可从 context 推导的比例（claim 级 LLM 核查）"),
        ("context_precision",  _agg(has_rag, "context_precision"),   "↑ 检索到的 chunk 对回答有用的比例（LLM 加权精度）"),
        # 检索运营指标
        ("── 检索运营指标 ────────────────────────────────────────────", None, ""),
        ("chunk_score_mean",   _agg(has_rag,   "score_mean"),        "Qdrant 原始检索分数均值（向量相似度，非独立评估）"),
        ("threshold_pass_rate", tpr,                                  f"分数 > {THRESHOLD} 的 chunk 比例"),
        # 系统指标
        ("── 系统指标 ────────────────────────────────────────────────", None, ""),
        ("evidence_gap_rate",  gap_rate,                             "↓ analyst 报告证据缺口比例（0 最优）"),
        ("avg_confidence",     avg_conf,                             "analyst 输出置信度均值（自评）"),
    ]

    print(f"\n  {'指标':<26} {'值':>8}   {'说明'}")
    print(f"  {'-' * 68}")
    for name, val, desc in rows:
        if val is None and desc == "":
            print(f"\n  {_d(name)}")
            continue
        if val is None:
            val_s = _d("N/A".rjust(8))
        else:
            pct = f"{val:.3f}"
            if name == "evidence_gap_rate":
                val_s = _g(pct.rjust(8)) if val == 0 else (_y(pct.rjust(8)) if val < 0.3 else _r(pct.rjust(8)))
            else:
                val_s = _g(pct.rjust(8)) if val >= 0.7 else (_y(pct.rjust(8)) if val >= 0.5 else _r(pct.rjust(8)))
        print(f"  {name:<26} {val_s}   {_d(desc)}")

    print(f"\n  总 chunk 数: {n_total_total}  通过阈值: {n_above_total}  子问题数: {n}")

    print(f"\n  {'节点':<22} {'耗时':>8}")
    print(f"  {'-' * 32}")
    for node, t in elapsed.items():
        print(f"  {node:<22} {t:>6.1f}s")
    print(f"  {'合计':<22} {sum(elapsed.values()):>6.1f}s")


# ── 流水线 ────────────────────────────────────────────────────────────────────

async def run_task(task: dict, run_report: bool, verbose: bool) -> dict:
    from app.graph.nodes import (
        analyst_node, content_extractor_node, evidence_builder_node,
        fact_checker_node, planner_node, report_writer_node,
        retriever_node, source_evaluator_node,
    )
    import app.graph.nodes as _nodes
    _nodes._ANALYST_LLM_CONCURRENCY = 3

    from datetime import datetime
    now = datetime.now().isoformat()

    state: dict[str, Any] = {
        "task_id": f"rag_test_{task['id']}", "query": task["query"],
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
    if run_report:
        pipeline += [("fact_checker", fact_checker_node), ("report_writer", report_writer_node)]

    elapsed: dict[str, float] = {}
    for name, fn in pipeline:
        print(f"\n  {_d('→')} 节点: {_b(name)}", end="", flush=True)
        t0 = time.perf_counter()
        try:
            result = await fn(state)
            state.update(result)
        except Exception as exc:
            print(f"  {_r('FAILED:')} {exc}")
            import traceback; traceback.print_exc()
            break
        elapsed[name] = round(time.perf_counter() - t0, 2)
        print(f"  ({elapsed[name]}s)")

    return {"state": state, "elapsed": elapsed}


# ── 主程序 ────────────────────────────────────────────────────────────────────

async def main(task_id: str, run_report: bool, verbose: bool, skip_ragas: bool) -> None:
    _patch_rag()

    all_tasks = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    task = next((t for t in all_tasks if t["id"] == task_id), None)
    if task is None:
        print(f"找不到任务 {task_id!r}，可用: {[t['id'] for t in all_tasks]}", file=sys.stderr)
        sys.exit(1)

    print(f"{'═' * 72}")
    print(f"  {_b('test_rag')} — RAG 质量诊断（RAGAS LLM-judge）")
    print(f"  任务: [{task['id']}] {task['name']}  ({task.get('report_type','?')} / {task.get('search_depth','?')})")
    if not run_report:
        print(f"  {_d('跳过 fact_checker 和 report_writer（--no-report 模式）')}")
    if skip_ragas:
        print(f"  {_y('RAGAS 评测已跳过（--skip-ragas 模式）')}")
    print(f"{'═' * 72}")

    out     = await run_task(task, run_report, verbose)
    state   = out["state"]
    elapsed = out["elapsed"]

    sub_questions = state.get("sub_questions", [])
    sub_answers   = state.get("sub_answers",   [])

    if not sub_questions:
        print(_r("planner 未生成子问题，流水线中止"))
        return

    if not _rag_trace:
        print(_y("RAG 未触发，无法计算检索指标"))

    label = "计算运营指标" if skip_ragas else "RAGAS 评测中（LLM judge，需消耗 API token）…"
    print(f"\n{'─' * 72}")
    print(f"  {_b(label)}")
    t0 = time.perf_counter()
    metrics = await compute_metrics(sub_questions, sub_answers, skip_ragas=skip_ragas)
    dt = round(time.perf_counter() - t0, 2)
    print(f"  完成 ({dt}s)")

    print_per_question(metrics, verbose)
    print_summary(task, metrics, elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG 质量诊断 — RAGAS LLM-judge 评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("task_id", help="任务 ID，如 04")
    parser.add_argument("--no-report", dest="no_report", action="store_true",
                        help="跳过 fact_checker 和 report_writer，只跑到 analyst")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="打印每条 chunk 文本摘要")
    parser.add_argument("--skip-ragas", dest="skip_ragas", action="store_true",
                        help="跳过 RAGAS 评测，只显示运营指标（快速模式）")
    args = parser.parse_args()
    asyncio.run(main(args.task_id, run_report=not args.no_report,
                     verbose=args.verbose, skip_ragas=args.skip_ragas))
