"""Retriever 基准测试 — 直接调用 planner_node + retriever_node（当前实现）

用法：
  python benchmark/retriever_benchmark.py              # 全部 12 个任务
  python benchmark/retriever_benchmark.py 01 03 07    # 指定任务
  python benchmark/retriever_benchmark.py -c 2        # 限制并发数
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

TASKS_FILE  = Path(__file__).parent / "tasks.json"
RESULTS_DIR = Path(__file__).parent / "results"

# ── ANSI ──────────────────────────────────────────────────────────────────────
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


# ── 状态构建（与 planner_benchmark.py 完全一致）────────────────────────────────

def _build_state(task: dict) -> dict:
    now = datetime.now().isoformat()
    return {
        "task_id":           f"bench_{task['id']}",
        "query":             task["query"],
        "language":          task.get("language", "zh-CN"),
        "max_rounds":        1,
        "current_round":     1,
        "status":            "planning",
        "user_hints":        {},
        "research_strategy": {},
        "research_plan":     {},
        "sub_questions":     [],
        "search_queries":    [],
        "search_results":    [],
        "search_summaries":  [],
        "crawled_documents": [],
        "evaluated_sources": [],
        "evidence_chunks":   [],
        "sub_answers":       [],
        "fact_check_result": {},
        "fact_check_passed": True,
        "follow_up_queries": [],
        "final_report":      "",
        "errors":            [],
        "progress":          0,
        "progress_message":  "",
        "created_at":        now,
        "updated_at":        now,
    }


# ── 单任务执行 ────────────────────────────────────────────────────────────────

async def run_task(task: dict) -> dict:
    from app.graph.nodes import planner_node, retriever_node

    base = {
        "benchmark_task_id":   task["id"],
        "benchmark_task_name": task["name"],
        "query":               task["query"],
        "run_at":              datetime.now().isoformat(),
    }

    state = _build_state(task)

    # ── Planner（与 planner_benchmark.py 调用方式相同）──────────────────────
    tp = time.perf_counter()
    try:
        pout = await planner_node(state)
    except Exception as exc:
        return {**base, "error": f"Planner 失败: {exc}"}
    planner_elapsed = time.perf_counter() - tp
    state.update(pout)

    sub_questions = state["sub_questions"]
    flat_queries  = [q for sq in sub_questions for q in sq.get("search_queries", [])]
    strategy      = pout.get("research_strategy", {})

    # ── Retriever（直接调用 retriever_node，与 nodes.py 完全一致）───────────
    tr = time.perf_counter()
    try:
        rout = await retriever_node(state)
    except Exception as exc:
        return {
            **base,
            "error":     f"Retriever 失败: {exc}",
            "strategy":  strategy,
            "planner_s": round(planner_elapsed, 2),
        }
    retriever_elapsed = time.perf_counter() - tr
    state.update(rout)

    search_results   = state["search_results"]
    search_summaries = state["search_summaries"]

    # 子问题覆盖统计
    sq_counts: dict[str, int] = {sq["id"]: 0 for sq in sub_questions}
    for r in search_results:
        sid = r.get("sub_question_id")
        if sid in sq_counts:
            sq_counts[sid] += 1
    empty_sqs = [sid for sid, n in sq_counts.items() if n == 0]

    # raw_content 覆盖率（Tavily 直接提供正文）
    n_raw   = sum(1 for r in search_results if r.get("raw_content"))
    raw_pct = round(n_raw / max(len(search_results), 1) * 100)

    return {
        **base,
        "strategy": {
            "intent": strategy.get("intent", "—"),
            "domain": strategy.get("domain", "—"),
            "depth":  strategy.get("depth", "—"),
        },
        "sub_questions":  len(sub_questions),
        "total_queries":  len(flat_queries),
        "planner_s":      round(planner_elapsed, 2),
        "retriever_s":    round(retriever_elapsed, 2),
        "results": {
            "total":     len(search_results),
            "summaries": len(search_summaries),
            "empty_sqs": empty_sqs,
            "sq_counts": sq_counts,
            "raw_pct":   raw_pct,
        },
    }


# ── 单任务详情输出 ─────────────────────────────────────────────────────────────

def _print_task(task: dict, r: dict) -> None:
    header = f"[{task['id']}] {task['name']}"
    print(f"\n{'═' * 70}")
    print(f"  {_b(header)}")
    print(f"  {_d(task['query'])}")
    print(f"{'─' * 70}")

    if r.get("error"):
        print(f"  {_r('ERROR:')} {r['error']}")
        return

    s   = r["strategy"]
    res = r["results"]
    dc  = {"shallow": _g, "medium": _y, "deep": _c}.get(s["depth"], _b)

    print(f"  {dc(s['depth'])} · {s['intent']} · {s['domain']}  "
          f"| {r['sub_questions']} 个子问题 · {r['total_queries']} 个查询\n")
    print(f"  {'Planner':<16} {r['planner_s']:.1f}s")
    print(f"  {'Retriever':<16} {r['retriever_s']:.1f}s")
    print(f"  {'搜索结果':<16} {res['total']} 条")
    print(f"  {'search_summaries':<16} {res['summaries']} 条")

    raw_color = _g if res["raw_pct"] >= 80 else _y if res["raw_pct"] >= 50 else _r
    print(f"  {'raw_content':<16} {raw_color(str(res['raw_pct']) + '%')}")

    empty = res["empty_sqs"]
    if empty:
        print(f"  {'零结果子问题':<16} {_r(str(empty))}")
    else:
        print(f"  {'sub_q 归属':<16} {_g('全覆盖')}")


# ── 汇总表 ────────────────────────────────────────────────────────────────────

def _print_summary(results: list[dict], all_tasks: list[dict]) -> None:
    task_map = {t["id"]: t for t in all_tasks}

    print(f"\n\n{'═' * 70}")
    print(f"  {_b('Retriever 基准测试汇总')}")
    print(f"{'═' * 70}")
    print(f"  {'ID':>4}  {'名称':<22}  {'深度':<8}  {'Planner':>8}  {'Retriever':>10}"
          f"  {'结果':>6}  {'摘要':>6}  {'raw%':>5}  {'归属'}")
    print(f"  {'─' * 68}")

    for r in results:
        tid  = r["benchmark_task_id"]
        name = task_map.get(tid, {}).get("name", "")[:22]

        if r.get("error"):
            print(f"  {tid:>4}  {name:<22}  {_r('ERROR: ' + r['error'][:30])}")
            continue

        s   = r["strategy"]
        res = r["results"]
        dc  = {"shallow": _g, "medium": _y, "deep": _c}.get(s["depth"], str)
        empty = res["empty_sqs"]
        attr  = _g("✓") if not empty else _r(f"✗{len(empty)}")
        raw_c = _g if res["raw_pct"] >= 80 else _y if res["raw_pct"] >= 50 else _r

        print(
            f"  {tid:>4}  {name:<22}  {dc(s['depth']):<8}  "
            f"{r['planner_s']:>7.1f}s  {r['retriever_s']:>9.1f}s  "
            f"{res['total']:>6}  {res['summaries']:>6}  "
            f"{raw_c(str(res['raw_pct']) + '%'):>5}  {attr}"
        )

    errors = sum(1 for r in results if r.get("error"))
    print(f"  {'─' * 68}")
    if errors:
        print(f"\n  {_y(f'⚠ {errors} 个任务出错')}")
    print(f"\n  图例  深度: {_g('shallow')}  {_y('medium')}  {_c('deep')}")


# ── 并发执行 ──────────────────────────────────────────────────────────────────

async def _run_with_sem(task: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        result = await run_task(task)

    if result.get("error"):
        print(f"  {_r('✗')} [{task['id']}] {task['name']:<22}  {_r(result['error'][:40])}")
    else:
        s   = result["strategy"]
        res = result["results"]
        dc  = {"shallow": _g, "medium": _y, "deep": _c}.get(s["depth"], str)
        empty = res["empty_sqs"]
        attr  = _g("全覆盖") if not empty else _r(f"空SQ:{len(empty)}")
        print(f"  {_g('✓')} [{task['id']}] {task['name']:<22}  "
              f"{s['intent']} / {dc(s['depth'])}  ({result['planner_s']:.1f}s)  "
              f"结果 {res['total']} 条  摘要 {res['summaries']}  {attr}")

    return result


async def main(task_ids: list[str], concurrency: int) -> None:
    all_tasks: list[dict] = json.loads(TASKS_FILE.read_text(encoding="utf-8"))

    if task_ids:
        selected = [t for t in all_tasks if t["id"] in task_ids]
        missing  = set(task_ids) - {t["id"] for t in selected}
        if missing:
            print(f"未找到任务: {missing}", file=sys.stderr)
    else:
        selected = all_tasks

    if not selected:
        print("没有可运行的任务。", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'═' * 70}")
    print(
        f"  {_b('Retriever 基准测试')}  "
        f"{_d(f'共 {len(selected)} 个任务 · 并发 {min(concurrency, len(selected))}')}"
    )
    print(f"{'═' * 70}\n")

    sem = asyncio.Semaphore(concurrency)
    wall_start = time.perf_counter()
    results_unordered: list[dict] = await asyncio.gather(
        *[_run_with_sem(t, sem) for t in selected]
    )
    wall_elapsed = time.perf_counter() - wall_start

    order   = {t["id"]: i for i, t in enumerate(selected)}
    results = sorted(results_unordered, key=lambda r: order.get(r["benchmark_task_id"], 999))

    task_map = {t["id"]: t for t in selected}
    for result in results:
        _print_task(task_map[result["benchmark_task_id"]], result)

    _print_summary(results, all_tasks)
    print(f"\n  总耗时 {wall_elapsed:.1f}s（并发 {min(concurrency, len(selected))}）")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"retriever_{ts}.json"
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"  结果已保存: {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retriever 基准测试")
    parser.add_argument(
        "tasks", nargs="*",
        help="任务 ID（如 01 03 07），不填则运行全部",
    )
    parser.add_argument(
        "-c", "--concurrency", type=int, default=1,
        help="最大并发任务数（默认 1；Tavily dev key 建议 1-2）",
    )
    args = parser.parse_args()
    asyncio.run(main(args.tasks, args.concurrency))
