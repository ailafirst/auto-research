"""Planner 输出基准测试

批量调用 planner_node，记录并展示每个任务的 Planner 决策结果
（intent / domain / depth / dimensions / 子问题 / 搜索词）。

用法：
  python benchmark/planner_benchmark.py              # 全部任务
  python benchmark/planner_benchmark.py 01 03 07    # 指定任务
  python benchmark/planner_benchmark.py -c 4        # 限制并发数
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
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def _g(t: str) -> str: return f"{GREEN}{t}{RESET}"
def _y(t: str) -> str: return f"{YELLOW}{t}{RESET}"
def _c(t: str) -> str: return f"{CYAN}{t}{RESET}"
def _b(t: str) -> str: return f"{BOLD}{t}{RESET}"
def _d(t: str) -> str: return f"{DIM}{t}{RESET}"


# ── 执行 ──────────────────────────────────────────────────────────────────────

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


async def run_task(task: dict) -> dict:
    from app.graph.nodes import planner_node  # noqa: PLC0415

    base = {
        "benchmark_task_id":   task["id"],
        "benchmark_task_name": task["name"],
        "query":               task["query"],
        "run_at":              datetime.now().isoformat(),
    }

    start = time.perf_counter()
    try:
        output = await planner_node(_build_state(task))
        return {
            **base,
            "planner_output": {
                "research_plan":     output.get("research_plan", {}),
                "research_strategy": output.get("research_strategy", {}),
                "sub_questions":     output.get("sub_questions", []),
                "search_queries":    output.get("search_queries", []),
            },
            "elapsed_seconds": round(time.perf_counter() - start, 2),
        }
    except Exception as exc:
        return {
            **base,
            "error":           str(exc),
            "elapsed_seconds": round(time.perf_counter() - start, 2),
        }


# ── 输出 ──────────────────────────────────────────────────────────────────────

def print_task_result(task: dict, result: dict) -> None:
    header = f"[{task['id']}] {task['name']}"
    print(f"\n{'═' * 70}")
    print(f"  {_b(header)}")
    print(f"  {_d(task['query'])}")
    print(f"{'─' * 70}")

    if result.get("error"):
        print(f"  {_y('ERROR:')} {result['error']}")
        return

    po       = result["planner_output"]
    strategy = po.get("research_strategy", po.get("research_plan", {}).get("question_analysis", {}))
    sub_qs   = po.get("sub_questions", [])

    intent    = strategy.get("intent", "—")
    domain    = strategy.get("domain", "—")
    depth     = strategy.get("depth", "—")
    dims      = strategy.get("dimensions", [])
    reasoning = strategy.get("reasoning", "")
    goal      = po.get("research_plan", {}).get("research_goal", "")

    depth_color = {"shallow": _g, "medium": _y, "deep": _c}.get(depth, _b)

    print(f"  intent    {_b(intent)}")
    print(f"  domain    {_b(domain)}")
    print(f"  depth     {depth_color(_b(depth))}")
    print(f"  dims      {_c(str(dims))}")
    if reasoning:
        print(f"  reasoning {_d(reasoning)}")
    if goal:
        print(f"  goal      {goal}")
    print(f"{'─' * 70}")
    print(f"  {_b(f'{len(sub_qs)} 个子问题')}\n")

    for sq in sub_qs:
        sq_id    = sq.get("id", "")
        angle    = sq.get("angle", "")
        question = sq.get("question", "")
        queries  = sq.get("search_queries", [])
        priority = sq.get("priority", "")
        print(f"  {_b(f'[{sq_id}]')} {_c(angle)}  {_d(f'priority={priority}')}")
        print(f"       {question}")
        for q in queries:
            print(f"       {_d('›')} {q}")
        print()

    elapsed_s  = result["elapsed_seconds"]
    n_queries  = len(po.get("search_queries", []))
    print(f"  {_d(f'耗时 {elapsed_s:.1f}s · 搜索词共 {n_queries} 条')}")


def print_summary(results: list[dict], tasks: list[dict]) -> None:
    task_map = {t["id"]: t for t in tasks}

    print(f"\n\n{'═' * 70}")
    print(f"  {_b('Planner 输出摘要')}")
    print(f"{'═' * 70}")
    print(f"  {'ID':>4}  {'名称':<22}  {'意图':<20}  {'领域':<10}  {'深度':<8}  子Q  耗时")
    print(f"  {'─' * 66}")

    for r in results:
        tid  = r["benchmark_task_id"]
        name = (task_map.get(tid, {}).get("name", ""))[:22]

        if r.get("error"):
            print(f"  {tid:>4}  {name:<22}  {_y('ERROR')}")
            continue

        po       = r["planner_output"]
        strategy = po.get("research_strategy", po.get("research_plan", {}).get("question_analysis", {}))
        sub_qs   = po.get("sub_questions", [])

        intent = strategy.get("intent", "—")
        domain = strategy.get("domain", "—")
        depth  = strategy.get("depth", "—")
        depth_color = {"shallow": _g, "medium": _y, "deep": _c}.get(depth, str)

        print(
            f"  {tid:>4}  {name:<22}  {intent:<20}  {domain:<10}  "
            f"{depth_color(depth):<8}  {len(sub_qs):>2}   {r['elapsed_seconds']:.1f}s"
        )

    errors = sum(1 for r in results if r.get("error"))
    print(f"  {'─' * 66}")
    print(f"\n  共 {len(results)} 个任务" + (f"，{_y(f'{errors} 个错误')}" if errors else ""))
    print(f"\n  深度图例: {_g('shallow')}  {_y('medium')}  {_c('deep')}")


# ── 并发执行 ──────────────────────────────────────────────────────────────────

async def _run_with_sem(task: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        result = await run_task(task)
    elapsed = result.get("elapsed_seconds", 0)
    if result.get("error"):
        print(f"  {_y('✗')} [{task['id']}] {task['name']:<22} ERROR ({elapsed:.1f}s)")
    else:
        po       = result["planner_output"]
        strategy = po.get("research_strategy", {})
        intent   = strategy.get("intent", "—")
        depth    = strategy.get("depth", "—")
        print(f"  {_g('✓')} [{task['id']}] {task['name']:<22} {intent} / {depth} ({elapsed:.1f}s)")
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
        f"  {_b('Planner 输出基准测试')}  "
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
        print_task_result(task_map[result["benchmark_task_id"]], result)

    print_summary(results, all_tasks)
    print(f"\n  总耗时 {wall_elapsed:.1f}s（并发 {min(concurrency, len(selected))}）")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"planner_{ts}.json"
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"  结果已保存: {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Planner 输出基准测试")
    parser.add_argument(
        "tasks", nargs="*",
        help="任务 ID（如 01 03 07），不填则运行全部",
    )
    parser.add_argument(
        "-c", "--concurrency", type=int, default=6,
        help="最大并发任务数（默认 6）",
    )
    args = parser.parse_args()
    asyncio.run(main(args.tasks, args.concurrency))
