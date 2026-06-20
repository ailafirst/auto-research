"""快速测试 Planner 输出

直接调用 planner_node，打印完整的规划结果，不做评分对比。

用法：
  python benchmark/test_planner.py "你的研究问题"
  python benchmark/test_planner.py "问题1" "问题2" "问题3"
  python benchmark/test_planner.py -l en "your research question"
"""

from __future__ import annotations

import asyncio
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

GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def _b(t: str) -> str: return f"{BOLD}{t}{RESET}"
def _d(t: str) -> str: return f"{DIM}{t}{RESET}"
def _c(t: str) -> str: return f"{CYAN}{t}{RESET}"
def _g(t: str) -> str: return f"{GREEN}{t}{RESET}"
def _y(t: str) -> str: return f"{YELLOW}{t}{RESET}"


def _build_state(query: str, language: str) -> dict:
    now = datetime.now().isoformat()
    return {
        "task_id":           "test_planner",
        "query":             query,
        "language":          language,
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


def print_planner_output(query: str, output: dict, elapsed: float) -> None:
    strategy = output.get("research_strategy", {})
    plan     = output.get("research_plan", {})
    sub_qs   = output.get("sub_questions", [])

    intent    = strategy.get("intent", "—")
    domain    = strategy.get("domain", "—")
    depth     = strategy.get("depth", "—")
    dims      = strategy.get("dimensions", [])
    reasoning = strategy.get("reasoning", "")
    goal      = plan.get("research_goal", "")

    depth_color = {"shallow": _g, "medium": _y, "deep": _c}.get(depth, _b)

    print(f"\n{'═' * 70}")
    print(f"  {_b(query)}")
    print(f"{'─' * 70}")
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
        sq_id     = sq.get("id", "")
        angle     = sq.get("angle", "")
        question  = sq.get("question", "")
        queries   = sq.get("search_queries", [])
        priority  = sq.get("priority", "")

        print(f"  {_b(f'[{sq_id}]')} {_c(angle)}  {_d(f'priority={priority}')}")
        print(f"       {question}")
        for q in queries:
            print(f"       {_d('›')} {q}")
        print()

    n_queries = len(output.get("search_queries", []))
    print(f"  {_d(f'耗时 {elapsed:.1f}s · 搜索词共 {n_queries} 条')}")


async def run_query(query: str, language: str) -> None:
    from app.graph.nodes import planner_node

    state = _build_state(query, language)
    start = time.perf_counter()
    try:
        output = await planner_node(state)
        elapsed = time.perf_counter() - start
        print_planner_output(query, output, elapsed)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        print(f"\n  ERROR ({elapsed:.1f}s): {exc}")


async def main(queries: list[str], language: str) -> None:
    for query in queries:
        await run_query(query, language)
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="快速测试 Planner 输出")
    parser.add_argument("queries", nargs="+", help="一个或多个研究问题")
    parser.add_argument("-l", "--language", default="zh-CN", help="输出语言（默认 zh-CN）")
    args = parser.parse_args()

    asyncio.run(main(args.queries, args.language))
