"""快速测试 Retriever 输出 — 直接调用 retriever_node（当前实现）

用法：
  python benchmark/test_retriever.py "你的研究问题"
  python benchmark/test_retriever.py "问题" -l en
  python benchmark/test_retriever.py "问题" --no-gantt
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


# ── 搜索调用日志（用于甘特图可视化）─────────────────────────────────────────────

_call_log: list[dict] = []
_run_t0: float = 0.0


def _install_patch() -> None:
    """Monkey-patch Provider.search，记录每次搜索的起止时间和结果数。"""
    import app.services.search_service as _ss

    _orig_ddg = _ss.DuckDuckGoProvider.search
    _orig_tav = _ss.TavilyProvider.search

    async def _wrap_ddg(self, query, max_results=5):
        t0 = time.perf_counter()
        result = await _orig_ddg(self, query, max_results)
        t1 = time.perf_counter()
        _call_log.append({"q": query[:50], "s": t0 - _run_t0, "e": t1 - _run_t0, "n": len(result)})
        return result

    async def _wrap_tav(self, query, max_results=5):
        t0 = time.perf_counter()
        result = await _orig_tav(self, query, max_results)
        t1 = time.perf_counter()
        _call_log.append({"q": query[:50], "s": t0 - _run_t0, "e": t1 - _run_t0, "n": len(result)})
        return result

    _ss.DuckDuckGoProvider.search = _wrap_ddg  # type: ignore[method-assign]
    _ss.TavilyProvider.search     = _wrap_tav  # type: ignore[method-assign]


# ── 甘特图 ────────────────────────────────────────────────────────────────────

def _gantt(log: list[dict], wall: float, width: int = 55) -> None:
    if not log:
        print(_d("    （无记录）"))
        return

    tick = max(1, width // 6)
    spc  = wall / width if wall > 0 else 1.0

    ruler     = "    " + "".join("|" if i % tick == 0 else "·" for i in range(width))
    label_row = "    "
    for i in range(0, width, tick):
        cell = f"{i * spc:.1f}s"
        label_row += cell[:tick].ljust(tick)
    print(_d(ruler))
    print(_d(label_row))

    peak = 0
    for ev in log:
        s = int(ev["s"] / wall * width) if wall > 0 else 0
        e = max(s + 1, int(ev["e"] / wall * width) if wall > 0 else 1)
        e = min(e, width)
        bar_ch = "█" if ev["n"] > 0 else "░"
        color  = CYAN if ev["n"] > 0 else RED
        bar    = " " * s + f"{color}{bar_ch * (e - s)}{RESET}" + " " * (width - e)
        n_str  = _g(f" {ev['n']}条") if ev["n"] > 0 else _r(" 0条")
        dur    = ev["e"] - ev["s"]
        label  = (ev["q"][:45] + "...") if len(ev["q"]) > 45 else ev["q"]
        print(f"    {_d(f'{label:<45}')}  {bar}  {n_str} {_d(f'{dur:.2f}s')}")

    times = sorted({
        round(ev["s"] + i * 0.1, 1)
        for ev in log
        for i in range(int((ev["e"] - ev["s"]) / 0.1) + 1)
    })
    for t in times:
        c = sum(1 for ev in log if ev["s"] <= t < ev["e"])
        peak = max(peak, c)
    print(f"\n    {_d('并发峰值:')} {_b(str(peak))} 个同时进行")


# ── 归属验证 ──────────────────────────────────────────────────────────────────

def _print_attribution(sub_questions: list[dict], results: list[dict]) -> None:
    counts  = {sq["id"]: 0 for sq in sub_questions}
    orphans = 0
    for r in results:
        sid = r.get("sub_question_id")
        if sid in counts:
            counts[sid] += 1
        else:
            orphans += 1

    for sq in sub_questions:
        sq_id = sq["id"]
        n     = counts[sq_id]
        bar   = _c("▓" * min(n, 18)) + _d("░" * max(0, 18 - n))
        ok    = _g("✓") if n > 0 else _r("✗")
        print(f"    {ok} [{_b(sq_id)}] {sq.get('angle', '')}  {bar}  {_b(str(n))} 条")

    if orphans:
        print(f"\n    {_y(f'⚠ {orphans} 条无有效归属')}")
    else:
        print(f"\n    {_g('✓ 全部结果均携带有效 sub_question_id')}")


# ── 状态构建（与 planner_benchmark.py 完全一致）────────────────────────────────

def _build_state(query: str, language: str) -> dict:
    now = datetime.now().isoformat()
    return {
        "task_id":           "test_retriever",
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


# ── 主流程 ────────────────────────────────────────────────────────────────────

async def run(query: str, language: str, show_gantt: bool) -> None:
    global _run_t0
    from app.graph.nodes import planner_node, retriever_node

    print(f"\n{'═' * 70}")
    print(f"  {_b('Retriever 输出测试（当前实现）')}")
    print(f"  {_d(query)}")
    print(f"{'═' * 70}")

    state = _build_state(query, language)

    # ── Planner ────────────────────────────────────────────────────────────
    print(f"\n  {_d('① Planner ...')}", end="", flush=True)
    tp   = time.perf_counter()
    pout = await planner_node(state)
    state.update(pout)
    sub_questions = state["sub_questions"]
    flat_queries  = [q for sq in sub_questions for q in sq.get("search_queries", [])]
    strategy      = pout.get("research_strategy", {})
    print(f"\r  {_g('✓')} Planner  "
          f"{strategy.get('intent', '—')} / {strategy.get('depth', '—')}  "
          f"{len(sub_questions)} 个子问题 · {len(flat_queries)} 个查询  "
          f"{_d(f'{time.perf_counter() - tp:.1f}s')}")

    # ── Retriever ──────────────────────────────────────────────────────────
    print(f"\n  {_d('② Retriever ...')}", end="", flush=True)
    _call_log.clear()
    _run_t0 = time.perf_counter()
    rout = await retriever_node(state)
    retriever_wall = time.perf_counter() - _run_t0
    state.update(rout)

    search_results   = state["search_results"]
    search_summaries = state["search_summaries"]
    n_raw   = sum(1 for r in search_results if r.get("raw_content"))
    raw_pct = round(n_raw / max(len(search_results), 1) * 100)

    print(f"\r  {_g('✓')} Retriever  "
          f"{len(search_results)} 条结果  "
          f"摘要 {len(search_summaries)} 条  "
          f"raw_content {raw_pct}%  "
          f"{_d(f'{retriever_wall:.1f}s')}")

    # ── 搜索时间线（甘特图）────────────────────────────────────────────────
    if show_gantt and _call_log:
        print(f"\n{'─' * 70}")
        print(f"  {_b('搜索时间线')}  （{_r('░=0条')} {_c('█=有结果')}）\n")
        _gantt(_call_log, retriever_wall)

    # ── sub_question_id 归属 ───────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  {_b('sub_question_id 归属')}  （{len(sub_questions)} 个子问题）\n")
    _print_attribution(sub_questions, search_results)

    # ── search_summaries ───────────────────────────────────────────────────
    if search_summaries:
        print(f"\n{'─' * 70}")
        print(f"  {_b('search_summaries（Tavily 摘要）')}  {len(search_summaries)} 条\n")
        for s in search_summaries[:4]:
            sq_id  = s.get("sq_id", "?")
            answer = s.get("answer", "")
            print(f"  [{_b(sq_id)}]")
            print(f"  {_d(answer[:200])}{'...' if len(answer) > 200 else ''}\n")

    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Retriever 输出测试（当前实现）")
    parser.add_argument("query", help="研究问题")
    parser.add_argument("-l", "--language", default="zh-CN")
    parser.add_argument("--no-gantt", action="store_true", help="不显示搜索时间线图")
    args = parser.parse_args()

    _install_patch()
    asyncio.run(run(args.query, args.language, show_gantt=not args.no_gantt))
