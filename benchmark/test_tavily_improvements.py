"""Tavily 三项改进专项测试

验证:
  ① raw_content  — content_extractor 跳过爬虫的比例和内容质量
  ② tavily_score — source_evaluator 评分分布和筛选结果
  ③ query_answer — analyst 上下文中摘要的注入情况

用法:
  python -X utf8 benchmark/test_tavily_improvements.py "研究问题"
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
RED    = "\033[31m"
RESET  = "\033[0m"

def _b(t: str) -> str: return f"{BOLD}{t}{RESET}"
def _d(t: str) -> str: return f"{DIM}{t}{RESET}"
def _g(t: str) -> str: return f"{GREEN}{t}{RESET}"
def _y(t: str) -> str: return f"{YELLOW}{t}{RESET}"
def _c(t: str) -> str: return f"{CYAN}{t}{RESET}"
def _r(t: str) -> str: return f"{RED}{t}{RESET}"


def _build_state(query: str) -> dict:
    now = datetime.now().isoformat()
    return {
        "task_id": "test_tavily",
        "query": query,
        "language": "zh-CN",
        "max_rounds": 1,
        "current_round": 1,
        "status": "planning",
        "user_hints": {},
        "research_strategy": {},
        "research_plan": {},
        "sub_questions": [],
        "search_queries": [],
        "search_results": [],
        "search_summaries": [],
        "crawled_documents": [],
        "evaluated_sources": [],
        "evidence_chunks": [],
        "sub_answers": [],
        "fact_check_result": {},
        "fact_check_passed": True,
        "follow_up_queries": [],
        "final_report": "",
        "errors": [],
        "progress": 0,
        "progress_message": "",
        "created_at": now,
        "updated_at": now,
    }


async def main(query: str) -> None:
    from app.graph.nodes import planner_node, retriever_node, content_extractor_node, source_evaluator_node

    print(f"\n{'═' * 70}")
    print(f"  {_b('Tavily 三项改进专项测试')}")
    print(f"  {_d(query)}")
    print(f"{'═' * 70}")

    state = _build_state(query)

    # ── 1. Planner ────────────────────────────────────────────────────────────
    print(f"\n  {_d('① Planner ...')}", end="", flush=True)
    t0 = time.perf_counter()
    planner_out = await planner_node(state)
    state.update(planner_out)
    print(f"\r  {_g('✓')} Planner  {len(state['sub_questions'])} 个子问题 · {len(state['search_queries'])} 个查询  {_d(f'{time.perf_counter()-t0:.1f}s')}")

    # ── 2. Retriever ─────────────────────────────────────────────────────────
    print(f"  {_d('② Retriever ...')}", end="", flush=True)
    t0 = time.perf_counter()
    retriever_out = await retriever_node(state)
    state.update(retriever_out)
    elapsed = time.perf_counter() - t0

    results = state["search_results"]
    summaries = state["search_summaries"]
    n_raw = sum(1 for r in results if r.get("raw_content"))
    n_scored = sum(1 for r in results if r.get("tavily_score") is not None)
    print(f"\r  {_g('✓')} Retriever  {len(results)} 条结果  {_d(f'{elapsed:.1f}s')}")

    # ── 验证 ①：raw_content ─────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  {_b('① raw_content 覆盖率')}")
    print(f"{'─' * 70}")

    print(f"  结果总数:        {len(results)}")
    pct_raw = n_raw / len(results) * 100 if results else 0
    color = _g if pct_raw >= 80 else _y if pct_raw >= 50 else _r
    print(f"  含 raw_content: {color(f'{n_raw}/{len(results)} ({pct_raw:.0f}%)')}")

    # 展示内容长度分布
    raw_lens = [len(r["raw_content"]) for r in results if r.get("raw_content")]
    snip_lens = [len(r.get("snippet", "")) for r in results]
    if raw_lens:
        avg_raw = sum(raw_lens) / len(raw_lens)
        avg_snip = sum(snip_lens) / len(snip_lens) if snip_lens else 0
        print(f"  raw_content 均长:  {avg_raw:,.0f} chars")
        print(f"  snippet 均长:      {avg_snip:,.0f} chars")
        print(f"  信息量提升:        {_g(f'×{avg_raw/avg_snip:.1f}')} 倍")

    # ── 验证 ②：tavily_score ─────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  {_b('② tavily_score 分布')}")
    print(f"{'─' * 70}")

    scores = [r["tavily_score"] for r in results if r.get("tavily_score") is not None]
    print(f"  含 score 结果: {n_scored}/{len(results)}")
    if scores:
        print(f"  最高: {_g(f'{max(scores):.3f}')}  最低: {min(scores):.3f}  均值: {sum(scores)/len(scores):.3f}")

        # 按分数段统计
        bands = {"≥0.8": 0, "0.6-0.8": 0, "0.4-0.6": 0, "<0.4": 0}
        for s in scores:
            if s >= 0.8:     bands["≥0.8"] += 1
            elif s >= 0.6:   bands["0.6-0.8"] += 1
            elif s >= 0.4:   bands["0.4-0.6"] += 1
            else:            bands["<0.4"] += 1
        for band, cnt in bands.items():
            bar = "█" * cnt
            print(f"  {band:<10} {_c(bar):<30} {cnt}")

        # 对比旧评分（length/3000）
        print(f"\n  {_d('旧评分 vs 新评分（前5条）:')}")
        print(f"  {'标题':<35} {'旧(len/3k)':>10} {'新(score)':>10}")
        print(f"  {'─' * 57}")
        for r in results[:5]:
            old = min(1.0, len(r.get("snippet", "")) / 3000)
            new = r.get("tavily_score", 0)
            title = r.get("title", "")[:33]
            diff = new - old
            diff_str = f"{_g(f'+{diff:.2f}')}" if diff > 0.05 else f"{_r(f'{diff:.2f}')}" if diff < -0.05 else f"{diff:.2f}"
            print(f"  {title:<35} {old:>10.3f} {new:>10.3f}  {diff_str}")

    # ── 验证 ③：query_answer ─────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  {_b('③ search_summaries（Tavily 摘要答案）')}")
    print(f"{'─' * 70}")

    print(f"  摘要数量: {_g(str(len(summaries))) if summaries else _r('0')}")
    for s in summaries[:4]:
        sq_id = s.get("sq_id", "?")
        answer = s.get("answer", "")
        print(f"\n  [{_b(sq_id)}]")
        print(f"  {_d(answer[:200])}{'...' if len(answer) > 200 else ''}")

    # ── 3. Content Extractor ─────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  {_b('content_extractor_node 执行')}")
    print(f"{'─' * 70}")
    print(f"  {_d('③ 内容提取 ...')}", end="", flush=True)
    t0 = time.perf_counter()
    extractor_out = await content_extractor_node(state)
    state.update(extractor_out)
    elapsed_ext = time.perf_counter() - t0

    docs = state["crawled_documents"]
    n_from_tavily = sum(1 for d in docs if d.get("tavily_score") is not None or d.get("fetch_time") == 0.0)
    n_crawled = len(docs) - n_from_tavily
    valid = sum(1 for d in docs if d.get("content") and not d.get("error"))

    print(f"\r  {_g('✓')} 内容提取完成  {_d(f'{elapsed_ext:.1f}s')}")
    print(f"  来自 Tavily raw_content: {_g(str(n_from_tavily))} 条（0 次 HTTP）")
    print(f"  爬虫补充:                {n_crawled} 条")
    print(f"  有效文档:                {valid}/{len(docs)}")

    # ── 4. Source Evaluator ──────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  {_b('source_evaluator_node 执行')}")
    print(f"{'─' * 70}")
    print(f"  {_d('④ 信源评估 ...')}", end="", flush=True)
    t0 = time.perf_counter()
    evaluator_out = await source_evaluator_node(state)
    state.update(evaluator_out)
    elapsed_eval = time.perf_counter() - t0

    evaluated = state["evaluated_sources"]
    accepted = [e for e in evaluated if e.get("accepted")]
    rejected = [e for e in evaluated if not e.get("accepted")]

    print(f"\r  {_g('✓')} 信源评估完成  {_d(f'{elapsed_eval:.1f}s')}")
    print(f"  通过: {_g(str(len(accepted)))}  拒绝: {_r(str(len(rejected)))}")

    if accepted:
        acc_scores = [e["final_score"] for e in accepted]
        print(f"  通过分数范围: {min(acc_scores):.3f} - {max(acc_scores):.3f}（均值 {sum(acc_scores)/len(acc_scores):.3f}）")
    if rejected:
        print(f"  拒绝原因: {_d(', '.join(set(e.get('reason','')[:30] for e in rejected[:3])))}")

    # ── 最终摘要 ─────────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  {_b('改进效果摘要')}")
    print(f"{'═' * 70}")
    print(f"  raw_content 覆盖: {pct_raw:.0f}%（{n_raw}/{len(results)} 条，免爬取）")
    print(f"  content_extractor 耗时: {elapsed_ext:.1f}s（原本需 HTTP×{len(results)}）")
    sc_str = f"{len(scores)} 条结果有 Tavily 精确相关度分" if scores else "无 score 数据"
    print(f"  source_evaluator: {sc_str}")
    print(f"  search_summaries: {len(summaries)} 条摘要注入 analyst 上下文")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="大语言模型推理加速技术有哪些主要方向")
    args = parser.parse_args()
    asyncio.run(main(args.query))
