#!/usr/bin/env python3
"""RAG 实验编排 —— 在黄金集上对 baseline / GraphRAG-V 做确定性检索质量对比。

用法:
  python rag_experiments/run.py                          # 全部 12 任务，两个方案
  python rag_experiments/run.py --tasks 01 04 07         # 指定任务
  python rag_experiments/run.py --schemes graph_v        # 仅 GraphRAG-V
  python rag_experiments/run.py --save                   # 保存结果到 results/
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import sys
import time
from pathlib import Path

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

GOLDEN_DIR = ROOT / "benchmark" / "GoldenDataset"
RESULTS_DIR = Path(__file__).parent / "results"
GREEN = "\033[32m"; YELLOW = "\033[33m"; RED = "\033[31m"
BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _cid(chunk: dict) -> str:
    return f"{chunk.get('source_id', '')}#{chunk.get('chunk_index', 0)}"

def _rank_of(ranked: list[dict], gold: set[str]) -> int | None:
    for i, c in enumerate(ranked, 1):
        if _cid(c) in gold:
            return i
    return None

def _recall(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0

def _ndcg(rank: int | None, k: int) -> float:
    return 1.0 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0

def _c(v: float, good: float = 0.7, mid: float = 0.5) -> str:
    s = f"{v:.3f}"
    return (GREEN if v >= good else (YELLOW if v >= mid else RED)) + s + RESET

def _delta(new: float, base: float) -> str:
    d = new - base
    if abs(d) < 0.001:
        return f"  {DIM}—{RESET}"
    c = GREEN if d > 0 else RED
    return f" {c}{d:+.3f}{RESET}"


# ── 模型预热 ───────────────────────────────────────────────────────────────────

async def _warmup():
    from app.core.config import settings as _s
    from app.services.rag_service import _get_reranker, get_rag_service
    t0 = time.perf_counter()
    print(f"  {DIM}预热模型...{RESET}", flush=True)
    rag = get_rag_service()
    if _s.embedding_provider == "st":
        await rag._get_st_model()
    elif _s.embedding_provider == "fastembed":
        await rag._get_fastembed_model()
    if _s.reranker_enabled:
        await _get_reranker()
    if _s.xling_enabled:
        from app.services.translation_service import get_translation_service
        await get_translation_service().translate("预热")
    print(f"  {DIM}预热完成（{time.perf_counter() - t0:.1f}s）{RESET}", flush=True)
    return rag


# ── 索引（同 eval_retrieval）───────────────────────────────────────────────────

async def _index_corpus(rag, task_id: str, corpus: list[dict]) -> int:
    from app.models.source import EvidenceChunk
    chunks = [
        EvidenceChunk(
            task_id=task_id, source_id=c["source_id"], url=c["url"],
            title=c["title"], chunk_index=c["chunk_index"], text=c["text"],
        )
        for c in corpus
    ]
    texts = [f"{c.title}\n\n{c.text}" if c.title else c.text for c in chunks]
    embs = await rag._embed(texts)
    valid = [(c, e) for c, e in zip(chunks, embs)
             if e and not any(math.isnan(v) for v in e)]
    if not valid:
        return 0
    vc, ve = zip(*valid)
    await rag.vector_store.store_chunks(list(vc), list(ve))
    return len(vc)


# ── 方案 1：Baseline（标准检索）───────────────────────────────────────────────

async def _run_baseline(rag, qa: dict, task_id: str, retrieve_k: int, rerank_k: int) -> dict:
    from app.services.rag_service import rerank_chunks
    gold = set(qa["gold_cids"])
    vec = await rag.retrieve_evidence(query=qa["question"], task_id=task_id, top_k=retrieve_k)
    rer = await rerank_chunks(qa["question"], vec, top_k=rerank_k) if vec else []
    r_vec = _rank_of(vec, gold) if vec else None
    r_rer = _rank_of(rer, gold) if rer else None
    return {"q": qa["question"], "r_vec": r_vec, "r_rer": r_rer}


# ── 方案 2：GraphRAG-V（社区扩张）─────────────────────────────────────────────

async def _run_graph_v(rag, qa: dict, task_id: str, retrieve_k: int, rerank_k: int,
                       graph) -> dict:
    from app.services.rag_service import rerank_chunks
    gold = set(qa["gold_cids"])

    # 1. 标准向量检索
    vec = await rag.retrieve_evidence(query=qa["question"], task_id=task_id, top_k=retrieve_k)

    # 2. rerank → top-k（同 baseline）
    rer = await rerank_chunks(qa["question"], vec, top_k=rerank_k) if vec else []
    vec_by_key = {_cid(c): c for c in vec}
    top_keys = {_cid(c) for c in rer}

    # 3. 查 top-k 各属哪个社区，收集邻居（含远邻 — 不在 vec 中）
    extra: list[dict] = []
    seen_extra: set[str] = set()

    for ck in top_keys:
        if len(extra) >= 4:
            break
        cid = graph.key_to_community.get(ck)
        if cid is None:
            continue
        for comm in graph.communities:
            if comm["id"] != cid:
                continue
            for mk in comm["keys"]:
                if mk in top_keys or mk in seen_extra:
                    continue
                seen_extra.add(mk)
                # 近邻：在 vec（top-40）中
                if mk in vec_by_key:
                    extra.append(vec_by_key[mk])
                # 远邻：不在 vec 中，从 chunk_lookup 捞
                elif mk in graph.chunk_lookup:
                    cd = graph.chunk_lookup[mk]
                    extra.append({
                        "score": 0.0,
                        "text": cd["text"],
                        "url": cd["url"],
                        "title": cd["title"],
                        "source_id": cd["source_id"],
                        "chunk_index": cd["chunk_index"],
                    })
                if len(extra) >= 4:
                    break
            break

    # 4. 扩张池 = rer + extra，再 rerank 取 top-k
    expanded = rer + extra
    re_rer = await rerank_chunks(qa["question"], expanded, top_k=rerank_k) if expanded else []

    r_vec = _rank_of(vec, gold) if vec else None
    r_rer = _rank_of(rer, gold) if rer else None
    r_graph = _rank_of(re_rer, gold) if re_rer else None

    return {"q": qa["question"], "r_vec": r_vec, "r_rer": r_rer, "r_graph": r_graph}


# ── 合拢结果 ───────────────────────────────────────────────────────────────────

def _agg(records: list[dict], field: str, fn, *args) -> float:
    return sum(fn(r[field], *args) for r in records) / len(records) if records else 0.0


def _collate(task_results: list[dict], field: str) -> list:
    return [r for t in task_results for r in t["records"]]


# ── 报告 ───────────────────────────────────────────────────────────────────────

def _section(title: str):
    print(f"\n  {BOLD}{title}{RESET}")
    print(f"  {'─' * 80}")


def print_comparison(baseline_tasks: list[dict], graph_tasks: list[dict],
                     retrieve_k: int, rerank_k: int) -> None:
    all_base = _collate(baseline_tasks, "records")
    all_graph = _collate(graph_tasks, "records")

    has_graph = bool(all_graph and "r_graph" in all_graph[0])

    print(f"\n{'═' * 100}")
    print(f"  {BOLD}RAG 方案对比{RESET}  "
          f"（向量 top-{retrieve_k} → rerank → top-{rerank_k}")
    print(f"  黄金集: {len(all_base)} 条黄金问题")
    print(f"{'═' * 100}")

    # ── 逐任务 ──
    _section("逐任务 — Recall@k")
    hdr = (f"  {'ID':<4} {'任务名':<18} {'问题':>4}  "
           f"{'BaseR@6':>9} {'GraphR@6':>9} {'Δ':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for tbl, tgg in zip(baseline_tasks, graph_tasks):
        n = len(tbl["records"])
        base_r6 = _agg(tbl["records"], "r_rer", _recall, rerank_k)
        if has_graph:
            graph_r6 = _agg(tgg["records"], "r_graph", _recall, rerank_k)
        else:
            graph_r6 = base_r6
        print(f"  {tbl['id']:<4} {tbl['name'][:16]:<18} {n:>4}  "
              f"  {_c(base_r6):>20}  {_c(graph_r6):>20} {_delta(graph_r6, base_r6)}")

    # ── 汇总 ──
    _section("汇总")
    base_vec_r = _agg(all_base, "r_vec", _recall, retrieve_k)
    base_rer_r = _agg(all_base, "r_rer", _recall, rerank_k)
    graph_vec_r = _agg(all_graph, "r_vec", _recall, retrieve_k) if has_graph else base_vec_r
    graph_rer_r = _agg(all_graph, "r_rer", _recall, rerank_k) if has_graph else base_rer_r
    graph_final_r = _agg(all_graph, "r_graph", _recall, rerank_k) if has_graph else graph_rer_r

    base_mrr = _agg(all_base, "r_rer", lambda r: 1.0 / r if r else 0.0)
    graph_mrr = _agg(all_graph, "r_graph", lambda r: 1.0 / r if r else 0.0) if has_graph else base_mrr
    base_ndcg = _agg(all_base, "r_rer", _ndcg, rerank_k)
    graph_ndcg = _agg(all_graph, "r_graph", _ndcg, rerank_k) if has_graph else base_ndcg

    print(f"  {'指标':<20}  {'Baseline':>12}  {'GraphRAG-V':>12}  {'Δ':>8}")
    print(f"  {'─' * 56}")
    print(f"  {'向量召回上限':<14}  Recall@{retrieve_k:<4}  {_c(base_vec_r):>12}  "
          f"{_c(graph_vec_r):>12}  {_delta(graph_vec_r, base_vec_r)}")
    print(f"  {'rerank 后':<16}  Recall@{rerank_k:<4}  {_c(base_rer_r):>12}  "
          f"{_c(graph_rer_r):>12}  {_delta(graph_rer_r, base_rer_r)}")
    print(f"  {'社区增强后':<16}  Recall@{rerank_k:<4}  {'—':>12}  "
          f"{_c(graph_final_r):>12}  {_delta(graph_final_r, base_rer_r)}")
    print(f"  {'MRR':<20}  {'':>12}  {_c(base_mrr):>12}  "
          f"{_c(graph_mrr):>12}  {_delta(graph_mrr, base_mrr)}")
    print(f"  {'NDCG@' + str(rerank_k):<19}   {'':>12}  {_c(base_ndcg):>12}  "
          f"{_c(graph_ndcg):>12}  {_delta(graph_ndcg, base_ndcg)}")

    # ── 诊断 ──
    _section("诊断")
    vector_miss = 1.0 - base_vec_r
    rerank_loss = base_vec_r - base_rer_r
    graph_recover = graph_final_r - base_rer_r
    print(f"    向量漏检:         {RED}{vector_miss:.1%}{RESET}")
    print(f"    rerank 截断损失:  {YELLOW}{rerank_loss:.1%}{RESET}")
    if graph_recover > 0:
        print(f"    社区回捞:         {GREEN}+{graph_recover:.1%}{RESET}"
              f"  {DIM}(rerank 截断损失中恢复了 {graph_recover/rerank_loss:.0%}){RESET}" if rerank_loss > 0 else "")
    elif graph_recover < 0:
        print(f"    社区退化:         {RED}{graph_recover:.1%}{RESET}")

    # 漏检清单
    misses = [r for r in all_base if r["r_vec"] is None]
    if misses:
        print(f"\n  向量硬漏检（{len(misses)} 条）:")
        for r in misses[:5]:
            print(f"    {RED}✗{RESET} {r['q'][:60]}")
        if len(misses) > 5:
            print(f"    ... 还有 {len(misses) - 5} 条")

    dropped = [r for r in all_graph if r.get("r_rer") is not None and r.get("r_graph") != r["r_rer"]]
    improved = [r for r in dropped if r.get("r_graph") is not None and r["r_rer"] is not None
                and r["r_graph"] < r["r_rer"]]
    if improved:
        print(f"\n  社区扩张改善了（{len(improved)} 条）:")
        for r in improved[:3]:
            print(f"    {GREEN}↑{RESET} {r['q'][:60]}  {DIM}rank {r['r_rer']}→{r['r_graph']}{RESET}")
        if len(improved) > 3:
            print(f"    ... 还有 {len(improved) - 3} 条")


# ── 主流程 ─────────────────────────────────────────────────────────────────────

async def main(golden_name: str, task_ids: list[str], schemes: list[str],
               save: bool) -> None:
    from app.core.config import settings
    from app.services.rag_service import get_rag_service

    # 加载黄金集
    path = GOLDEN_DIR / golden_name
    if not path.exists():
        print(f"黄金集不存在: {path}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data["tasks"]
    if task_ids:
        tasks = [t for t in tasks if t["id"] in task_ids]
    if not tasks:
        print("无有效任务", file=sys.stderr)
        sys.exit(1)

    retrieve_k = settings.reranker_retrieve_k if settings.reranker_enabled else settings.rag_top_k
    rerank_k = settings.reranker_top_k

    print("═" * 72)
    print("  RAG 实验对比 — 黄金集检索质量评测")
    print(f"  黄金集: {golden_name}  ({data.get('built_at','?')[:19]})")
    print(f"  embedding={settings.embedding_model}  retrieve_k={retrieve_k}  "
          f"rerank={'on' if settings.reranker_enabled else 'off'}  "
          f"xling={'on' if settings.xling_enabled else 'off'}")
    print(f"  方案: {', '.join(schemes)}")
    print("═" * 72)

    rag = await _warmup()
    t0 = time.perf_counter()

    run_baseline = "baseline" in schemes
    run_graph_v = "graph_v" in schemes

    baseline_tasks: list[dict] = []
    graph_tasks: list[dict] = []

    for task in tasks:
        tid = task["id"]
        tname = task["name"]
        task_id = task["task_id"]

        print(f"\n[{tid}] {tname} — 索引语料...", end=" ", flush=True)
        n_corpus = await _index_corpus(rag, task_id, task["corpus"])
        print(f"{n_corpus} chunks", flush=True)

        # Baseline
        if run_baseline:
            t1 = time.perf_counter()
            records = await asyncio.gather(*[
                _run_baseline(rag, qa, task_id, retrieve_k, rerank_k)
                for qa in task["qa"]
            ])
            dt = time.perf_counter() - t1
            r6 = _agg(records, "r_rer", _recall, rerank_k)
            baseline_tasks.append({
                "id": tid, "name": tname, "records": records,
            })
            print(f"  {DIM}baseline: R@{rerank_k}={r6:.3f}  ({dt:.1f}s){RESET}", flush=True)

        # GraphRAG-V
        if run_graph_v:
            from rag_experiments.graph_builder import build_graph

            t1 = time.perf_counter()
            graph = await build_graph(rag.vector_store, task_id)
            dt_graph = time.perf_counter() - t1

            if graph is None:
                print(f"  {YELLOW}graph_v: 跳過（建图失败, {n_corpus} chunks）{RESET}", flush=True)
                records = []
                for qa in task["qa"]:
                    from app.services.rag_service import rerank_chunks
                    gold = set(qa["gold_cids"])
                    vec = await rag.retrieve_evidence(
                        query=qa["question"], task_id=task_id, top_k=retrieve_k,
                    )
                    rer = await rerank_chunks(qa["question"], vec, top_k=rerank_k) if vec else []
                    r_rer = _rank_of(rer, gold) if rer else None
                    records.append({
                        "q": qa["question"], "r_vec": _rank_of(vec, gold) if vec else None,
                        "r_rer": r_rer, "r_graph": r_rer,
                    })
                graph_tasks.append({"id": tid, "name": tname, "records": records})
                continue

            t2 = time.perf_counter()
            records = await asyncio.gather(*[
                _run_graph_v(rag, qa, task_id, retrieve_k, rerank_k, graph)
                for qa in task["qa"]
            ])
            dt_query = time.perf_counter() - t2
            r6 = _agg(records, "r_graph", _recall, rerank_k)
            print(f"  {DIM}graph_v: 建图{dt_graph:.2f}s+检索{dt_query:.1f}s  "
                  f"R@{rerank_k}={r6:.3f}{RESET}", flush=True)
            graph_tasks.append({"id": tid, "name": tname, "records": records})

    # 输出对比
    if run_baseline and run_graph_v and baseline_tasks and graph_tasks:
        print_comparison(baseline_tasks, graph_tasks, retrieve_k, rerank_k)
    elif run_baseline:
        all_r = _collate(baseline_tasks, "records")
        r6 = _agg(all_r, "r_rer", _recall, rerank_k)
        print(f"\n  Baseline 完成: R@{rerank_k}={r6:.3f}")
    elif run_graph_v:
        all_r = _collate(graph_tasks, "records")
        r6 = _agg(all_r, "r_graph", _recall, rerank_k)
        print(f"\n  GraphRAG-V 完成: R@{rerank_k}={r6:.3f}")

    total = time.perf_counter() - t0
    print(f"\n  总耗时: {total:.1f}s")

    # 保存
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        result = {
            "config": {
                "embedding_model": settings.embedding_model,
                "retrieve_k": retrieve_k,
                "rerank_k": rerank_k,
                "rerank_enabled": settings.reranker_enabled,
                "xling_enabled": settings.xling_enabled,
            },
            "schemes": schemes,
            "tasks": [t["id"] for t in tasks],
            "baseline": [{"id": t["id"], "records": t["records"]} for t in baseline_tasks],
            "graph_v": [{"id": t["id"], "records": t["records"]} for t in graph_tasks],
            "elapsed_seconds": round(total, 1),
        }
        out = RESULTS_DIR / f"experiment_{ts}.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  已保存: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 实验对比")
    parser.add_argument("--tasks", nargs="+", default=[],
                        help="任务 ID（默认全部）")
    parser.add_argument("--golden", default="golden_set.json",
                        help="黄金集文件名（benchmark/GoldenDataset/）")
    parser.add_argument("--schemes", nargs="+", default=["baseline", "graph_v"],
                        choices=["baseline", "graph_v"],
                        help="要跑的实验方案")
    parser.add_argument("--save", action="store_true", help="保存结果 JSON")
    args = parser.parse_args()

    asyncio.run(main(args.golden, args.tasks, args.schemes, args.save))
