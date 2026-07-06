#!/usr/bin/env python3
"""Query Rewriting 实验 —— 黄金集上对比 Baseline vs Keyword Expansion。

用法:
  python rag_experiments/experiment_query_rewrite.py                          # 全部 12 任务
  python rag_experiments/experiment_query_rewrite.py --tasks 01 04 07         # 指定任务
  python rag_experiments/experiment_query_rewrite.py --save                   # 保存结果
  python rag_experiments/experiment_query_rewrite.py --schemes baseline       # 仅基线
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


# ── 指标工具 ──

def _cid(chunk: dict) -> str:
    return f"{chunk.get('source_id', '')}#{chunk.get('chunk_index', 0)}"

def _rank_of(hits: list[dict], gold: set[str]) -> int | None:
    for i, c in enumerate(hits, 1):
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

def _agg(records: list[dict], field: str, fn, *args) -> float:
    vals = [r[field] for r in records if r[field] is not None]
    return sum(fn(v, *args) for v in vals) / len(records) if vals else 0.0

def _collate(task_results: list[dict], field: str) -> list:
    return [r for t in task_results for r in t[field]]


# ── 模型预热 ──

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


# ── 索引 ──

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


# ── 方案 1：Baseline（标准检索，走生产 retrieve_evidence）──

async def _run_baseline(rag, qa: dict, task_id: str, retrieve_k: int, rerank_k: int) -> dict:
    from app.services.rag_service import rerank_chunks
    gold = set(qa["gold_cids"])
    vec = await rag.retrieve_evidence(query=qa["question"], task_id=task_id, top_k=retrieve_k)
    rer = await rerank_chunks(qa["question"], vec, top_k=rerank_k) if vec else []
    return {
        "q": qa["question"],
        "r_vec": _rank_of(vec, gold) if vec else None,
        "r_rer": _rank_of(rer, gold) if rer else None,
    }


# ── 方案 2：Keyword Expansion（改写优先 → xling 逐 query → 合并）──

async def _run_keyword(rag, qa: dict, task_id: str, retrieve_k: int, rerank_k: int) -> dict:
    from app.services.rag_service import rerank_chunks
    from rag_experiments.query_rewriter import rewrite_keywords, multi_query_retrieve

    gold = set(qa["gold_cids"])
    keywords = await rewrite_keywords(qa["question"], num_groups=4)
    vec = await multi_query_retrieve(
        rag, qa["question"], task_id, retrieve_k, extra_queries=keywords if keywords else None,
    )
    rer = await rerank_chunks(qa["question"], vec, top_k=rerank_k) if vec else []
    return {
        "q": qa["question"],
        "keywords": keywords,
        "r_vec": _rank_of(vec, gold) if vec else None,
        "r_rer": _rank_of(rer, gold) if rer else None,
    }


# ── 方案 3：HyDE（假设文档嵌入检索）──

async def _run_hyde(rag, qa: dict, task_id: str, retrieve_k: int, rerank_k: int) -> dict:
    from app.services.rag_service import rerank_chunks
    from rag_experiments.query_rewriter import rewrite_hyde, multi_query_retrieve

    gold = set(qa["gold_cids"])
    hyde = await rewrite_hyde(qa["question"])
    vec = await multi_query_retrieve(
        rag, qa["question"], task_id, retrieve_k, extra_queries=hyde if hyde else None,
    )
    rer = await rerank_chunks(qa["question"], vec, top_k=rerank_k) if vec else []
    return {
        "q": qa["question"],
        "hyde_passage": hyde[0] if hyde else "",
        "r_vec": _rank_of(vec, gold) if vec else None,
        "r_rer": _rank_of(rer, gold) if rer else None,
    }


# ── 方案 4: HyDE + Keyword 融合 ──

async def _run_hyde_keyword(rag, qa, task_id, retrieve_k, rerank_k):
    from app.services.rag_service import rerank_chunks
    from rag_experiments.query_rewriter import rewrite_hyde_keyword_fusion, multi_query_retrieve

    gold = set(qa["gold_cids"])
    extra = await rewrite_hyde_keyword_fusion(qa["question"], num_groups=4)
    vec = await multi_query_retrieve(
        rag, qa["question"], task_id, retrieve_k, extra_queries=extra if extra else None,
    )
    rer = await rerank_chunks(qa["question"], vec, top_k=rerank_k) if vec else []
    return {
        "q": qa["question"],
        "fusion_extra": extra,
        "r_vec": _rank_of(vec, gold) if vec else None,
        "r_rer": _rank_of(rer, gold) if rer else None,
    }


# ── 报告 ──

def _section(title: str):
    print(f"\n  {BOLD}{title}{RESET}")
    print(f"  {'─' * 80}")

def _scheme_label(scheme: str) -> str:
    return {"baseline": "Baseline", "keyword": "Keyword", "hyde": "HyDE",
            "hyde_keyword": "HyDE+KW"}.get(scheme, scheme)

def _scheme_extra_key(scheme: str) -> str:
    """方案报告中用于诊断的额外字段名（如 keywords/hyde_passage）。"""
    return {"keyword": "keywords", "hyde": "hyde_passage",
            "hyde_keyword": "fusion_extra"}.get(scheme, "")

def print_comparison(baseline_tasks: list[dict], scheme_tasks: list[dict],
                     scheme: str, retrieve_k: int, rerank_k: int) -> None:
    all_base = _collate(baseline_tasks, "records")
    all_sch = _collate(scheme_tasks, "records")
    label = _scheme_label(scheme)
    extra_key = _scheme_extra_key(scheme)

    print(f"\n{'═' * 100}")
    print(f"  {BOLD}Query Rewriting 检索对比{RESET}  "
          f"（向量 top-{retrieve_k} → rerank → top-{rerank_k}")
    print(f"  黄金集: {len(all_base)} 条黄金问题")
    print(f"  {label} vs Baseline")
    print(f"{'═' * 100}")

    # ── 逐任务 ──
    _section("逐任务 — Recall@k")
    hdr = (f"  {'ID':<4} {'任务名':<18} {'问题':>4}  "
           f"{'BaseR@6':>9} {f'{label[:4]}R@6':>9} {'Δ':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for tbl, tgg in zip(baseline_tasks, scheme_tasks):
        n = len(tbl["records"])
        base_r6 = _agg(tbl["records"], "r_rer", _recall, rerank_k)
        sch_r6 = _agg(tgg["records"], "r_rer", _recall, rerank_k)
        print(f"  {tbl['id']:<4} {tbl['name'][:16]:<18} {n:>4}  "
              f"  {_c(base_r6):>20}  {_c(sch_r6):>20} {_delta(sch_r6, base_r6)}")

    # ── 汇总 ──
    _section("汇总")
    base_vec_r = _agg(all_base, "r_vec", _recall, retrieve_k)
    base_rer_r = _agg(all_base, "r_rer", _recall, rerank_k)
    sch_vec_r = _agg(all_sch, "r_vec", _recall, retrieve_k)
    sch_rer_r = _agg(all_sch, "r_rer", _recall, rerank_k)

    base_mrr = _agg(all_base, "r_rer", lambda r: 1.0 / r if r else 0.0)
    sch_mrr = _agg(all_sch, "r_rer", lambda r: 1.0 / r if r else 0.0)
    base_ndcg = _agg(all_base, "r_rer", _ndcg, rerank_k)
    sch_ndcg = _agg(all_sch, "r_rer", _ndcg, rerank_k)

    print(f"  {'指标':<20}  {'Baseline':>12}  {label:>12}  {'Δ':>8}")
    print(f"  {'─' * 56}")
    print(f"  {'向量原始':<16}  Recall@{retrieve_k:<4}  {_c(base_vec_r):>12}  "
          f"{_c(sch_vec_r):>12}  {_delta(sch_vec_r, base_vec_r)}")
    print(f"  {'rerank 后':<16}  Recall@{rerank_k:<4}  {_c(base_rer_r):>12}  "
          f"{_c(sch_rer_r):>12}  {_delta(sch_rer_r, base_rer_r)}")
    print(f"  {'MRR':<20}  {'':>12}  {_c(base_mrr):>12}  "
          f"{_c(sch_mrr):>12}  {_delta(sch_mrr, base_mrr)}")
    print(f"  {'NDCG@' + str(rerank_k):<19}   {'':>12}  {_c(base_ndcg):>12}  "
          f"{_c(sch_ndcg):>12}  {_delta(sch_ndcg, base_ndcg)}")

    # ── 诊断 ──
    _section("诊断")
    vector_miss = 1.0 - base_vec_r
    rerank_loss = base_vec_r - base_rer_r
    improvement = sch_rer_r - base_rer_r
    print(f"    向量漏检:         {RED}{vector_miss:.1%}{RESET}")
    print(f"    rerank 截断损失:  {YELLOW}{rerank_loss:.1%}{RESET}")
    if improvement > 0:
        print(f"    {label} 提升:      {GREEN}+{improvement:.1%}{RESET}")
    elif improvement < 0:
        print(f"    {label} 退化:      {RED}{improvement:.1%}{RESET}")
    else:
        print(f"    {label} 变化:      {DIM}0{RESET}")

    # 命中变化明细（仅对比有 baseline 记录的问题）
    n_common = min(len(all_base), len(all_sch))
    improved_q = [
        (i, r) for i, r in enumerate(all_sch[:n_common])
        if r["r_rer"] is not None
        and (all_base[i]["r_rer"] is None or r["r_rer"] < all_base[i]["r_rer"])
    ]
    harmed_q = [
        i for i in range(n_common)
        if all_sch[i]["r_rer"] is None and all_base[i]["r_rer"] is not None
    ]
    if improved_q:
        print(f"\n  {label} 改善（{len(improved_q)} 条）:")
        for i, r in improved_q[:5]:
            base_r = all_base[i]["r_rer"]
            extra = f", {extra_key}: {r.get(extra_key, '')[:70]}" if extra_key else ""
            print(f"    {GREEN}↑{RESET} {r['q'][:60]}  {DIM}rank {base_r}→{r['r_rer']}{extra}{RESET}")
        if len(improved_q) > 5:
            print(f"    ... 还有 {len(improved_q) - 5} 条")
    if harmed_q:
        print(f"\n  {label} 退化（{len(harmed_q)} 条）:")
        for i in harmed_q[:3]:
            base_r = all_base[i]["r_rer"]
            r = all_sch[i]
            print(f"    {RED}↓{RESET} {r['q'][:60]}  {DIM}rank {base_r}→miss{RESET}")


# ── 主流程 ──

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
    print("  Query Rewriting 实验 — 黄金集检索质量对比")
    print(f"  黄金集: {golden_name}  ({data.get('built_at','?')[:19]})")
    print(f"  embedding={settings.embedding_model}  retrieve_k={retrieve_k}  "
          f"rerank={'on' if settings.reranker_enabled else 'off'}  "
          f"xling={'on' if settings.xling_enabled else 'off'}")
    print(f"  方案: {', '.join(schemes)}")
    print("═" * 72)

    rag = await _warmup()
    t0 = time.perf_counter()

    # ── 方案调度映射 ──
    RUNNERS = {
        "baseline": _run_baseline,
        "keyword": _run_keyword,
        "hyde": _run_hyde,
        "hyde_keyword": _run_hyde_keyword,
    }
    active_schemes = {s: RUNNERS[s] for s in schemes if s in RUNNERS}
    results: dict[str, list[dict]] = {s: [] for s in active_schemes}

    for task in tasks:
        tid = task["id"]
        tname = task["name"]
        task_id = task["task_id"]

        print(f"\n[{tid}] {tname} — 索引语料...", end=" ", flush=True)
        n_corpus = await _index_corpus(rag, task_id, task["corpus"])
        print(f"{n_corpus} chunks", flush=True)

        for sname, runner in active_schemes.items():
            t1 = time.perf_counter()
            records = await asyncio.gather(*[
                runner(rag, qa, task_id, retrieve_k, rerank_k)
                for qa in task["qa"]
            ])
            dt = time.perf_counter() - t1
            r6 = _agg(records, "r_rer", _recall, rerank_k)
            results[sname].append({
                "id": tid, "name": tname, "records": records,
            })
            extra = ""
            if sname == "keyword":
                ok = sum(1 for r in records if r.get("keywords"))
                extra = f", {ok}/{len(records)} 成功改写"
            elif sname == "hyde":
                ok = sum(1 for r in records if r.get("hyde_passage"))
                extra = f", {ok}/{len(records)} 成功生成"
            elif sname == "hyde_keyword":
                ok = sum(1 for r in records if r.get("fusion_extra"))
                avg = int(sum(len(r.get("fusion_extra", [])) for r in records) / max(len(records), 1))
                extra = f", {ok}/{len(records)} 成功, 均{avg}条"
            print(f"  {DIM}{sname}: R@{rerank_k}={r6:.3f}  ({dt:.1f}s{extra}){RESET}", flush=True)

    # 输出对比
    base_tasks = results.get("baseline", [])
    for sname, stasks in results.items():
        if sname == "baseline":
            continue
        if base_tasks and stasks:
            print_comparison(base_tasks, stasks, sname, retrieve_k, rerank_k)

    if not base_tasks:
        # 无 baseline：各方案独立汇总
        for sname, stasks in results.items():
            all_r = _collate(stasks, "records")
            r6 = _agg(all_r, "r_rer", _recall, rerank_k)
            print(f"\n  {_scheme_label(sname)} 完成: R@{rerank_k}={r6:.3f}")

    total = time.perf_counter() - t0
    print(f"\n  总耗时: {total:.1f}s")

    # 保存
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_data = {
            "config": {
                "embedding_model": settings.embedding_model,
                "retrieve_k": retrieve_k,
                "rerank_k": rerank_k,
                "rerank_enabled": settings.reranker_enabled,
                "xling_enabled": settings.xling_enabled,
            },
            "schemes": schemes,
            "tasks": [t["id"] for t in tasks],
            "elapsed_seconds": round(total, 1),
        }
        for sname, stasks in results.items():
            out_data[sname] = [{"id": t["id"], "records": t["records"]} for t in stasks]
        out = RESULTS_DIR / f"experiment_query_rewrite_{ts}.json"
        out.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  已保存: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Rewriting 检索对比实验")
    parser.add_argument("--tasks", nargs="+", default=[],
                        help="任务 ID（默认全部）")
    parser.add_argument("--golden", default="golden_set.json",
                        help="黄金集文件名")
    parser.add_argument("--schemes", nargs="+", default=["baseline", "keyword"],
                        choices=["baseline", "keyword", "hyde", "hyde_keyword"],
                        help="要跑的方案")
    parser.add_argument("--save", action="store_true", help="保存结果 JSON")
    args = parser.parse_args()

    asyncio.run(main(args.golden, args.tasks, args.schemes, args.save))
