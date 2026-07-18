#!/usr/bin/env python3
"""FP32 vs FP16 检索质量对比 —— 120 题黄金集。

只改「模型精度」这一个变量，其余（chunk、retrieve_k、rerank_k、xling）全部保持
生产配置不变，评测口径与 rag_experiments/run.py 完全一致（Recall@k / NDCG@k）。

每个精度必须跑在独立进程里：fp16 的 embedding 与 fp32 不同，向量索引要整个重建，
且模型是进程级单例、无法在进程内换精度。

用法:
  python rag_experiments/experiment_fp16.py --dtype fp32 --save     # 跑基线
  python rag_experiments/experiment_fp16.py --dtype fp16 --save     # 跑 fp16
  python rag_experiments/experiment_fp16.py --compare               # 对比最近两次
  python rag_experiments/experiment_fp16.py --both                  # 自动依次跑两次再对比
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

GOLDEN = ROOT / "benchmark" / "GoldenDataset" / "golden_set.json"
RESULTS_DIR = Path(__file__).parent / "results"
GREEN = "\033[32m"; YELLOW = "\033[33m"; RED = "\033[31m"
BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"


# ── 评测口径（与 run.py 一致）─────────────────────────────────────────────────

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


def _mrr(rank: int | None) -> float:
    return 1.0 / rank if rank is not None else 0.0


def _agg(records: list[dict], field: str, fn, *a) -> float:
    if not records:
        return 0.0
    return sum(fn(r[field], *a) for r in records) / len(records)


def _vram_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 2 ** 20
    except Exception:
        pass
    return 0.0


# ── 单精度跑一遍 ──────────────────────────────────────────────────────────────

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
    valid = [(c, e) for c, e in zip(chunks, embs) if e and not any(math.isnan(v) for v in e)]
    if not valid:
        return 0
    vc, ve = zip(*valid)
    await rag.vector_store.store_chunks(list(vc), list(ve))
    return len(vc)


async def run_one(dtype: str, task_filter: list[str] | None) -> dict:
    # 模型是懒加载的单例：只要在首次 _get_st_model()/_get_reranker() 之前改掉
    # settings.model_dtype 就会生效；加载之后再改无效（故每个精度需独立进程）。
    from app.core.config import settings
    settings.model_dtype = dtype  # type: ignore[assignment]

    from app.services.rag_service import _get_reranker, get_rag_service, rerank_chunks

    retrieve_k = settings.reranker_retrieve_k
    rerank_k = settings.reranker_top_k

    print(f"\n{BOLD}{'=' * 84}{RESET}")
    print(f"{BOLD}精度 {dtype.upper()}{RESET}   "
          f"embed={settings.embedding_model}  rerank={settings.reranker_model}")
    print(f"检索链路: 向量 top-{retrieve_k} → rerank → top-{rerank_k}   "
          f"xling={settings.xling_enabled}")
    print(f"{BOLD}{'=' * 84}{RESET}")

    t0 = time.perf_counter()
    rag = get_rag_service()
    await rag._get_st_model()
    vram_embed = _vram_mb()
    if settings.reranker_enabled:
        await _get_reranker()
    vram_all = _vram_mb()
    if settings.xling_enabled:
        from app.services.translation_service import get_translation_service
        await get_translation_service().translate("预热")
    load_s = time.perf_counter() - t0

    # 确认精度真的生效了，别测了个寂寞
    import torch
    p = next(rag._embedding_model.parameters())
    print(f"  {DIM}模型加载 {load_s:.1f}s | embed 权重 dtype={p.dtype} | "
          f"VRAM embed={vram_embed:.0f} MB, +reranker={vram_all:.0f} MB{RESET}")
    expect = torch.float16 if dtype == "fp16" else torch.float32
    if p.dtype != expect:
        print(f"  {RED}警告: 期望 {expect} 实际 {p.dtype} —— 精度未生效！{RESET}")

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    tasks = golden["tasks"]
    if task_filter:
        tasks = [t for t in tasks if t["id"] in task_filter]

    per_task, all_records = [], []
    t_embed_total = t_query_total = 0.0

    for t in tasks:
        tid = f"fp16exp_{dtype}_{t['id']}"
        te = time.perf_counter()
        n = await _index_corpus(rag, tid, t["corpus"])
        t_embed = time.perf_counter() - te
        t_embed_total += t_embed

        recs = []
        tq = time.perf_counter()
        for qa in t["qa"]:
            gold = set(qa["gold_cids"])
            vec = await rag.retrieve_evidence(
                query=qa["question"], task_id=tid, top_k=retrieve_k,
            )
            rer = await rerank_chunks(qa["question"], vec, top_k=rerank_k) if vec else []
            recs.append({
                "q": qa["question"],
                "r_vec": _rank_of(vec, gold) if vec else None,
                "r_rer": _rank_of(rer, gold) if rer else None,
            })
        t_query = time.perf_counter() - tq
        t_query_total += t_query

        r6 = _agg(recs, "r_rer", _recall, rerank_k)
        rv = _agg(recs, "r_vec", _recall, retrieve_k)
        per_task.append({
            "id": t["id"], "name": t["name"], "n_chunks": n, "n_qa": len(recs),
            "recall_rerank": r6, "recall_vec": rv,
            "ndcg": _agg(recs, "r_rer", _ndcg, rerank_k),
            "mrr": _agg(recs, "r_rer", _mrr),
            "embed_s": round(t_embed, 1), "query_s": round(t_query, 1),
            "records": recs,
        })
        all_records.extend(recs)
        print(f"  {t['id']} {t['name']:<18} chunks={n:>5}  "
              f"Recall@{rerank_k}={r6:.3f}  Recall@{retrieve_k}={rv:.3f}  "
              f"{DIM}embed {t_embed:.1f}s / query {t_query:.1f}s{RESET}")

    metrics = {
        "dtype": dtype,
        "n_questions": len(all_records),
        "recall_rerank": _agg(all_records, "r_rer", _recall, rerank_k),
        "recall_vec": _agg(all_records, "r_vec", _recall, retrieve_k),
        "ndcg": _agg(all_records, "r_rer", _ndcg, rerank_k),
        "mrr": _agg(all_records, "r_rer", _mrr),
        "retrieve_k": retrieve_k,
        "rerank_k": rerank_k,
        "vram_embed_mb": round(vram_embed),
        "vram_total_mb": round(vram_all),
        "load_s": round(load_s, 1),
        "embed_s": round(t_embed_total, 1),
        "query_s": round(t_query_total, 1),
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "xling": settings.xling_enabled,
        "run_at": datetime.now().isoformat(),
    }

    print(f"\n  {BOLD}合计 {metrics['n_questions']} 题{RESET}  "
          f"Recall@{rerank_k}={GREEN}{metrics['recall_rerank']:.4f}{RESET}  "
          f"Recall@{retrieve_k}={metrics['recall_vec']:.4f}  "
          f"NDCG@{rerank_k}={metrics['ndcg']:.4f}  MRR={metrics['mrr']:.4f}")
    print(f"  VRAM {metrics['vram_total_mb']} MB  ·  "
          f"embed {metrics['embed_s']:.0f}s  ·  query {metrics['query_s']:.0f}s")

    return {"metrics": metrics, "per_task": per_task}


# ── 对比 ──────────────────────────────────────────────────────────────────────

def _fmt_delta(new: float, base: float, higher_better: bool = True) -> str:
    d = new - base
    if abs(d) < 1e-9:
        return f"{DIM}±0{RESET}"
    good = (d > 0) if higher_better else (d < 0)
    return f"{GREEN if good else RED}{d:+.4f}{RESET}"


def print_compare(a: dict, b: dict) -> None:
    """a = fp32 基线, b = fp16"""
    ma, mb = a["metrics"], b["metrics"]
    k, rk = ma["rerank_k"], ma["retrieve_k"]

    print(f"\n{BOLD}{'=' * 84}{RESET}")
    print(f"{BOLD}FP32 vs FP16 — {ma['n_questions']} 题黄金集{RESET}")
    print(f"{BOLD}{'=' * 84}{RESET}")
    print(f"{'指标':<22} {'FP32':>12} {'FP16':>12} {'Δ':>16}")
    print("-" * 68)
    rows = [
        (f"Recall@{k} (rerank后)", "recall_rerank", True),
        (f"Recall@{rk} (向量上限)", "recall_vec", True),
        (f"NDCG@{k}", "ndcg", True),
        ("MRR", "mrr", True),
    ]
    for label, key, hb in rows:
        print(f"{label:<22} {ma[key]:>12.4f} {mb[key]:>12.4f} "
              f"{_fmt_delta(mb[key], ma[key], hb):>25}")
    print("-" * 68)
    for label, key, unit in [
        ("显存 (VRAM)", "vram_total_mb", "MB"),
        ("索引 embed 耗时", "embed_s", "s"),
        ("检索+rerank 耗时", "query_s", "s"),
        ("模型加载耗时", "load_s", "s"),
    ]:
        va, vb = ma[key], mb[key]
        pct = (vb - va) / va * 100 if va else 0
        col = GREEN if vb < va else (RED if vb > va else DIM)
        print(f"{label:<22} {va:>10.0f}{unit:<2} {vb:>10.0f}{unit:<2} "
              f"{col}{pct:>+14.1f}%{RESET}")

    # 逐任务，暴露是否有某个任务被 fp16 打崩
    print(f"\n{BOLD}逐任务 Recall@{k}{RESET}")
    print(f"{'ID':>3}  {'名称':<20} {'FP32':>8} {'FP16':>8} {'Δ':>14}")
    print("-" * 60)
    pa = {t["id"]: t for t in a["per_task"]}
    for tb in b["per_task"]:
        ta = pa.get(tb["id"])
        if not ta:
            continue
        print(f"{tb['id']:>3}  {tb['name']:<20} {ta['recall_rerank']:>8.3f} "
              f"{tb['recall_rerank']:>8.3f} {_fmt_delta(tb['recall_rerank'], ta['recall_rerank']):>23}")

    d = mb["recall_rerank"] - ma["recall_rerank"]
    n = ma["n_questions"]
    print(f"\n{BOLD}结论{RESET}")
    print(f"  Recall@{k} 变化 {d:+.4f}（{d * n:+.1f} 题 / {n} 题）")
    saved = ma["vram_total_mb"] - mb["vram_total_mb"]
    print(f"  显存节省 {saved} MB（{saved / ma['vram_total_mb'] * 100:.0f}%）"
          if ma["vram_total_mb"] else "")
    if abs(d * n) < 1:
        print(f"  {GREEN}→ 质量差异不足 1 题，fp16 可安全采用{RESET}")
    elif d < 0:
        print(f"  {YELLOW}→ fp16 掉了 {abs(d * n):.0f} 题，需权衡{RESET}")
    else:
        print(f"  {GREEN}→ fp16 反而更好（属噪声范围，说明无损）{RESET}")


def _save(payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = RESULTS_DIR / f"fp16exp_{payload['metrics']['dtype']}_{ts}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def _latest(dtype: str) -> dict | None:
    files = sorted(RESULTS_DIR.glob(f"fp16exp_{dtype}_*.json"))
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="FP32 vs FP16 检索质量对比（120 题黄金集）")
    ap.add_argument("--dtype", choices=["fp32", "fp16"], help="跑单个精度")
    ap.add_argument("--tasks", nargs="*", help="只跑指定任务 ID（如 01 04）")
    ap.add_argument("--save", action="store_true", help="保存结果到 results/")
    ap.add_argument("--compare", action="store_true", help="对比最近一次 fp32 与 fp16")
    ap.add_argument("--both", action="store_true",
                    help="依次在两个独立子进程跑 fp32/fp16 再对比（推荐）")
    args = ap.parse_args()

    if args.compare:
        a, b = _latest("fp32"), _latest("fp16")
        if not a or not b:
            print("缺少结果：需先跑 --dtype fp32 --save 和 --dtype fp16 --save", file=sys.stderr)
            sys.exit(1)
        print_compare(a, b)
        return

    if args.both:
        for d in ("fp32", "fp16"):
            cmd = [sys.executable, __file__, "--dtype", d, "--save"]
            if args.tasks:
                cmd += ["--tasks", *args.tasks]
            print(f"\n{DIM}$ {' '.join(cmd)}{RESET}")
            r = subprocess.run(cmd, cwd=str(ROOT))
            if r.returncode != 0:
                print(f"{RED}{d} 运行失败，中止{RESET}", file=sys.stderr)
                sys.exit(1)
        print_compare(_latest("fp32"), _latest("fp16"))
        return

    if not args.dtype:
        ap.print_help()
        sys.exit(0)

    payload = asyncio.run(run_one(args.dtype, args.tasks))
    if args.save:
        print(f"  已保存: {_save(payload)}")


if __name__ == "__main__":
    main()
