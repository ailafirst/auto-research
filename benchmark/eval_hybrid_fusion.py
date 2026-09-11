"""阶段三 —— 融合方式与 reranker 截断深度的离线对照。

回答两个问题（阶段一/二已确认 hybrid 无 Recall 增益）：
  Q1 换融合方式（RRF / sparse-fallback）能否打平 dense@40？
  Q2 reranker 截断有多深——gold 落在 rerank rank 7~10 还是更靠后？
     若 Recall@10 明显高于 Recall@6，则真正的杠杆是 reranker_top_k / 二阶段粗筛，
     与 hybrid 正交。

复用 eval_hybrid.py 的 BM25 / 分词 / 工具函数。分词器用 jieba+ascii（对齐生产
app/services/sparse_tokenizer.py），非阶段一测得略优的 ascii-only。

用法：  python benchmark/eval_hybrid_fusion.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.eval_hybrid import (  # noqa: E402
    _CJK_RE,
    GOLDEN_DIR,
    TOKENIZERS,
    _bm25_topn,
    _build_bm25,
    _cid,
    _index_corpus,
    _lines,
    _mk_hit,
    _rank_of,
    _warmup,
    out,
)

RERANK_KS = (6, 8, 10)
DENSE_K = 20
DENSE_K_HIGH = 40
ADD_NS = (10, 15, 20, 25)
RRF_NS = (15, 25)
RRF_C = 60
VARIANT = "jieba_ascii"   # 对齐生产分词器


def _recall_at(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def _rrf_fuse(dense: list[dict], sparse_cids: list[str], cmap: dict) -> list[dict]:
    """RRF：对 dense∪sparse 按 1/(c+rank) 之和重排，返回重排后的候选列表。"""
    score: dict[str, float] = {}
    hit: dict[str, dict] = {}
    for r, h in enumerate(dense, 1):
        c = _cid(h)
        score[c] = score.get(c, 0.0) + 1.0 / (RRF_C + r)
        hit[c] = h
    for r, c in enumerate(sparse_cids, 1):
        score[c] = score.get(c, 0.0) + 1.0 / (RRF_C + r)
        if c not in hit and c in cmap:
            hit[c] = _mk_hit(cmap[c])
    ordered = sorted(score, key=lambda c: score[c], reverse=True)
    return [hit[c] for c in ordered if c in hit]


async def main() -> None:
    from app.services.rag_service import get_rag_service, rerank_chunks
    from app.services.translation_service import get_translation_service

    data = json.loads((GOLDEN_DIR / "golden_set.json").read_text(encoding="utf-8"))
    out("=" * 96)
    out("  eval_hybrid_fusion —— 阶段三：融合方式 + reranker 截断深度")
    out(f"  variant={VARIANT}  dense_k={DENSE_K}  加性N={ADD_NS}  RRF_N={RRF_NS}  "
        f"rerank@{RERANK_KS}")
    out("=" * 96)
    await _warmup()
    rag = get_rag_service()

    configs: list[str] = (
        ["D20", "D40"]
        + [f"ADD-{n}" for n in ADD_NS]
        + [f"RRF-{n}" for n in RRF_NS]
    )
    # rows[config][k] = 命中数
    rows = {c: {k: 0 for k in RERANK_KS} for c in configs}
    n_total = 0
    t0 = time.perf_counter()

    for task in data["tasks"]:
        tid = task["task_id"]
        cmap = await _index_corpus(rag, tid, task["corpus"])
        bm25, corp = _build_bm25(task["corpus"], VARIANT)

        for qa in task["qa"]:
            q = qa["question"]
            gold = set(qa["gold_cids"])
            n_total += 1

            dense20 = await rag.retrieve_evidence(query=q, task_id=tid, top_k=DENSE_K)
            dense40 = await rag.retrieve_evidence(query=q, task_id=tid, top_k=DENSE_K_HIGH)
            d20_cids = {_cid(h) for h in dense20}

            en = ""
            if _CJK_RE.search(q):
                en = await get_translation_service().translate(q) or ""
            qtok = TOKENIZERS[VARIANT](f"{q}\n{en}")
            sparse_cids_all = _bm25_topn(bm25, corp, qtok, max(max(ADD_NS), max(RRF_NS)))

            async def _rr(pool: list[dict]) -> int | None:
                r = await rerank_chunks(q, pool, top_k=max(RERANK_KS))
                return _rank_of(r, gold)

            results: dict[str, int | None] = {}
            results["D20"] = await _rr(dense20)
            results["D40"] = await _rr(dense40)
            for n in ADD_NS:
                added = [c for c in sparse_cids_all[:n] if c not in d20_cids]
                pool = dense20 + [_mk_hit(cmap[c]) for c in added if c in cmap]
                results[f"ADD-{n}"] = await _rr(pool)
            for n in RRF_NS:
                fused = _rrf_fuse(dense20, sparse_cids_all[:n], cmap)[:30]
                results[f"RRF-{n}"] = await _rr(fused)

            for c, rank in results.items():
                for k in RERANK_KS:
                    rows[c][k] += int(_recall_at(rank, k))

        out(f"  [{task['id']}] 累计 {n_total} 题, {time.perf_counter() - t0:.0f}s")

    out("\n" + "=" * 96)
    out(f"  逐配置 Recall@k（{n_total} 题）")
    out(f"  {'配置':<10} {'Recall@6':>10} {'Recall@8':>10} {'Recall@10':>10}")
    out("  " + "-" * 44)
    for c in configs:
        r = {k: rows[c][k] / n_total for k in RERANK_KS}
        out(f"  {c:<10} {r[6]:>10.4f} {r[8]:>10.4f} {r[10]:>10.4f}")
    out("=" * 96)
    d40_6 = rows["D40"][6] / n_total
    beat = [c for c in configs if c not in ("D20", "D40")
            and rows[c][6] / n_total >= d40_6]
    out(f"  打平/超过 dense@40 Recall@6({d40_6:.4f}) 的 hybrid 配置：{beat or '无'}")
    trunc = max(rows[c][10] - rows[c][6] for c in configs) / n_total
    out(f"  最大 (Recall@10 − Recall@6) = {trunc:.4f}  → 截断深度 / reranker_top_k 空间")

    OUT = GOLDEN_DIR / "_eval_hybrid_fusion_out.txt"
    OUT.write_text("\n".join(_lines), encoding="utf-8")
    out(f"  写入 {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
