"""锚点子集对照评测 —— 在扩充后的 162 题黄金集上，把"精确锚点"42 题单独拆出来，
直接测 hybrid 在这个专门为它构造的场景里是否真的有效。

背景：用户质疑此前否决 hybrid 检索的基准不公平——① 不该拿 dense@40 做对照，该
拿 dense@20；② 黄金集构造时没考虑 BM25 该赢的场景（专有名词/型号/法条/DOI 精确
匹配），样本量不足以下结论。第一点已用数据回应（生产路径 hybrid@20 = dense@20，
零增益，不是基准选择问题）；第二点是可以修的，本脚本就是补上这块样本后的验证。

复用 eval_hybrid.py 的 dense 检索 / BM25 / 分词 / 加性并集融合逻辑，分三个子集分别
报 Recall@6：
  - 锚点子集（42 题，qa.anchor_terms 存在）—— hybrid 该赢的场景，专门构造
  - 原始子集（120 题，无 anchor_terms）—— 之前反复验证过的基线
  - 全体（162 题）

用法：  python benchmark/eval_anchor_subset.py
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

VARIANT = "jieba_ascii"   # 对齐生产 sparse_tokenizer.py
DENSE_K = 20
DENSE_K_HIGH = 40
SPARSE_N = 15              # 阶段一/三里加性并集的最优 N
RERANK_K = 6


async def main() -> None:
    from app.services.rag_service import get_rag_service, rerank_chunks
    from app.services.translation_service import get_translation_service

    data = json.loads((GOLDEN_DIR / "golden_set.json").read_text(encoding="utf-8"))
    out("=" * 96)
    out("  eval_anchor_subset —— 锚点子集(42) vs 原始子集(120) vs 全体(162)")
    out(f"  variant={VARIANT}  dense_k={DENSE_K}/{DENSE_K_HIGH}  sparse_N={SPARSE_N}  "
        f"rerank@{RERANK_K}")
    out("=" * 96)
    await _warmup()
    rag = get_rag_service()

    # records: [{is_anchor, r_d20, r_d40, r_hybrid}]
    records = []
    t0 = time.perf_counter()
    for task in data["tasks"]:
        tid = task["task_id"]
        cmap = await _index_corpus(rag, tid, task["corpus"])
        bm25, corp = _build_bm25(task["corpus"], VARIANT)

        for qa in task["qa"]:
            q = qa["question"]
            gold = set(qa["gold_cids"])
            is_anchor = "anchor_terms" in qa

            dense20 = await rag.retrieve_evidence(query=q, task_id=tid, top_k=DENSE_K)
            dense40 = await rag.retrieve_evidence(query=q, task_id=tid, top_k=DENSE_K_HIGH)
            d20_cids = {_cid(h) for h in dense20}

            en = ""
            if _CJK_RE.search(q):
                en = await get_translation_service().translate(q) or ""
            qtok = TOKENIZERS[VARIANT](f"{q}\n{en}")
            sparse_cids = _bm25_topn(bm25, corp, qtok, SPARSE_N)
            added = [c for c in sparse_cids if c not in d20_cids]
            pool = dense20 + [_mk_hit(cmap[c]) for c in added if c in cmap]

            rer_d20 = await rerank_chunks(q, dense20, top_k=RERANK_K)
            rer_d40 = await rerank_chunks(q, dense40, top_k=RERANK_K)
            rer_hy = await rerank_chunks(q, pool, top_k=RERANK_K)

            records.append({
                "task": task["id"], "q": q, "is_anchor": is_anchor,
                "r_d20": _rank_of(rer_d20, gold),
                "r_d40": _rank_of(rer_d40, gold),
                "r_hybrid": _rank_of(rer_hy, gold),
            })
        out(f"  [{task['id']}] 累计 {len(records)} 题, {time.perf_counter() - t0:.0f}s")

    def recall(recs, field):
        if not recs:
            return float("nan")
        return sum(1 for r in recs if r[field] is not None and r[field] <= RERANK_K) / len(recs)

    out("\n" + "=" * 96)
    out(f"  子集对照 Recall@{RERANK_K}")
    hdr = f"  {'子集':<12} {'n':>4} {'dense@20':>10} {'dense@40':>10} {'hybrid@20':>10}  hybrid vs d40"
    out(hdr)
    out("  " + "-" * (len(hdr) - 2))
    for name, recs in [
        ("锚点子集(42)", [r for r in records if r["is_anchor"]]),
        ("原始子集(120)", [r for r in records if not r["is_anchor"]]),
        ("全体(162)", records),
    ]:
        d20, d40, hy = recall(recs, "r_d20"), recall(recs, "r_d40"), recall(recs, "r_hybrid")
        out(f"  {name:<12} {len(recs):>4} {d20:>10.4f} {d40:>10.4f} {hy:>10.4f}  "
            f"{hy - d40:>+.4f}")

    # 锚点子集逐题明细：hybrid 相对 dense@40 的输赢
    anchor_recs = [r for r in records if r["is_anchor"]]
    out(f"\n  锚点子集逐题（hybrid@20 vs dense@40，{RERANK_K}命中=✓）")
    for r in anchor_recs:
        d40ok = "✓" if r["r_d40"] is not None and r["r_d40"] <= RERANK_K else "✗"
        hyok = "✓" if r["r_hybrid"] is not None and r["r_hybrid"] <= RERANK_K else "✗"
        mark = "  ← hybrid赢" if hyok == "✓" and d40ok == "✗" else (
            "  ← hybrid输" if hyok == "✗" and d40ok == "✓" else "")
        out(f"    [{r['task']}] d40={d40ok} hybrid={hyok}{mark}  {r['q'][:55]}")

    out(f"\n  总耗时: {time.perf_counter() - t0:.1f}s")
    OUT = GOLDEN_DIR / "_eval_anchor_subset_out.txt"
    OUT.write_text("\n".join(_lines), encoding="utf-8")
    out(f"  写入 {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
