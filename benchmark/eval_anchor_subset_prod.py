"""锚点子集对照评测 —— 生产路径版（HYBRID_ENABLED=1，走真实 Qdrant 命名向量 + sparse +
Modifier.IDF，而非 eval_anchor_subset.py 用的理想化 per-task BM25Okapi）。

阶段二已经证明过一次：per-task BM25 理想 IDF 和生产 collection 级 IDF 算出来的结果
不一样（原 120 题上从 +0.8~1.7pp 直接归零）。既然要认真回应"黄金集没考虑 BM25 场景"
这个质疑，这一步就不能只信理想化实验室数字——必须用会真正上线的代码路径复核一遍
锚点子集的 +14.3pp 是否也在生产 IDF 下成立。

用法：HYBRID_ENABLED=1 python benchmark/eval_anchor_subset_prod.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.eval_hybrid import GOLDEN_DIR, _lines, _rank_of, _warmup, out  # noqa: E402


async def main() -> None:
    from app.core.config import settings
    from app.services.rag_service import get_rag_service, rerank_chunks

    if not settings.hybrid_enabled:
        out("  错误：需要 HYBRID_ENABLED=1 才能测生产路径，当前为关闭状态")
        return

    data = json.loads((GOLDEN_DIR / "golden_set.json").read_text(encoding="utf-8"))
    out("=" * 96)
    out("  eval_anchor_subset_prod —— 生产 hybrid 路径（Qdrant sparse + Modifier.IDF）")
    out(f"  hybrid_dense_k={settings.hybrid_dense_k}  hybrid_sparse_k={settings.hybrid_sparse_k}  "
        f"dense@40 对照同时测")
    out("=" * 96)
    await _warmup()
    rag = get_rag_service()

    # 用 eval_retrieval.py 的 _index_corpus，不是 eval_hybrid.py 的——后者是阶段一为
    # local BM25Okapi 写的，从不算/存稀疏向量；HYBRID_ENABLED=1 时若用它索引语料，
    # collection 是 hybrid schema 但每个 point 都没有 sparse 字段，search_sparse
    # 永远查不到东西，"生产 hybrid" 实测出来的其实是纯 dense@20——踩过这个坑，
    # 教训直接写在这里避免下次重犯。
    from benchmark.eval_retrieval import _index_corpus

    records = []
    t0 = time.perf_counter()
    for task in data["tasks"]:
        tid = task["task_id"]
        await _index_corpus(rag, tid, task["corpus"])

        for qa in task["qa"]:
            q = qa["question"]
            gold = set(qa["gold_cids"])
            is_anchor = "anchor_terms" in qa

            # retrieve_evidence 现在按查询是否含锚点自适应决定要不要做 dense+sparse
            # 加性并集（sparse_tokenizer.has_lexical_anchor），dense 预算也是它内部
            # 自己收窄，调用方统一传 reranker_retrieve_k（与 retriever_node 传参一致）
            # 即可——不用像早期版本那样手动传 hybrid_dense_k。
            # dense@40 对照仍需临时关掉 hybrid_enabled：开着时任何查询的预算都可能被
            # 内部收窄（锚点查询），两栏会不可比，取完立刻恢复。
            hy = await rag.retrieve_evidence(query=q, task_id=tid, top_k=settings.reranker_retrieve_k)
            settings.hybrid_enabled = False
            try:
                d40 = await rag.retrieve_evidence(query=q, task_id=tid, top_k=40)
            finally:
                settings.hybrid_enabled = True

            rer_hy = await rerank_chunks(q, hy, top_k=6)
            rer_d40 = await rerank_chunks(q, d40, top_k=6)

            records.append({
                "task": task["id"], "q": q, "is_anchor": is_anchor,
                "r_hybrid": _rank_of(rer_hy, gold),
                "r_d40": _rank_of(rer_d40, gold),
            })
        out(f"  [{task['id']}] 累计 {len(records)} 题, {time.perf_counter() - t0:.0f}s")

    def recall(recs, field):
        if not recs:
            return float("nan")
        return sum(1 for r in recs if r[field] is not None and r[field] <= 6) / len(recs)

    out("\n" + "=" * 96)
    out("  子集对照 Recall@6（生产 hybrid 路径）")
    hdr = f"  {'子集':<12} {'n':>4} {'dense@40':>10} {'hybrid(生产)':>12}  差值"
    out(hdr)
    out("  " + "-" * (len(hdr) - 2))
    for name, recs in [
        ("锚点子集(42)", [r for r in records if r["is_anchor"]]),
        ("原始子集(120)", [r for r in records if not r["is_anchor"]]),
        ("全体(162)", records),
    ]:
        d40, hy = recall(recs, "r_d40"), recall(recs, "r_hybrid")
        out(f"  {name:<12} {len(recs):>4} {d40:>10.4f} {hy:>12.4f}  {hy - d40:>+.4f}")

    out(f"\n  总耗时: {time.perf_counter() - t0:.1f}s")
    OUT = GOLDEN_DIR / "_eval_anchor_subset_prod_out.txt"
    OUT.write_text("\n".join(_lines), encoding="utf-8")
    out(f"  写入 {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
