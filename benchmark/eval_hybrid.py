"""阶段一最小验证 —— dense + BM25 稀疏检索 加性并集融合，离线跑黄金集。

结论（2026-09-10，见 GoldenDataset/improvement_plan.md「Hybrid Sparse 二次验证」）：
❌ 负结果，不进阶段二。hybrid@20 最好只打平 dense@40（0.9000），未反超；jieba 中文
分词零贡献（ascii-only 反而最优）；复现 P1 结论——瓶颈在 reranker 提取而非检索广度。
本脚本作为实验记录保留，jieba / rank_bm25 不进 requirements.txt。


不改 app/、不改 Qdrant collection、不引生产依赖。只回答一个问题：
在 retrieve_k=20 的 dense 基线上加一路 BM25（jieba 中文分词 + 复用 P-XLing 译文
喂 sparse），用「加性并集」融合后 rerank，最终 Recall@6 能否追平 dense@k=40 的
0.900，且不把已命中的 dense gold 挤出 top-6。

融合方式（关键，区别于此前失败的 P1）：
  dense top-20 候选**全部原样保留**，BM25 top-N 里 dense 没有的追加到池尾，
  一起进 reranker。BM25 只能「加」候选，不做会顶掉 dense 候选的 RRF 全局重排。
  唯一的伤害路径 = BM25 追加的 chunk 在 rerank 阶段顶掉了 dense gold（可度量）。

三条基线在**同一次进程、同一模型状态**下算出，消除跨轮噪声：
  dense@20-rerank / dense@40-rerank / hybrid(variant,N)-rerank

用法：
  python benchmark/eval_hybrid.py                       # 全量 120 题，6 个 hybrid 配置
  python benchmark/eval_hybrid.py 02 05 06 10           # 仅指定任务
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import re
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

import jieba  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

jieba.setLogLevel(60)

GOLDEN_DIR = Path(__file__).parent / "GoldenDataset"
OUT_PATH = Path(__file__).parent / "GoldenDataset" / "_eval_hybrid_out.txt"

DENSE_K = 20          # hybrid 的 dense 侧候选数（阶段一基线）
DENSE_K_HIGH = 40     # 对照：当前生产的 retrieve_k
RERANK_K = 6
SPARSE_NS = [5, 10, 15]
VARIANTS = ["jieba_ascii", "ascii"]

_ASCII_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[一-鿿]")

_lines: list[str] = []


def out(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


# ── 分词器 ─────────────────────────────────────────────────────────────────────

def tok_ascii(s: str) -> list[str]:
    return _ASCII_RE.findall(s.lower())


def tok_jieba_ascii(s: str) -> list[str]:
    """英文/数字走正则（保留 h200/pipl 这类锚点），中文走 jieba 搜索模式。"""
    toks = tok_ascii(s)
    for w in jieba.lcut_for_search(s):
        w = w.strip()
        if w and _CJK_RE.search(w):
            toks.append(w)
    return toks


TOKENIZERS = {"jieba_ascii": tok_jieba_ascii, "ascii": tok_ascii}


def _cid(x: dict) -> str:
    return f"{x.get('source_id', '')}#{x.get('chunk_index', 0)}"


def _rank_of(ranked: list[dict], gold: set[str]) -> int | None:
    for i, c in enumerate(ranked, 1):
        if _cid(c) in gold:
            return i
    return None


def _recall(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


# ── 语料索引 ───────────────────────────────────────────────────────────────────

async def _index_corpus(rag, task_id: str, corpus: list[dict]) -> dict[str, dict]:
    """dense 侧写内存向量库（复刻 build_evidence 的 title 前缀 embedding）。"""
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
    vc, ve = zip(*valid)
    await rag.vector_store.store_chunks(list(vc), list(ve))
    return {_cid({"source_id": c["source_id"], "chunk_index": c["chunk_index"]}): c
            for c in corpus}


def _build_bm25(corpus: list[dict], variant: str):
    tk = TOKENIZERS[variant]
    docs = [tk(f"{c['title']}\n\n{c['text']}" if c.get("title") else c["text"])
            for c in corpus]
    return BM25Okapi(docs), [c for c in corpus]


def _bm25_topn(bm25, corpus: list[dict], query_tokens: list[str], n: int) -> list[str]:
    if not query_tokens:
        return []
    scores = bm25.get_scores(query_tokens)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [_cid(corpus[i]) for i in order[:n] if scores[i] > 0.0]


def _mk_hit(c: dict) -> dict:
    return {"source_id": c["source_id"], "chunk_index": c["chunk_index"],
            "url": c.get("url", ""), "title": c.get("title", ""),
            "text": c.get("text", ""), "score": 0.0}


# ── 单任务评测 ─────────────────────────────────────────────────────────────────

async def eval_task(rag, task: dict) -> list[dict]:
    from app.services.rag_service import rerank_chunks
    from app.services.translation_service import get_translation_service

    task_id = task["task_id"]
    cmap = await _index_corpus(rag, task_id, task["corpus"])

    bm25 = {}
    for v in VARIANTS:
        bm25[v] = _build_bm25(task["corpus"], v)

    records = []
    for qa in task["qa"]:
        q = qa["question"]
        gold = set(qa["gold_cids"])

        # dense：retrieve_evidence 内部自动跑 P-XLing 双路
        dense20 = await rag.retrieve_evidence(query=q, task_id=task_id, top_k=DENSE_K)
        dense40 = await rag.retrieve_evidence(query=q, task_id=task_id, top_k=DENSE_K_HIGH)
        dense20_cids = {_cid(h) for h in dense20}

        en = ""
        if _CJK_RE.search(q):
            en = await get_translation_service().translate(q) or ""
        q_for_sparse = f"{q}\n{en}".strip()

        rec = {
            "task": task["id"], "q": q, "gold": gold,
            "r_d20_pool": _rank_of(dense20, gold),
            "gold_in_d20": bool(gold & dense20_cids),
        }
        # 基线 rerank
        rer_d20 = await rerank_chunks(q, dense20, top_k=RERANK_K)
        rer_d40 = await rerank_chunks(q, dense40, top_k=RERANK_K)
        rec["r_d20_rr"] = _rank_of(rer_d20, gold)
        rec["r_d40_rr"] = _rank_of(rer_d40, gold)

        for v in VARIANTS:
            b, corp = bm25[v]
            qtok = TOKENIZERS[v](q_for_sparse)
            for n in SPARSE_NS:
                sparse_cids = _bm25_topn(b, corp, qtok, n)
                added = [c for c in sparse_cids if c not in dense20_cids]
                pool = dense20 + [_mk_hit(cmap[c]) for c in added if c in cmap]
                rer = await rerank_chunks(q, pool, top_k=RERANK_K)
                key = f"{v}_N{n}"
                rec[f"r_{key}"] = _rank_of(rer, gold)
                rec[f"pool_{key}"] = len(pool)
                rec[f"sparsehit_{key}"] = bool(gold & set(sparse_cids))
        records.append(rec)
    return records


# ── 汇总 ───────────────────────────────────────────────────────────────────────

def report(all_rec: list[dict]) -> None:
    n = len(all_rec)
    out("\n" + "=" * 92)
    out(f"  阶段一 hybrid 最小验证汇总（{n} 条黄金问题）")
    out("=" * 92)

    def recall(field: str) -> float:
        return sum(_recall(r.get(field), RERANK_K) for r in all_rec) / n

    base20 = recall("r_d20_rr")
    base40 = recall("r_d40_rr")
    out(f"\n  基线  dense@20 → rerank  Recall@6 = {base20:.4f}")
    out(f"  基线  dense@40 → rerank  Recall@6 = {base40:.4f}   (当前生产)")
    out(f"  目标  hybrid@20 Recall@6 ≥ {base40:.4f}  且  挤占损失 ≤ 1\n")

    hdr = f"  {'配置':<16} {'Recall@6':>9} {'vs d40':>8} {'池均值':>7} {'捞回':>6} {'挤占':>6} {'净':>5}"
    out(hdr)
    out("  " + "-" * (len(hdr) - 2))

    best = None
    for v in VARIANTS:
        for nn in SPARSE_NS:
            key = f"{v}_N{nn}"
            fld = f"r_{key}"
            rc = recall(fld)
            pool_mean = sum(r[f"pool_{key}"] for r in all_rec) / n
            # 捞回：dense@20 池里没有 gold，但 hybrid rerank 进了 top-6
            recovered = [r for r in all_rec
                         if not r["gold_in_d20"] and _recall(r.get(fld), RERANK_K)]
            # 挤占：dense@20-rerank 命中 top-6，hybrid 反而丢了
            crowded = [r for r in all_rec
                       if _recall(r.get("r_d20_rr"), RERANK_K)
                       and not _recall(r.get(fld), RERANK_K)]
            net = len(recovered) - len(crowded)
            mark = ""
            if rc >= base40 and len(crowded) <= 1:
                mark = "  ← GO"
                if best is None or rc > best[1]:
                    best = (key, rc, recovered, crowded)
            out(f"  {key:<16} {rc:>9.4f} {rc - base40:>+8.4f} {pool_mean:>7.1f} "
                f"{len(recovered):>6} {len(crowded):>6} {net:>+5}{mark}")

    # 逐条明细：捞回与挤占
    for v in VARIANTS:
        for nn in SPARSE_NS:
            key = f"{v}_N{nn}"
            fld = f"r_{key}"
            recovered = [r for r in all_rec
                         if not r["gold_in_d20"] and _recall(r.get(fld), RERANK_K)]
            crowded = [r for r in all_rec
                       if _recall(r.get("r_d20_rr"), RERANK_K)
                       and not _recall(r.get(fld), RERANK_K)]
            if not recovered and not crowded:
                continue
            out(f"\n  [{key}]")
            for r in recovered:
                out(f"    捞回  [{r['task']}] {r['q'][:60]}  (sparse_hit={r[f'sparsehit_{key}']})")
            for r in crowded:
                out(f"    挤占  [{r['task']}] {r['q'][:60]}  (d20_rr_rank={r['r_d20_rr']})")

    # sparse 能覆盖多少「dense@20 硬漏检」
    d20_miss = [r for r in all_rec if not r["gold_in_d20"]]
    out(f"\n  dense@20 硬漏检 {len(d20_miss)} 条；各配置 sparse top-N 命中 gold 的条数：")
    for v in VARIANTS:
        for nn in SPARSE_NS:
            key = f"{v}_N{nn}"
            hit = sum(1 for r in d20_miss if r.get(f"sparsehit_{key}"))
            out(f"    {key:<16} {hit}/{len(d20_miss)}")

    out("\n" + "=" * 92)
    if best:
        out(f"  判定：GO —— 最优配置 {best[0]}  Recall@6={best[1]:.4f} "
            f"(≥ dense@40 {base40:.4f})，捞回 {len(best[2])} 挤占 {len(best[3])}")
    else:
        out(f"  判定：STOP —— 无配置能在 挤占≤1 的前提下追平 dense@40 ({base40:.4f})")
    out("=" * 92)


async def _warmup():
    from app.core.config import settings as _s
    from app.services.rag_service import _get_reranker, get_rag_service
    out("  预热模型...")
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
    out("  预热完成")


async def main(golden_name: str, task_ids: list[str]) -> None:
    from app.core.config import settings
    from app.services.rag_service import get_rag_service

    path = GOLDEN_DIR / golden_name
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data["tasks"]
    if task_ids:
        tasks = [t for t in tasks if t["id"] in task_ids]

    out("=" * 92)
    out("  eval_hybrid —— dense + BM25 加性并集 最小验证")
    out(f"  黄金集: {golden_name}  embedding={data.get('embedding_model')}  "
        f"xling={'ON' if settings.xling_enabled else 'OFF'}  "
        f"reranker={settings.reranker_model}")
    out(f"  dense_k={DENSE_K}  对照 dense_k={DENSE_K_HIGH}  sparse_N={SPARSE_NS}  "
        f"variants={VARIANTS}")
    out("=" * 92)

    await _warmup()
    rag = get_rag_service()

    t0 = time.perf_counter()
    all_rec: list[dict] = []
    for task in tasks:
        recs = await eval_task(rag, task)
        all_rec.extend(recs)
        out(f"  [{task['id']}] {task['name']}: {len(recs)} 题  "
            f"(累计 {len(all_rec)}, {time.perf_counter() - t0:.0f}s)")

    report(all_rec)
    out(f"\n  总耗时: {time.perf_counter() - t0:.1f}s")
    OUT_PATH.write_text("\n".join(_lines), encoding="utf-8")
    out(f"  结果已写入 {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task_ids", nargs="*", default=[])
    parser.add_argument("--golden", default="golden_set.json")
    args = parser.parse_args()
    asyncio.run(main(args.golden, args.task_ids))
