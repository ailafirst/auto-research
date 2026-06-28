"""离线 RAG 检索质量评测 — 确定性 Recall@k / MRR / NDCG@k。

加载 build_golden_set.py 冻结的黄金集（固定语料 + question→gold chunk 标注），
在内存向量库中复刻**生产检索路径**（向量 top-20 → rerank → top-6），对每条黄金
问题计算 gold chunk 的命中排名，汇总业内标准检索指标。全程零网络、零 LLM、确定性
可复现，是召回率调参时盯的主标尺。

关键诊断：分离两类丢失
  ── 向量阶段 Recall@20：embedding 是否把 gold chunk 召进候选池（召回上限）
  ── rerank 后 Recall@6：reranker 是否把 gold chunk 留在最终 top-k（排序损失）
若 Recall@20 高、Recall@6 低 → 问题在 rerank/截断；若 Recall@20 本身就低 →
问题在 embedding/切片/query，需扩大 retrieve_k 或改写 query。

指标（单 gold/问题，故 Recall@k == Hit@k）：
  Recall@k  前 k 个结果命中 gold 的问题比例（↑）
  MRR       gold 命中排名倒数的均值（↑，衡量排得多靠前）
  NDCG@k    带位置折扣的命中质量（↑，排序金标准）

用法：
  python benchmark/eval_retrieval.py                       # 用 golden/golden_set.json
  python benchmark/eval_retrieval.py --golden golden_set.json
  python benchmark/eval_retrieval.py 05 07 09              # 仅评测指定任务
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

GOLDEN_DIR = Path(__file__).parent / "GoldenDataset"

GREEN = "\033[32m"; YELLOW = "\033[33m"; RED = "\033[31m"
BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"


def _cid_of(chunk: dict) -> str:
    return f"{chunk.get('source_id', '')}#{chunk.get('chunk_index', 0)}"


def _rank_of(ranked: list[dict], gold: set[str]) -> int | None:
    """gold cid 在 ranked 列表中的 1-based 排名，未命中返回 None。"""
    for i, c in enumerate(ranked, 1):
        if _cid_of(c) in gold:
            return i
    return None


def _recall(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def _ndcg(rank: int | None, k: int) -> float:
    # 单 gold：IDCG = 1/log2(2) = 1，故 NDCG@k = 1/log2(rank+1)（命中且在前 k）
    return 1.0 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0


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


async def _index_corpus(rag, task_id: str, corpus: list[dict]) -> int:
    """把黄金集语料写入内存向量库，复刻 build_evidence 的 title 前缀 embedding。"""
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


async def eval_task(rag, task: dict, retrieve_k: int, rerank_k: int, use_rerank: bool) -> dict:
    from app.services.rag_service import rerank_chunks
    task_id = task["task_id"]
    n = await _index_corpus(rag, task_id, task["corpus"])

    records = []
    for qa in task["qa"]:
        gold = set(qa["gold_cids"])
        # 跨语言由 retrieve_evidence 内部按 settings.xling_enabled 自动处理（XLING_ENABLED=1）
        vec = await rag.retrieve_evidence(query=qa["question"], task_id=task_id,
                                          top_k=retrieve_k)
        r_vec = _rank_of(vec, gold)            # 向量阶段排名（候选池内）
        r_norr = _rank_of(vec[:rerank_k], gold)  # 不 rerank，直接取向量 top-k
        if use_rerank and vec:
            rer = await rerank_chunks(qa["question"], vec, top_k=rerank_k)
            r_rer = _rank_of(rer, gold)
        else:
            r_rer = r_norr
        records.append({"q": qa["question"], "r_vec": r_vec, "r_norr": r_norr, "r_rer": r_rer})

    return {"id": task["id"], "name": task["name"], "n_corpus": n,
            "n_q": len(records), "records": records}


def _agg(records: list[dict], field: str, fn, *args) -> float:
    return sum(fn(r[field], *args) for r in records) / len(records) if records else 0.0


def _c(v: float, good: float = 0.7, mid: float = 0.5) -> str:
    s = f"{v:.3f}"
    return (GREEN if v >= good else (YELLOW if v >= mid else RED)) + s + RESET


def print_report(results: list[dict], retrieve_k: int, rerank_k: int, use_rerank: bool) -> None:
    all_rec = [r for t in results for r in t["records"]]

    print(f"\n{'═' * 100}")
    print(f"  {BOLD}RAG 检索质量评测{RESET}  "
          f"（向量 top-{retrieve_k} → {'rerank top-' + str(rerank_k) if use_rerank else '无 rerank'}）")
    print(f"{'═' * 100}")

    # ── 逐任务：最终（rerank 后）指标 ──
    print(f"\n  {BOLD}逐任务 — 最终 top-{rerank_k} 检索质量{RESET}")
    hdr = f"  {'ID':<4} {'任务名':<18} {'语料':>5} {'问题':>4}  {'R@1':>7} {'R@3':>7} {'R@'+str(rerank_k):>7} {'MRR':>7} {'NDCG':>7}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for t in results:
        rs = t["records"]
        if not rs:
            print(f"  {t['id']:<4} {t['name'][:16]:<18} {t['n_corpus']:>5} {'0':>4}  (无问题)"); continue
        print(f"  {t['id']:<4} {t['name'][:16]:<18} {t['n_corpus']:>5} {t['n_q']:>4}  "
              f"{_c(_agg(rs,'r_rer',_recall,1)):>16} {_c(_agg(rs,'r_rer',_recall,3)):>16} "
              f"{_c(_agg(rs,'r_rer',_recall,rerank_k)):>16} {_c(_agg(rs,'r_rer',lambda r:1.0/r if r else 0.0)):>16} "
              f"{_c(_agg(rs,'r_rer',_ndcg,rerank_k)):>16}")

    # ── 汇总 + 召回损失诊断 ──
    print(f"\n  {BOLD}汇总（{len(all_rec)} 条黄金问题）{RESET}")
    rec_vec20  = _agg(all_rec, "r_vec",  _recall, retrieve_k)
    rec_vec6   = _agg(all_rec, "r_norr", _recall, rerank_k)
    rec_rer6   = _agg(all_rec, "r_rer",  _recall, rerank_k)
    print(f"    {'向量召回上限':<22} Recall@{retrieve_k:<3} = {_c(rec_vec20)}   "
          f"{DIM}(embedding 是否把 gold 召进候选池){RESET}")
    print(f"    {'不 rerank（向量序）':<20} Recall@{rerank_k:<3} = {_c(rec_vec6)}   "
          f"{DIM}MRR={_agg(all_rec,'r_norr',lambda r:1.0/r if r else 0.0):.3f}{RESET}")
    print(f"    {'rerank 后（最终）':<21} Recall@{rerank_k:<3} = {_c(rec_rer6)}   "
          f"{DIM}MRR={_agg(all_rec,'r_rer',lambda r:1.0/r if r else 0.0):.3f}  "
          f"NDCG@{rerank_k}={_agg(all_rec,'r_rer',_ndcg,rerank_k):.3f}{RESET}")

    # 阶梯 Recall（向量阶段，定位召回上限在哪个 k 饱和）
    ladder = "  ".join(f"@{k}={_agg(all_rec,'r_vec',_recall,k):.3f}" for k in (1, 3, 5, 10, retrieve_k))
    print(f"\n    {DIM}向量阶段 Recall 阶梯: {ladder}{RESET}")

    # 诊断结论
    lost_vec = 1.0 - rec_vec20
    lost_rer = rec_vec20 - rec_rer6
    print(f"\n  {BOLD}诊断{RESET}")
    print(f"    向量漏检（candidates 都没召回）: {RED}{lost_vec:.1%}{RESET}  → 改 embedding/切片/query/retrieve_k")
    print(f"    rerank 截断损失（召回了但被挤出 top-{rerank_k}）: {YELLOW}{lost_rer:.1%}{RESET}  → 改 reranker/扩大 top_k")

    # 漏检清单：gold 未进 top-{retrieve_k} 的硬漏检（定位是噪声问题还是真·难检索）
    misses = [r for r in all_rec if r["r_vec"] is None]
    if misses:
        print(f"\n  {BOLD}向量硬漏检清单（gold 未进 top-{retrieve_k}，{len(misses)} 条）{RESET}")
        for r in misses:
            print(f"    {RED}✗{RESET} {r['q']}")
    # rerank 把已召回的 gold 挤出 top-k（向量召回了但最终丢失）
    dropped = [r for r in all_rec if r["r_vec"] is not None and (r["r_rer"] is None or r["r_rer"] > rerank_k)]
    if dropped:
        print(f"\n  {BOLD}rerank 截断丢失（已召回但挤出 top-{rerank_k}，{len(dropped)} 条）{RESET}")
        for r in dropped:
            print(f"    {YELLOW}✗{RESET} 向量rank={r['r_vec']}  {r['q']}")


async def main(golden_name: str, task_ids: list[str]) -> None:
    from app.core.config import settings
    from app.services.rag_service import get_rag_service

    path = GOLDEN_DIR / golden_name
    if not path.exists():
        print(f"黄金集不存在: {path}\n请先运行 build_golden_set.py", file=sys.stderr); sys.exit(1)
    data = json.loads(path.read_text(encoding="utf-8"))

    tasks = data["tasks"]
    if task_ids:
        tasks = [t for t in tasks if t["id"] in task_ids]

    retrieve_k  = settings.reranker_retrieve_k if settings.reranker_enabled else settings.rag_top_k
    rerank_k    = settings.reranker_top_k
    use_rerank  = settings.reranker_enabled

    print("═" * 72)
    print("  eval_retrieval — 离线 RAG 检索质量评测")
    print(f"  黄金集: {golden_name}  (built {data.get('built_at','?')[:19]})")
    print(f"  embedding={data.get('embedding_model')}  retrieve_k={retrieve_k}  "
          f"rerank={'on top-' + str(rerank_k) if use_rerank else 'off'}"
          f"{'  跨语言双路=ON' if settings.xling_enabled else ''}")
    if data.get("embedding_model") != settings.embedding_model:
        print(f"  {YELLOW}⚠ 当前 embedding({settings.embedding_model}) 与黄金集构建时不同{RESET}")
    print("═" * 72)

    await _warmup()
    rag = get_rag_service()

    t0 = time.perf_counter()
    results = []
    for task in tasks:
        r = await eval_task(rag, task, retrieve_k, rerank_k, use_rerank)
        results.append(r)
        print(f"  [{r['id']}] {r['name']}: 语料 {r['n_corpus']}，评测 {r['n_q']} 问题", flush=True)

    print_report(results, retrieve_k, rerank_k, use_rerank)
    print(f"\n  总耗时: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="离线 RAG 检索质量评测（Recall@k / MRR / NDCG@k）")
    parser.add_argument("task_ids", nargs="*", default=[], help="仅评测指定任务 ID（默认全部）")
    parser.add_argument("--golden", default="golden_set.json", help="黄金集文件名（benchmark/GoldenDataset/）")
    args = parser.parse_args()
    asyncio.run(main(args.golden, args.task_ids))
