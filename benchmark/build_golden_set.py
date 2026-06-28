"""构建 RAG 检索评测黄金集（LLM 合成 question → gold chunk 对）。

目的：本系统语料是每次查询动态抓取的，没有固定的相关性标注（qrels），无法计算
业内标准的 Recall@k / MRR / NDCG@k。本脚本对每个基准任务跑一遍抓取+切片管线，
拿到该任务的全部 chunk 后，让 LLM 为抽样的 chunk 各生成一个「答案明确出自该 chunk」
的中文问题——该 chunk 即这条问题的 gold 正例。冻结成固定 JSON 后，eval_retrieval.py
可离线、零网络、零 LLM、确定性地复用。

设计：
  - 单 gold chunk/问题（LlamaIndex generate_question_context_pairs 同款），
    Recall@k 即 Hit@k，MRR/NDCG 语义清晰。
  - 来源多样：每个来源 URL 最多取 MAX_PER_SOURCE 个 chunk，避免黄金集偏向单源。
  - 固定随机种子，保证抽样可复现。

用法：
  python benchmark/build_golden_set.py                 # 全部 12 个任务
  python benchmark/build_golden_set.py 05 07 09        # 指定任务
  python benchmark/build_golden_set.py --out golden_set.json
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

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

TASKS_FILE  = Path(__file__).parent / "tasks.json"
GOLDEN_DIR  = Path(__file__).parent / "GoldenDataset"
DEFAULT_TASKS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

QUESTIONS_PER_TASK = 10      # 每任务合成问题数
MIN_CHUNK_LEN      = 200     # chunk 短于此不作为出处（碎片无法生成可检索问题）
MAX_PER_SOURCE     = 2       # 每个来源 URL 最多抽几个 chunk（保证来源多样）
SEED               = 42

# 管线裁剪（与 benchmark_rag 对齐）：质量优先取 top-N，防止耗时失控
MAX_URLS    = 24
MAX_SOURCES = 16

DIM = "\033[2m"; RESET = "\033[0m"


def _cid(chunk: dict) -> str:
    """稳定 chunk 标识（EvidenceChunk.chunk_id 默认 id(object()) 不可靠）。"""
    return f"{chunk.get('source_id', '')}#{chunk.get('chunk_index', 0)}"


def _cap(state: dict, after: str) -> None:
    """各节点后裁剪中间状态，防止 embedding 耗时爆炸（质量优先取 top-N）。"""
    if after == "retriever":
        results = sorted(
            state.get("search_results", []),
            key=lambda r: r.get("tavily_score") or 0.0, reverse=True,
        )
        state["search_results"] = results[:MAX_URLS]
    elif after == "source_evaluator":
        srcs = state.get("evaluated_sources", [])
        if len(srcs) > MAX_SOURCES:
            top = sorted(srcs, key=lambda x: x.get("final_score", 0), reverse=True)[:MAX_SOURCES]
            keep = {s.get("url", "") for s in top}
            state["evaluated_sources"] = top
            state["crawled_documents"] = [
                d for d in state.get("crawled_documents", []) if d.get("url") in keep
            ]


async def _warmup() -> None:
    from app.core.config import settings as _s
    from app.services.rag_service import get_rag_service
    t0 = time.perf_counter()
    print(f"  {DIM}预热 embedding 模型...{RESET}", flush=True)
    rag = get_rag_service()
    if _s.embedding_provider == "st":
        await rag._get_st_model()
    elif _s.embedding_provider == "fastembed":
        await rag._get_fastembed_model()
    print(f"  {DIM}预热完成（{time.perf_counter() - t0:.1f}s）{RESET}", flush=True)


async def _run_pipeline(task: dict) -> list[dict]:
    """跑 planner→…→evidence_builder，返回该任务全部 evidence_chunks（dict 列表）。"""
    from app.core.config import settings as _settings
    from app.graph.nodes import (
        content_extractor_node, evidence_builder_node, planner_node,
        retriever_node, source_evaluator_node,
    )
    _settings.max_sources_per_round = MAX_URLS

    now = datetime.now().isoformat()
    state: dict[str, Any] = {
        "task_id": f"golden_{task['id']}", "query": task["query"],
        "language": task.get("language", "zh-CN"), "max_rounds": 1,
        "current_round": 1, "status": "planning", "user_hints": {},
        "research_strategy": {}, "research_plan": {}, "sub_questions": [],
        "search_queries": [], "search_results": [], "search_summaries": [],
        "crawled_documents": [], "evaluated_sources": [], "evidence_chunks": [],
        "sub_answers": [], "fact_check_result": {}, "fact_check_passed": True,
        "follow_up_queries": [], "final_report": "", "errors": [],
        "progress": 0, "progress_message": "", "created_at": now, "updated_at": now,
    }
    pipeline = [
        ("planner", planner_node), ("retriever", retriever_node),
        ("content_extractor", content_extractor_node),
        ("source_evaluator", source_evaluator_node),
        ("evidence_builder", evidence_builder_node),
    ]
    for name, fn in pipeline:
        result = await fn(state)
        state.update(result)
        _cap(state, after=name)
    return state.get("evidence_chunks", [])


def _clean_corpus(chunks: list[dict]) -> list[dict]:
    """对语料应用与生产切片器一致的 markdown 清洗 + 样板丢弃（保留原 cid）。"""
    from app.services.markdown_cleaner import is_boilerplate, md_to_text
    out: list[dict] = []
    for c in chunks:
        raw = c.get("text", "")
        cleaned = md_to_text(raw)
        if is_boilerplate(raw, cleaned):
            continue
        out.append({**c, "text": cleaned})
    return out


def _select_chunks(chunks: list[dict]) -> list[dict]:
    """抽样：过滤碎片 → 每来源限量 → 固定种子打散 → 取 QUESTIONS_PER_TASK 个。"""
    pool = [c for c in chunks if len(c.get("text", "")) >= MIN_CHUNK_LEN]
    rng = random.Random(SEED)
    rng.shuffle(pool)
    picked: list[dict] = []
    per_source: dict[str, int] = {}
    for c in pool:
        url = c.get("url", "")
        if per_source.get(url, 0) >= MAX_PER_SOURCE:
            continue
        per_source[url] = per_source.get(url, 0) + 1
        picked.append(c)
        if len(picked) >= QUESTIONS_PER_TASK:
            break
    return picked


_GEN_SYS = "你是检索评测数据集构造助手，负责根据给定文本片段生成高质量的检索问题。"
_GEN_TMPL = """根据下面的文本片段，生成一个中文问题，要求：
1. 答案必须明确、完整地出自该片段中的具体事实/数据/结论，不能依赖片段外的常识；
2. 问题要具体、像真实用户的检索意图，不要写"这段讲了什么""作者的观点是什么"这类泛问；
3. 不要照抄片段原句的措辞，用自然的提问方式重新表达；
4. 只输出问题本身一行，不要任何前缀、编号、引号或解释。

文本片段：
{text}"""


async def _gen_question(llm, chunk: dict) -> dict | None:
    text = chunk.get("text", "")[:1500]
    try:
        out = await llm.chat(
            messages=[
                {"role": "system", "content": _GEN_SYS},
                {"role": "user", "content": _GEN_TMPL.format(text=text)},
            ],
            # mimo 等推理模型先消耗 ~1200-1700 reasoning token，预算太小会 finish=length
            # 把思维链截断、content 返回空；需留足空间给推理 + 问题本身
            temperature=0.4, max_tokens=4096,
        )
    except Exception as exc:  # LLM 输出不可控，任何异常都降级为"这条不要"，不拖垮整批
        print(f"    问题生成失败 {_cid(chunk)}: {exc}", file=sys.stderr)
        return None
    # LLM 可能返回空串/纯空白 → splitlines() 为空，需先过滤再取首行
    lines = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
    if not lines:
        return None
    q = lines[0].strip('"""「」 ').strip()
    if len(q) < 6 or q.startswith("无法"):
        return None
    return {"question": q, "gold_cids": [_cid(chunk)], "src_cid": _cid(chunk)}


async def build_task(task: dict) -> dict:
    from app.services.llm_service import LLMService
    # 语料缓存：抓取消耗 Tavily 额度且慢，缓存后调 prompt / 重新合成问题可零网络复跑
    cache = GOLDEN_DIR / f"_corpus_{task['id']}.json"
    if cache.exists():
        chunks = json.loads(cache.read_text(encoding="utf-8"))
        print(f"\n[{task['id']}] {task['name']} — 复用缓存语料 {len(chunks)} chunk", flush=True)
    else:
        print(f"\n[{task['id']}] {task['name']} — 跑管线抓取...", flush=True)
        t0 = time.perf_counter()
        chunks = await _run_pipeline(task)
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        print(f"    chunks={len(chunks)}  ({time.perf_counter() - t0:.0f}s，已缓存)", flush=True)

    raw_n = len(chunks)
    chunks = _clean_corpus(chunks)
    print(f"    markdown 清洗: {raw_n} → {len(chunks)} chunk（丢弃 {raw_n - len(chunks)} 样板）", flush=True)
    print("    合成问题中...", flush=True)
    picked = _select_chunks(chunks)
    llm = LLMService()
    qa_raw = await asyncio.gather(*[_gen_question(llm, c) for c in picked], return_exceptions=True)
    qa = [q for q in qa_raw if isinstance(q, dict)]

    # 黄金集语料：保留该任务全部 chunk（检索时它们都是候选/干扰项），单 gold 标注
    corpus = [
        {
            "cid": _cid(c), "source_id": c.get("source_id", ""),
            "chunk_index": c.get("chunk_index", 0), "url": c.get("url", ""),
            "title": c.get("title", ""), "text": c.get("text", ""),
        }
        for c in chunks if c.get("text")
    ]
    print(f"    → 语料 {len(corpus)} chunk，黄金问题 {len(qa)}/{len(picked)}", flush=True)
    return {
        "id": task["id"], "name": task["name"], "query": task["query"],
        "task_id": f"golden_{task['id']}", "corpus": corpus, "qa": qa,
    }


async def main(task_ids: list[str], out_name: str) -> None:
    from app.core.config import settings
    all_tasks = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    task_map = {t["id"]: t for t in all_tasks}
    selected = [task_map[t] for t in task_ids if t in task_map]
    if not selected:
        print("无有效任务", file=sys.stderr); sys.exit(1)

    print("═" * 72)
    print("  build_golden_set — 构建 RAG 检索评测黄金集")
    print(f"  任务: {[t['id'] for t in selected]}  embedding={settings.embedding_model}")
    print("═" * 72)
    await _warmup()

    results = []
    for task in selected:
        try:
            results.append(await build_task(task))
        except Exception as exc:
            print(f"    [{task['id']}] 失败: {exc}", file=sys.stderr)

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GOLDEN_DIR / out_name
    payload = {
        "built_at": datetime.now().isoformat(),
        "embedding_model": settings.embedding_model,
        "questions_per_task": QUESTIONS_PER_TASK,
        "tasks": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    total_q = sum(len(r["qa"]) for r in results)
    total_c = sum(len(r["corpus"]) for r in results)
    print(f"\n  ✅ 黄金集已保存: {out_path}")
    print(f"     {len(results)} 任务  /  {total_c} chunk 语料  /  {total_q} 条黄金问题")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 RAG 检索评测黄金集")
    parser.add_argument("task_ids", nargs="*", default=DEFAULT_TASKS, help="任务 ID（默认全部 12 个）")
    parser.add_argument("--out", default="golden_set.json", help="输出文件名（写入 benchmark/GoldenDataset/）")
    args = parser.parse_args()
    asyncio.run(main(args.task_ids, args.out))
