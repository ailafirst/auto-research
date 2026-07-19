"""增强 reranker 短板实验（仅测试，不改生产代码）。

背景：source_strategy 实验发现——② OpenAlex 把论文灌进候选池，却在全局 rerank 后被
筛出 top-10（bge-reranker 偏爱"网页"文本、压制"论文摘要"）。本实验做两件事：

  1) 诊断：逐条 rerank 分数按"车道"(lane) 分布，验证论文摘要是否被系统性压低
       lane: web(普通网页) / academic(学术域名网页) / openalex(论文标题+摘要)
  2) 试补偿：在同一 enriched 候选池上，比较 5 种"融合/校准"策略的 top-10 构成，
     同时用对照题(react)检查——补偿是否破坏了非研究题的 self-filtering(不该把
     学术源硬塞进工程题)。

策略：
  P0 global      : 现状，按原始 rerank 分全局排序
  P1 minmax      : 每条按其 lane 内 min-max 归一化后再全局排序（去除车道尺度差）
  P1z zscore     : 每条按其 lane 内 z-score 归一化后再全局排序
  P2 rrf         : 各 lane 内按分排名，RRF(1/(60+rank)) 融合（尺度无关，保证代表性）
  P3 quota       : 全局排序但 top-10 保底 >=4 个研究源（硬配额）

用法：D:\\conda\\envs\\deepresearch\\python.exe rag_experiments\\rerank_enhance_experiment.py
依赖：tavily.txt；model_server 在 8100（rerank 打分）。结果写 results/rerank_enhance_<ts>.md
"""

from __future__ import annotations

import asyncio
import statistics as stats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 复用第一支实验的检索与分类工具（导入不会触发其 main）
from source_strategy_experiment import (  # noqa: E402
    ACADEMIC_DOMAINS, QUESTIONS, _dedup, classify_source,
    openalex_search, tavily_search,
)

RRF_K = 60
QUOTA_RESEARCH = 4   # P3 top-10 保底研究源数
TOP_K = 10


def lane_of(item: dict) -> str:
    if item.get("origin") == "openalex":
        return "openalex"
    return "academic" if classify_source(item.get("url", "")) == "research" else "web"


def is_research(item: dict) -> bool:
    return item.get("_lane") in ("academic", "openalex")


async def rerank_scores(query: str, texts: list[str]) -> list[float]:
    """优先走模型服务，失败回退本地 CrossEncoder。"""
    from app.services.rag_service import _rerank_scores, _rerank_via_service
    try:
        return await _rerank_via_service(query, texts)
    except Exception:
        return await _rerank_scores(query, texts)


def _research_pct(items: list[dict]) -> float:
    if not items:
        return 0.0
    return 100.0 * sum(1 for x in items if is_research(x)) / len(items)


def _lane_counts(items: list[dict]) -> dict[str, int]:
    c: dict[str, int] = {"web": 0, "academic": 0, "openalex": 0}
    for x in items:
        c[x["_lane"]] = c.get(x["_lane"], 0) + 1
    return c


# ── 融合策略：输入 (item, raw_score) 列表，输出 top-K 排序 ────────────────────────
def policy_global(scored: list[tuple[dict, float]]) -> list[dict]:
    return [it for it, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:TOP_K]]


def _normalize_per_lane(scored, mode: str):
    by_lane: dict[str, list[float]] = {}
    for it, s in scored:
        by_lane.setdefault(it["_lane"], []).append(s)
    lane_stat = {}
    for lane, ss in by_lane.items():
        if mode == "minmax":
            lo, hi = min(ss), max(ss)
            lane_stat[lane] = (lo, hi)
        else:  # zscore
            mu = stats.mean(ss)
            sd = stats.pstdev(ss) or 1e-9
            lane_stat[lane] = (mu, sd)
    out = []
    for it, s in scored:
        lane = it["_lane"]
        if mode == "minmax":
            lo, hi = lane_stat[lane]
            ns = (s - lo) / (hi - lo) if hi > lo else 0.5
        else:
            mu, sd = lane_stat[lane]
            ns = (s - mu) / sd
        out.append((it, ns))
    return out


def policy_norm(scored, mode: str) -> list[dict]:
    normed = _normalize_per_lane(scored, mode)
    return [it for it, _ in sorted(normed, key=lambda x: x[1], reverse=True)[:TOP_K]]


def policy_rrf(scored: list[tuple[dict, float]]) -> list[dict]:
    by_lane: dict[str, list[tuple[dict, float]]] = {}
    for it, s in scored:
        by_lane.setdefault(it["_lane"], []).append((it, s))
    rrf: dict[int, float] = {}
    idmap: dict[int, dict] = {}
    for lane, arr in by_lane.items():
        arr.sort(key=lambda x: x[1], reverse=True)
        for rank, (it, _) in enumerate(arr):
            key = id(it)
            idmap[key] = it
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (RRF_K + rank)
    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    return [idmap[k] for k, _ in ranked]


def policy_quota(scored: list[tuple[dict, float]]) -> list[dict]:
    ordered = [it for it, _ in sorted(scored, key=lambda x: x[1], reverse=True)]
    top = ordered[:TOP_K]
    have = sum(1 for x in top if is_research(x))
    if have >= QUOTA_RESEARCH:
        return top
    # 用池中下一批最高分研究源，替换 top 里最低分的非研究源
    research_pool = [x for x in ordered if is_research(x) and x not in top]
    non_research_in_top = [x for x in reversed(top) if not is_research(x)]  # 从低分端替换
    need = QUOTA_RESEARCH - have
    for i in range(min(need, len(research_pool), len(non_research_in_top))):
        top[top.index(non_research_in_top[i])] = research_pool[i]
    return top


def _overlap(a: list[dict], b: list[dict]) -> int:
    ua = {x.get("url") for x in a}
    return sum(1 for x in b if x.get("url") in ua)


async def run_question(q: dict, report: list[str]) -> dict:
    qid, question, queries = q["id"], q["question"], q["queries"]
    print(f"\n=== [{qid}] {question}")

    base: list[dict] = []
    acad: list[dict] = []
    oa: list[dict] = []
    for sub in queries:
        base += await tavily_search(sub)
        acad += await tavily_search(sub, include_domains=ACADEMIC_DOMAINS)
        oa += await openalex_search(sub)

    pool = _dedup(base + acad + oa)
    for it in pool:
        it["_lane"] = lane_of(it)

    # rerank 打分
    scores = await rerank_scores(question, [it.get("text", "")[:1024] for it in pool])
    scored = list(zip(pool, scores))

    # 诊断：各 lane 分数分布
    lane_scores: dict[str, list[float]] = {}
    for it, s in scored:
        lane_scores.setdefault(it["_lane"], []).append(s)

    report.append(f"\n## [{qid}] {question}\n")
    report.append(f"- 候选池 {len(pool)} 条，lane 构成 {_lane_counts(pool)}")
    report.append("")
    report.append("### 诊断：各 lane 的 rerank 原始分分布")
    report.append("| lane | n | mean | median | max | min |")
    report.append("|---|---|---|---|---|---|")
    for lane in ("web", "academic", "openalex"):
        ss = lane_scores.get(lane, [])
        if ss:
            report.append(f"| {lane} | {len(ss)} | {stats.mean(ss):.3f} | "
                          f"{stats.median(ss):.3f} | {max(ss):.3f} | {min(ss):.3f} |")
        else:
            report.append(f"| {lane} | 0 | - | - | - | - |")

    # 各策略 top-10
    policies = {
        "P0 global": policy_global(scored),
        "P1 minmax": policy_norm(scored, "minmax"),
        "P1z zscore": policy_norm(scored, "zscore"),
        "P2 rrf": policy_rrf(scored),
        "P3 quota>=4": policy_quota(scored),
    }
    p0 = policies["P0 global"]

    report.append("")
    report.append("### 各融合策略 top-10 构成")
    report.append("| 策略 | 研究占比 | web | academic | openalex | 与P0重叠 |")
    report.append("|---|---|---|---|---|---|")
    res = {}
    for name, top in policies.items():
        lc = _lane_counts(top)
        res[name] = _research_pct(top)
        report.append(f"| {name} | {_research_pct(top):.0f}% | {lc['web']} | "
                      f"{lc['academic']} | {lc['openalex']} | {_overlap(p0, top)}/{TOP_K} |")

    return {"qid": qid, "kind": q["kind"], "res": res,
            "lane_means": {k: (stats.mean(v) if v else None) for k, v in lane_scores.items()}}


async def main() -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    report = [
        "# 增强 reranker 短板实验",
        f"> 时间: {ts} | 仅测试，未改生产代码",
        "> 目标：让权威研究源(尤其 OpenAlex 摘要)不被全局 rerank 一把筛掉，同时不污染非研究题",
    ]
    summary = []
    for q in QUESTIONS:
        summary.append(await run_question(q, report))

    report.append("\n---\n\n## 汇总：各策略 top-10 研究占比\n")
    names = ["P0 global", "P1 minmax", "P1z zscore", "P2 rrf", "P3 quota>=4"]
    report.append("| 问题 | 类型 | " + " | ".join(names) + " |")
    report.append("|---|---|" + "|".join(["---"] * len(names)) + "|")
    for s in summary:
        row = " | ".join(f"{s['res'][n]:.0f}%" for n in names)
        report.append(f"| {s['qid']} | {s['kind']} | {row} |")

    report.append("\n## 汇总：各 lane 平均 rerank 分（诊断系统性压制）\n")
    report.append("| 问题 | web | academic | openalex |")
    report.append("|---|---|---|---|")
    for s in summary:
        lm = s["lane_means"]
        def g(k):
            return "-" if lm.get(k) is None else f"{lm[k]:.3f}"
        report.append(f"| {s['qid']} | {g('web')} | {g('academic')} | {g('openalex')} |")

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"rerank_enhance_{ts}.md"
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"\nWROTE {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
