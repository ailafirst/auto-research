"""方案② 信源策略对比实验（仅测试，不改生产代码）。

对比三种"把权威证据灌进候选池"的策略——三者都**统一生效、不猜问题类型**，
相关性一律交给现有 bge-reranker 判定（自过滤）：

  baseline          : Tavily 通用搜索（= 现网行为）
  ①tavily_academic : baseline ∪ Tavily 限定静态学术白名单域（arxiv/pubmed/*.edu…）
  ②openalex        : baseline ∪ OpenAlex（2.5 亿论文，免 key，仅 httpx）

度量（对准诊断根因"评价型研究问题被市场报告淹没"）：
  - 候选池里"研究/学术类"信源占比 vs "市场/PR"占比
  - 经 rerank 后 top-10 里研究类占比（= 真正能到 analyst 面前的构成）
  - 各附加通道单独贡献了多少"新的研究源"

用法（在 deepresearch 环境，项目根目录）：
  D:\\conda\\envs\\deepresearch\\python.exe rag_experiments\\source_strategy_experiment.py
需要：tavily.txt（项目根，key 池）；model_server 在 8100（做 rerank，缺省则跳过 rerank 层）。
结果写到 rag_experiments/results/source_strategy_<ts>.md（UTF-8）。
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

# 让脚本能 import app.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.search_service import _load_tavily_keys  # noqa: E402

# ── 静态学术白名单域（① 用作 Tavily include_domains；分类器也用它判"研究类"）────────
ACADEMIC_DOMAINS = [
    "arxiv.org", "biorxiv.org", "medrxiv.org", "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov", "nature.com",
    "science.org", "sciencedirect.com", "springer.com", "link.springer.com",
    "ieee.org", "ieeexplore.ieee.org", "acm.org", "dl.acm.org", "cell.com",
    "pnas.org", "frontiersin.org", "mdpi.com", "plos.org", "elifesciences.org",
    "jstor.org", "wiley.com", "onlinelibrary.wiley.com", "tandfonline.com",
    "sagepub.com", "semanticscholar.org", "researchgate.net", "doi.org",
    "aps.org", "iop.org", "rsc.org", "acs.org", "oup.com",
]

# 市场报告 / PR 域（诊断案例里挤占候选池的那批）
MARKET_DOMAINS = [
    "marketsandmarkets", "snsinsider", "bccresearch", "polarismarketresearch",
    "datamintelligence", "grandviewresearch", "mordorintelligence",
    "fortunebusinessinsights", "alliedmarketresearch", "precedenceresearch",
    "researchandmarkets", "marketresearch", "market.us", "verifiedmarketresearch",
    "futuremarketinsights", "coherentmarketinsights", "globalmarketinsights",
]
PR_NEWS_DOMAINS = [
    "businesswire", "prnewswire", "globenewswire", "einpresswire", "openpr",
    "techcrunch", "venturebeat", "prweb", "yahoo.com/news",
]
BLOG_DOMAINS = ["medium.com", "substack.com", "linkedin.com", "blogspot", "wordpress"]
REFERENCE_DOMAINS = ["wikipedia.org", "investopedia.com", "britannica.com"]


def classify_source(url: str) -> str:
    """把一个来源 URL 归类到证据类型桶（启发式，用于看构成比例）。"""
    host = (urlparse(url).netloc or url).lower()
    full = url.lower()
    if host.endswith(".edu") or ".edu/" in full or host.endswith(".ac.uk") or host.endswith(".gov"):
        return "research"
    if any(d in host for d in ACADEMIC_DOMAINS):
        return "research"
    if any(d in host for d in MARKET_DOMAINS):
        return "market"
    if any(d in full for d in PR_NEWS_DOMAINS):
        return "pr_news"
    if any(d in host for d in REFERENCE_DOMAINS):
        return "reference"
    if any(d in host for d in BLOG_DOMAINS):
        return "blog"
    return "other"


# ── 测试问题（2 个研究/评价型 = 诊断靶心；1 个非研究型 = 对照，验证 ① 不污染）──────
QUESTIONS = [
    {
        "id": "bci",
        "kind": "research",
        "question": "2026 brain-computer interface latest research directions worth deep investment",
        "queries": [
            "brain-computer interface latest research directions 2026",
            "most promising brain-computer interface research directions worth investing",
        ],
    },
    {
        "id": "battery",
        "kind": "research",
        "question": "solid-state battery latest research breakthroughs and most promising directions 2026",
        "queries": [
            "solid-state battery latest research breakthroughs 2026",
            "most promising solid-state battery research directions",
        ],
    },
    {
        "id": "react",  # 对照：工程型问题，学术源应当很少、rerank 应自然过滤掉
        "kind": "control",
        "question": "React state management best practices 2026",
        "queries": [
            "React state management best practices 2026",
            "best React state management library 2026",
        ],
    },
]


# ── Tavily（复用生产 key 池；额度耗尽轮换）──────────────────────────────────────
_KEYS = _load_tavily_keys()
_key_idx = 0


async def tavily_search(query: str, include_domains: list[str] | None = None,
                        max_results: int = 10) -> list[dict]:
    """直连 Tavily（advanced，与现网一致），quota 耗尽轮换 key。返回结果 dict 列表。"""
    global _key_idx
    from tavily import AsyncTavilyClient
    from tavily.errors import ForbiddenError, InvalidAPIKeyError, UsageLimitExceededError

    for _ in range(len(_KEYS) + 1):
        if _key_idx >= len(_KEYS):
            print("  [tavily] 所有 key 耗尽")
            return []
        client = AsyncTavilyClient(api_key=_KEYS[_key_idx])
        try:
            kwargs = dict(query=query, max_results=max_results,
                          search_depth="advanced", include_answer=False)
            if include_domains:
                kwargs["include_domains"] = include_domains
            resp = await client.search(**kwargs)
            return [
                {"url": r.get("url", ""), "title": r.get("title", ""),
                 "text": f"{r.get('title', '')}. {r.get('content', '')}",
                 "tavily_score": r.get("score"), "origin": "tavily"}
                for r in resp.get("results", [])
            ]
        except (ForbiddenError, InvalidAPIKeyError):
            _key_idx += 1
            continue
        except UsageLimitExceededError:
            await asyncio.sleep(2)
            continue
        except Exception as exc:
            print(f"  [tavily] err: {exc}")
            return []
    return []


# ── OpenAlex（免 key）──────────────────────────────────────────────────────────
def _reconstruct_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos: dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


async def openalex_search(query: str, per_page: int = 10) -> list[dict]:
    """OpenAlex works 检索：近三年、按相关度。返回论文 dict（title+abstract 作 text）。"""
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per_page": per_page,
        "filter": "from_publication_date:2023-01-01",
        "mailto": "deepresearch-experiment@example.com",
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        print(f"  [openalex] err: {exc}")
        return []
    out: list[dict] = []
    for w in data.get("results", []):
        abstract = _reconstruct_abstract(w.get("abstract_inverted_index"))
        loc = (w.get("primary_location") or {})
        landing = loc.get("landing_page_url") or w.get("doi") or w.get("id", "")
        out.append({
            "url": landing, "title": w.get("title") or "",
            "text": f"{w.get('title') or ''}. {abstract}"[:1200],
            "cited_by": w.get("cited_by_count", 0),
            "year": w.get("publication_year"), "origin": "openalex",
        })
    return out


def _dedup(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for it in items:
        u = it.get("url", "")
        if u and u not in seen:
            seen.add(u)
            out.append(it)
    return out


def _compose(items: list[dict]) -> dict[str, int]:
    """按 source_type 统计构成。"""
    buckets: dict[str, int] = {}
    for it in items:
        t = classify_source(it.get("url", ""))
        it["_type"] = t
        buckets[t] = buckets.get(t, 0) + 1
    return buckets


def _pct_research(buckets: dict[str, int]) -> float:
    tot = sum(buckets.values()) or 1
    return 100.0 * buckets.get("research", 0) / tot


async def _maybe_rerank(question: str, items: list[dict], top_k: int = 10) -> list[dict] | None:
    """用现有 bge-reranker 重排（model_server 不可达则返回 None，跳过该层）。"""
    if not items:
        return []
    try:
        from app.services.rag_service import rerank_chunks
        return await rerank_chunks(question, items, top_k=top_k)
    except Exception as exc:
        print(f"  [rerank] 跳过（{exc}）")
        return None


async def run_question(q: dict, report: list[str]) -> dict:
    qid, question, queries = q["id"], q["question"], q["queries"]
    print(f"\n=== [{qid}] {question}")

    # baseline: Tavily 通用
    base_raw: list[dict] = []
    for sub in queries:
        base_raw += await tavily_search(sub)
    baseline = _dedup(base_raw)

    # ① 学术白名单路（单独），再与 baseline 合并
    acad_raw: list[dict] = []
    for sub in queries:
        acad_raw += await tavily_search(sub, include_domains=ACADEMIC_DOMAINS)
    academic_only = _dedup(acad_raw)
    strat1 = _dedup(baseline + academic_only)

    # ② OpenAlex（单独），再与 baseline 合并
    oa_raw: list[dict] = []
    for sub in queries:
        oa_raw += await openalex_search(sub)
    openalex_only = _dedup(oa_raw)
    strat2 = _dedup(baseline + openalex_only)

    cb, c1, c2 = _compose(baseline), _compose(strat1), _compose(strat2)
    ca, co = _compose(academic_only), _compose(openalex_only)

    # rerank 各策略 → top-10 构成
    async def topcompose(items):
        top = await _maybe_rerank(question, [dict(x) for x in items], top_k=10)
        if top is None:
            return None, None
        return _compose(top), top

    tb, _ = await topcompose(baseline)
    t1, _ = await topcompose(strat1)
    t2, _ = await topcompose(strat2)

    # 写报告段
    report.append(f"\n## [{qid}] {question}\n")
    report.append(f"- 子查询: {queries}")
    report.append("")
    report.append("| 策略 | 候选池大小 | 研究类 | 市场类 | PR类 | 博客/其它 | 研究占比 | rerank top10 研究占比 |")
    report.append("|---|---|---|---|---|---|---|---|")

    def row(name, items, comp, topcomp):
        n = len(items)
        research = comp.get("research", 0)
        market = comp.get("market", 0)
        pr = comp.get("pr_news", 0)
        rest = n - research - market - pr
        toppct = "-" if topcomp is None else f"{_pct_research(topcomp):.0f}%"
        report.append(f"| {name} | {n} | {research} | {market} | {pr} | {rest} | "
                      f"{_pct_research(comp):.0f}% | {toppct} |")

    row("baseline (现网)", baseline, cb, tb)
    row("① baseline+Tavily学术", strat1, c1, t1)
    row("② baseline+OpenAlex", strat2, c2, t2)
    report.append("")
    report.append(f"- 附加通道单独贡献：① 学术白名单路 {len(academic_only)} 条"
                  f"（研究占比 {_pct_research(ca):.0f}%）；② OpenAlex {len(openalex_only)} 条"
                  f"（研究占比 {_pct_research(co):.0f}%）")
    # baseline 里有多少市场/PR（诊断案例的病）
    report.append(f"- baseline 构成明细: {cb}")

    return {
        "qid": qid, "kind": q["kind"],
        "baseline_research_pct": _pct_research(cb),
        "s1_research_pct": _pct_research(c1),
        "s2_research_pct": _pct_research(c2),
        "s1_top_pct": None if t1 is None else _pct_research(t1),
        "s2_top_pct": None if t2 is None else _pct_research(t2),
        "base_top_pct": None if tb is None else _pct_research(tb),
        "acad_added": len(academic_only),
        "oa_added": len(openalex_only),
    }


async def main() -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    report: list[str] = [
        "# 方案② 信源策略对比实验",
        f"> 时间: {ts}  |  仅测试，未改生产代码",
        "> baseline=Tavily通用 · ①=+Tavily学术白名单 · ②=+OpenAlex；三者统一生效不猜类型，rerank 自过滤",
    ]
    if not _KEYS:
        report.append("\n**警告：tavily.txt 无 key，baseline/① 将为空**")

    summary: list[dict] = []
    for q in QUESTIONS:
        summary.append(await run_question(q, report))

    # 汇总
    report.append("\n---\n\n## 汇总\n")
    report.append("| 问题 | 类型 | 候选池研究占比 base→①→② | rerank top10 研究占比 base→①→② | ①加源 | ②加源 |")
    report.append("|---|---|---|---|---|---|")
    for s in summary:
        def f(x):
            return "-" if x is None else f"{x:.0f}%"
        report.append(
            f"| {s['qid']} | {s['kind']} | "
            f"{s['baseline_research_pct']:.0f}%→{s['s1_research_pct']:.0f}%→{s['s2_research_pct']:.0f}% | "
            f"{f(s['base_top_pct'])}→{f(s['s1_top_pct'])}→{f(s['s2_top_pct'])} | "
            f"+{s['acad_added']} | +{s['oa_added']} |"
        )

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"source_strategy_{ts}.md"
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"\nWROTE {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
