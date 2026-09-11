"""学术检索客户端 —— 差距 #4（结构化子域参数契约）的具体落地，academic 路由下

对比过 Semantic Scholar 和 CrossRef 两个免费学术 API（真实请求实测，见
docs/检索路由蓝图.md §9.15）：Semantic Scholar 匿名池持续返回 429（官方错误信息
明确要求申请 API key 才能获得可用限速），CrossRef 无需注册即可稳定使用，DOI
覆盖率 5/5。默认相关性排序明显优于按被引用数排序——按被引用数排序会把无关但
高被引的论文排到最前面（如 "vLLM inference" 查询排出一篇 DNA 测序论文，因为
两者都含 "inference" 一词但被引数悬殊）；因此固定不传 sort 参数。
`filter=has-abstract:true` 能把摘要覆盖率从约 1/5 提到 5/5，直接决定选它。

这不是又一份"域名白名单 + Tavily"式的垂直路由：返回的 DOI、被引次数、期刊、
发表年份是结构化字段，不是网页搜索摘要，对应 AnySearch sub_domain_params 的
落地方式（形态更简单——CrossRef 是查询参数而非 AnySearch 那种领域专属结构化
过滤条件，但同样是"结构化数据源"而非"网页检索+域名过滤"）。

新增 arXiv（见 docs/检索路由蓝图.md §9.16）：CrossRef 索引的是已经正式发表/
带 DOI 的文献，对 vLLM PagedAttention 这类小众前沿系统研究的覆盖天然滞后——
这类工作往往先挂 arXiv 预印本，几个月甚至一两年后才正式发表（或者永远不发表，
只挂会议 workshop）。arXiv 官方 API（export.arxiv.org，同样免费不用注册）直接
补上这块，两者是互补关系而非替代：arXiv 没有被引数字段（arXiv 本身不追踪引用），
也没有同行评审，_authority_score 里因此退到与其它垂直白名单命中同档的 0.75，
不单独抬高或压低——目前没有真实标注数据支撑"该给预印本打多少折扣"这个具体数字，
如实按经验默认值处理，不假装算出来的分数有精确依据。
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import xml.etree.ElementTree as ET

import httpx

from app.core.config import settings
from app.models.source import SearchResult
from app.services.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)

CROSSREF_API_URL = "https://api.crossref.org/works"
# CrossRef "polite pool"：带 mailto 参数的请求会被路由到更快更稳定的池子，
# 官方文档建议加，不加也能用但没有稳定性保证。
_POLITE_MAILTO = "deepresearch-poc@example.com"

# 进程级并发闸门：见 docs/检索路由蓝图.md §9.18，真实基准并发下两个源都被大量
# 429（CrossRef 41/43、arXiv 68/68）。只在真正发起 HTTP 请求时占用槽位，与
# llm_service.py 的 _llm_semaphore 是同一模式——跨任务并发时靠它压住同一时刻
# 在飞的请求数，不是限制总请求量。
#
# 光有本地信号量不够：§9.20/§9.21 真实数据测出——CrossRef 是"瞬时并发连接数"型
# 限速（≤8 个稳定成功），本地 semaphore=3 已在安全范围内，但多 worker 部署下
# 各 worker 独立持有一份，聚合并发仍可能超限；arXiv 是"短时间窗口累计请求数"型
# 限速（约 20 个/7 秒即触发，触发后冷却期约 5 分钟），本地并发上限对这种"发得
# 够多就撞线"的模式完全不设防，实测调小并发数无效。因此在信号量之前再加一层
# Redis 全局令牌桶（同 llm_service.py 的 _llm_semaphore 前置 get_rate_limiter()
# 那一套），令牌等待期间不占本地并发槽位；Redis 不可用时 acquire 直接放行，
# 由本地信号量兜底（降级安全）。
_crossref_semaphore = asyncio.Semaphore(settings.crossref_max_concurrency)
_arxiv_semaphore = asyncio.Semaphore(settings.arxiv_max_concurrency)

# arXiv 熔断：见 docs/检索路由蓝图.md §9.25——分速率压力测试确认 0.33/s 这个稳态
# 节奏本身没问题（单流串行 24 个 0 失败，生产同款"并发=2+0.33/s"发 40 个也 0 失败），
# 但一旦真的进入受限状态，冷却期是硬下限：单流串行、每次都规规矩矩等 ≥3 秒，连续
# 80 个请求、持续 240 秒，因为起跑时已在受限状态里，全部失败，拼节奏换不回时间。
# 既然等不来，与其让排在后面的请求继续走令牌桶傻等一个大概率仍是 429 的结果，不如
# 收到第一次 429 就短路：接下来一段冷却窗口内直接跳过 arXiv（只退回 CrossRef），
# 省下这段时间里注定失败的排队等待成本。只是进程内的本地状态，不追求跨 worker
# 共享——单个 worker 最多在冷却窗口开始前多付一次探测代价，不影响正确性。
_arxiv_circuit_open_until: float = 0.0

_TAG_RE = re.compile(r"<[^>]+>")


def _clean_abstract(raw: str) -> str:
    """CrossRef 摘要字段常带 JATS XML 标签（如 <jats:p>…</jats:p>），去标签只留文本。"""
    return _TAG_RE.sub(" ", raw or "").strip()


def _extract_year(item: dict) -> int | None:
    for key in ("published-print", "published-online", "published", "created"):
        date_parts = (item.get(key) or {}).get("date-parts")
        if date_parts and date_parts[0]:
            return date_parts[0][0]
    return None


async def search_crossref(query: str, max_results: int = 5) -> list[SearchResult]:
    """查询 CrossRef 学术文献库，返回带 DOI/被引数/期刊/年份的 SearchResult。

    只用默认相关性排序（不传 sort 参数，理由见模块 docstring），加
    filter=has-abstract:true 保证返回的记录都带摘要正文，下游可直接当
    raw_content 用、不必再爬一次。失败静默返回空列表（降级安全，不阻断
    retriever_node 里并发的其余检索路），与 search_service.py 里其他检索
    方法的失败处理方式一致。
    """
    params = {
        "query": query,
        "rows": max_results,
        "mailto": _POLITE_MAILTO,
        "filter": "has-abstract:true",
    }
    try:
        if settings.crossref_rate_limit_enabled:
            await get_rate_limiter().acquire(
                "crossref", settings.crossref_rate_limit_per_sec, settings.crossref_rate_burst,
            )
        async with _crossref_semaphore:
            async with httpx.AsyncClient(trust_env=True) as client:
                resp = await client.get(CROSSREF_API_URL, params=params, timeout=15.0)
                resp.raise_for_status()
    except Exception as exc:
        logger.warning("CrossRef 查询失败 [%s]: %s", query[:40], exc)
        return []

    try:
        items = resp.json().get("message", {}).get("items", [])
    except Exception as exc:
        logger.warning("CrossRef 响应解析失败 [%s]: %s", query[:40], exc)
        return []

    results: list[SearchResult] = []
    for i, item in enumerate(items):
        title = " ".join(item.get("title") or []) or "(无标题)"
        doi = item.get("DOI") or None
        url = item.get("URL") or (f"https://doi.org/{doi}" if doi else "")
        if not url:
            continue
        abstract = _clean_abstract(item.get("abstract", ""))
        results.append(SearchResult(
            title=title,
            url=url,
            snippet=abstract[:300],
            raw_content=abstract or None,
            position=i,
            provider="crossref",
            search_route="academic-structured",
            doi=doi,
            citation_count=item.get("is-referenced-by-count"),
            venue=(item.get("container-title") or [None])[0],
            published_year=_extract_year(item),
        ))
    logger.info("CrossRef[%s]: 返回 %d 条（含摘要）", query[:40], len(results))
    return results


ARXIV_API_URL = "https://export.arxiv.org/api/query"  # 官方文档给的是 http，实测已强制跳转 https
_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def _clean_text(raw: str | None) -> str:
    """arXiv Atom 里 title/summary 常带排版换行/多余空白，折叠成单行。"""
    return " ".join((raw or "").split())


async def _query_arxiv(search_query: str, max_results: int) -> list[SearchResult] | None:
    """发一次 arXiv API 请求并解析为 SearchResult 列表。

    返回 None 表示请求/解析本身失败（网络错误、非 200、XML 解析异常）；返回
    `[]`（可能是空列表）表示请求成功但确实 0 条命中——调用方 `search_arxiv`
    要靠这个区分来决定该不该换策略重试，不能把"失败"和"真的没有"混为一谈。
    """
    global _arxiv_circuit_open_until
    remaining = _arxiv_circuit_open_until - time.monotonic()
    if remaining > 0:
        logger.debug("arXiv 熔断中（预计 %.0fs 后解除），跳过 [%s]", remaining, search_query[:60])
        return None

    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        if settings.arxiv_rate_limit_enabled:
            await get_rate_limiter().acquire(
                "arxiv", settings.arxiv_rate_limit_per_sec, settings.arxiv_rate_burst,
                max_wait=settings.arxiv_rate_limit_max_wait,
            )
        async with _arxiv_semaphore:
            async with httpx.AsyncClient(trust_env=True, follow_redirects=True) as client:
                resp = await client.get(ARXIV_API_URL, params=params, timeout=15.0)
                resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            _arxiv_circuit_open_until = time.monotonic() + settings.arxiv_circuit_breaker_cooldown
            logger.warning(
                "arXiv 429，熔断 %.0fs [%s]", settings.arxiv_circuit_breaker_cooldown, search_query[:60],
            )
        else:
            logger.warning("arXiv 查询失败 [%s]: %s", search_query[:60], exc)
        return None
    except Exception as exc:
        logger.warning("arXiv 查询失败 [%s]: %s", search_query[:60], exc)
        return None

    try:
        root = ET.fromstring(resp.text)
    except Exception as exc:
        logger.warning("arXiv 响应解析失败 [%s]: %s", search_query[:60], exc)
        return None

    results: list[SearchResult] = []
    for i, entry in enumerate(root.findall(f"{_ATOM_NS}entry")):
        id_el = entry.find(f"{_ATOM_NS}id")
        url = (id_el.text or "").strip() if id_el is not None else ""
        if not url:
            continue
        title = _clean_text(entry.findtext(f"{_ATOM_NS}title")) or "(无标题)"
        abstract = _clean_text(entry.findtext(f"{_ATOM_NS}summary"))

        published_year = None
        published_raw = entry.findtext(f"{_ATOM_NS}published")
        if published_raw:
            try:
                published_year = int(published_raw[:4])
            except ValueError:
                pass

        primary_cat_el = entry.find(f"{_ARXIV_NS}primary_category")
        category = primary_cat_el.get("term") if primary_cat_el is not None else None
        doi = entry.findtext(f"{_ARXIV_NS}doi")  # 只有后来正式发表过的预印本才有

        results.append(SearchResult(
            title=title,
            url=url,
            snippet=abstract[:300],
            raw_content=abstract or None,
            position=i,
            provider="arxiv",
            search_route="academic-structured",
            doi=doi.strip() if doi else None,
            citation_count=None,
            venue=category,
            published_year=published_year,
        ))
    return results


async def search_arxiv(query: str, max_results: int = 5) -> list[SearchResult]:
    """查询 arXiv 官方 API，返回带分类/年份的预印本 SearchResult。

    两级策略，真实请求测出来的（见 docs/检索路由蓝图.md §9.16）：先按 " AND "
    把每个词强制连接做精确匹配——短查询（2 词）效果很好，能把目标论文排到
    第一（如 "vLLM PagedAttention"）；但 Planner 生成的查询常有 4 个以上关键词，
    要求全部同时出现在标题/摘要里过于严格，实测"vLLM PagedAttention inference
    acceleration"四词 AND 直接 0 命中（真实存在且相关的论文只含其中 2 个词）。
    因此 AND 精确匹配 0 条时降级为裸词拼接的宽松匹配——arXiv 对同字段内的多个
    裸词是隐式 OR 语义，召回面更宽但相关性更松（实测同一四词查询宽松匹配后
    total 命中数从 0 冲到 20 万+，目标论文排进前 5 但不再是断层领先）。
    两级都不生效才真的判定"arXiv 没有相关结果"，返回空列表。

    sortBy=relevance 与 CrossRef 不传 sort 走默认相关性排序是同一个考虑：要的是
    "和查询主题匹配"而非"最新"或"最多引用"。没有 citation_count（arXiv 不追踪
    引用），venue 字段借用存放 primary_category（如 "cs.LG"），不是真的期刊名，
    下游展示时需按 provider 区分对待。
    """
    words = query.split()
    and_query = "all:" + " AND ".join(words)
    results = await _query_arxiv(and_query, max_results)
    if results is None:
        return []  # 请求本身失败，不拿"没结果"的宽松匹配去掩盖真实的网络/解析错误
    if results:
        logger.info("arXiv[%s]: AND 精确匹配返回 %d 条", query[:40], len(results))
        return results

    loose_query = "all:" + " ".join(words)
    results = await _query_arxiv(loose_query, max_results)
    if results is None:
        return []
    logger.info("arXiv[%s]: AND 精确匹配 0 条，宽松匹配返回 %d 条", query[:40], len(results))
    return results
