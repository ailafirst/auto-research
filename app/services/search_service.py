"""搜索服务 — 支持多种搜索引擎。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.exceptions import SearchServiceError
from app.models.source import SearchResult

logger = logging.getLogger(__name__)


class TavilyRateLimitError(Exception):
    """Tavily 限速错误 — 可被 tenacity 捕获并按指数退避重试。"""


class BaseSearchProvider:
    """搜索提供者基类。"""

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        raise NotImplementedError


class DuckDuckGoProvider(BaseSearchProvider):
    """DuckDuckGo 搜索（免费，适合开发环境）。"""

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        try:
            from ddgs import DDGS

            def _search() -> list[dict[str, Any]]:
                ddgs = DDGS()
                return list(ddgs.text(query=query, max_results=max_results))

            raw_results = await asyncio.to_thread(_search)

            results: list[SearchResult] = []
            for i, item in enumerate(raw_results):
                title = item.get("title", "")
                href = item.get("href", "")
                snippet = item.get("body", "")

                if not href:
                    continue
                if href.startswith("//"):
                    href = "https:" + href

                results.append(SearchResult(
                    title=title or "无标题",
                    url=href,
                    snippet=snippet or "",
                    position=i + 1,
                ))

            return results

        except ImportError:
            logger.warning("duckduckgo_search 未安装，请执行: pip install duckduckgo_search")
            return []
        except Exception as exc:
            logger.warning("DuckDuckGo 搜索失败: %s", exc)
            return []


_tavily_semaphore = asyncio.Semaphore(3)   # 限制 Tavily 全局并发，避免 dev key 限速


class TavilyProvider(BaseSearchProvider):
    """Tavily Search API 搜索。"""

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        if not settings.tavily_api_key:
            logger.warning("Tavily API Key 未配置")
            return []

        async with _tavily_semaphore:
            try:
                from tavily import AsyncTavilyClient

                client = AsyncTavilyClient(api_key=settings.tavily_api_key)
                response = await client.search(
                    query=query,
                    max_results=max_results,
                    search_depth="advanced",
                    include_raw_content=True,
                    include_answer=True,
                )

                answer: str | None = response.get("answer") or None

                results: list[SearchResult] = []
                for i, item in enumerate(response.get("results", [])):
                    results.append(SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=item.get("content", ""),
                        position=i + 1,
                        raw_content=item.get("raw_content") or None,
                        tavily_score=item.get("score"),
                        query_answer=answer,
                    ))

                return results

            except Exception as exc:
                msg = str(exc)
                # 限速错误上抛，让 tenacity 按指数退避重试
                if any(k in msg.lower() for k in ("excessive", "rate", "blocked", "429", "too many")):
                    raise TavilyRateLimitError(msg) from exc
                logger.error("Tavily 搜索失败: %s", exc)
                return []
            finally:
                # 每次请求后短暂释放，避免连续触发限速
                await asyncio.sleep(0.3)


class SearchService:
    """搜索服务 — 统一入口，USE_TAVILY 开关选择引擎。"""

    def __init__(self) -> None:
        self.providers: dict[str, BaseSearchProvider] = {
            "duckduckgo": DuckDuckGoProvider(),
            "tavily": TavilyProvider(),
        }
        self._current_provider = "tavily" if settings.use_tavily else "duckduckgo"

    @property
    def provider(self) -> BaseSearchProvider:
        return self.providers[self._current_provider]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(TavilyRateLimitError),
        reraise=True,
    )
    async def _primary_search(self, query: str, max_results: int) -> list[SearchResult]:
        """调用主引擎，限速时由 tenacity 按指数退避自动重试（最多 3 次）。"""
        return await self.provider.search(query, max_results=max_results)

    async def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        """执行单次搜索，失败或零结果时自动降级到 DuckDuckGo。"""
        n_results = max_results or settings.max_search_results

        try:
            results = await self._primary_search(query, n_results)
        except TavilyRateLimitError:
            logger.warning("Tavily 限速重试耗尽，降级 DDG: query='%s'", query[:50])
            results = []

        # 零结果降级：Tavily 返回空（限速耗尽 or 真实无结果）→ DuckDuckGo 补充
        if not results and self._current_provider == "tavily":
            logger.warning("Tavily 零结果，DDG 降级: query='%s'", query[:50])
            results = await self.providers["duckduckgo"].search(query, max_results=n_results)

        engine = self._current_provider if results else "duckduckgo(fallback)"
        logger.info("搜索完成 [%s]: query='%s', results=%d", engine, query[:50], len(results))
        return results

    async def multi_search(
        self,
        queries: list[str],
        max_results_per_query: int = 5,
        concurrency: int | asyncio.Semaphore = 3,
    ) -> list[SearchResult]:
        """执行多关键词搜索并去重。
        concurrency 可传入 int（内部创建 Semaphore）或外部共享的 asyncio.Semaphore，
        后者用于跨多个 multi_search 调用共享全局并发上限。
        """
        semaphore = concurrency if isinstance(concurrency, asyncio.Semaphore) else asyncio.Semaphore(concurrency)

        async def _throttled(q: str) -> list[SearchResult]:
            async with semaphore:
                return await self.search(q, max_results=max_results_per_query)

        tasks = [_throttled(q) for q in queries]
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        seen_urls: set[str] = set()
        all_results: list[SearchResult] = []

        for results in results_lists:
            if isinstance(results, Exception):
                logger.warning("搜索异常: %s", results)
                continue
            for r in results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    all_results.append(r)

        return all_results
