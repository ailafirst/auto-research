"""搜索服务 — 支持多种搜索引擎。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.exceptions import SearchServiceError
from app.models.source import SearchResult

logger = logging.getLogger(__name__)


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


class TavilyProvider(BaseSearchProvider):
    """Tavily Search API 搜索。"""

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        if not settings.tavily_api_key:
            logger.warning("Tavily API Key 未配置")
            return []

        try:
            from tavily import AsyncTavilyClient

            client = AsyncTavilyClient(api_key=settings.tavily_api_key)
            response = await client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
            )

            results: list[SearchResult] = []
            for i, item in enumerate(response.get("results", [])):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    position=i + 1,
                ))

            return results

        except Exception as exc:
            logger.error("Tavily 搜索失败: %s", exc)
            return []


class SearchService:
    """搜索服务 — 统一入口，自动选择搜索引擎。"""

    def __init__(self) -> None:
        self.providers: dict[str, BaseSearchProvider] = {
            "duckduckgo": DuckDuckGoProvider(),
            "tavily": TavilyProvider(),
        }
        self._current_provider = settings.search_provider

    @property
    def provider(self) -> BaseSearchProvider:
        provider = self.providers.get(self._current_provider)
        if not provider:
            logger.warning("不支持的搜索提供者 '%s'，回退到 DuckDuckGo", self._current_provider)
            provider = self.providers["duckduckgo"]
        return provider

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        """执行单次搜索。"""
        n_results = max_results or settings.max_search_results
        results = await self.provider.search(query, max_results=n_results)
        logger.info("搜索完成: query='%s', results=%d", query[:50], len(results))
        return results

    async def multi_search(
        self,
        queries: list[str],
        max_results_per_query: int = 5,
    ) -> list[SearchResult]:
        """执行多关键词搜索并去重。"""
        tasks = [self.search(q, max_results=max_results_per_query) for q in queries]
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
