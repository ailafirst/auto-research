"""搜索服务 — 支持多种搜索引擎。"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.exceptions import SearchServiceError
from app.models.source import SearchResult

logger = logging.getLogger(__name__)


class TavilyRateLimitError(Exception):
    """Tavily 限速错误（429）— 可被 tenacity 捕获并按指数退避重试。"""


class TavilyQuotaError(Exception):
    """Tavily key 失效或月配额耗尽（432/403/401）— 需要切换到下一个 key。"""

    def __init__(self, msg: str, key_idx: int = 0) -> None:
        super().__init__(msg)
        self.key_idx = key_idx


def _load_tavily_keys() -> list[str]:
    """从项目根目录 tavily.txt 加载 API key 列表。"""
    key_file = Path(__file__).parent.parent.parent / "tavily.txt"
    if not key_file.exists():
        return []
    keys: list[str] = []
    for line in key_file.read_text(encoding="utf-8").splitlines():
        m = re.search(r"(tvly-\S+)", line.strip())
        if m:
            keys.append(m.group(1))
    return keys


class TavilyKeyPool:
    """Tavily API key 顺序轮换池 — 额度耗尽时切换到下一个 key。"""

    def __init__(self, keys: list[str]) -> None:
        self._keys = keys
        self._idx = 0
        if keys:
            logger.info("Tavily key pool 已加载 %d 个 key", len(keys))

    @property
    def current_key(self) -> str | None:
        return self._keys[self._idx] if self._idx < len(self._keys) else None

    @property
    def current_idx(self) -> int:
        return self._idx

    def rotate(self, from_idx: int) -> str | None:
        """将 from_idx 处的 key 标记为耗尽并切换到下一个。
        若 _idx 已超过 from_idx（并发场景下其他协程已先轮换），则幂等跳过。
        """
        if self._idx != from_idx:
            return self.current_key
        self._idx += 1
        if self._idx < len(self._keys):
            logger.warning(
                "Tavily key %d 额度耗尽，切换至 key %d/%d",
                from_idx + 1, self._idx + 1, len(self._keys),
            )
            return self._keys[self._idx]
        logger.error("Tavily 所有 %d 个 key 额度已耗尽", len(self._keys))
        return None


_key_pool: TavilyKeyPool | None = None


def _get_key_pool() -> TavilyKeyPool:
    global _key_pool
    if _key_pool is None:
        _key_pool = TavilyKeyPool(_load_tavily_keys())
    return _key_pool


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
    """Tavily Search API 搜索，支持多 key 顺序轮换。"""

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        pool = _get_key_pool()
        # 在进入 semaphore 前记录本次请求使用的 key index，随错误一起传出
        # 这样多个并发任务失败时，rotate(key_idx) 都用同一个 from_idx，
        # 幂等检查生效，只发生一次真正的轮换，避免级联耗尽所有 key。
        key_idx = pool.current_idx
        api_key = pool.current_key or settings.tavily_api_key
        if not api_key:
            logger.warning("Tavily API Key 未配置（tavily.txt 和 .env 均无有效 key）")
            return []

        async with _tavily_semaphore:
            try:
                from tavily import AsyncTavilyClient

                client = AsyncTavilyClient(api_key=api_key)
                response = await client.search(
                    query=query,
                    max_results=max_results,
                    search_depth=settings.tavily_search_depth,
                    include_raw_content=settings.tavily_include_raw_content,
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
                # Tavily 库的异常映射：
                #   429 → UsageLimitExceededError（限速，应重试）
                #   432/403 → ForbiddenError（key 失效或月配额耗尽，应换 key）
                #   401 → InvalidAPIKeyError（key 无效，应换 key）
                from tavily.errors import (
                    ForbiddenError as _TForbidden,
                    InvalidAPIKeyError as _TInvalidKey,
                    UsageLimitExceededError as _TUsageLimit,
                )
                if isinstance(exc, _TUsageLimit):
                    raise TavilyRateLimitError(str(exc)) from exc
                if isinstance(exc, (_TForbidden, _TInvalidKey)):
                    raise TavilyQuotaError(str(exc), key_idx) from exc
                logger.error("Tavily 搜索失败: %s", exc)
                return []
            finally:
                # 每次请求后短暂等待，避免连续触发限速
                await asyncio.sleep(0.5)


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
        wait=wait_exponential(multiplier=2, min=2, max=8),
        retry=retry_if_exception_type(TavilyRateLimitError),
        reraise=True,
    )
    async def _primary_search(self, query: str, max_results: int) -> list[SearchResult]:
        """调用主引擎，限速时由 tenacity 按指数退避自动重试（最多 3 次）。"""
        return await self.provider.search(query, max_results=max_results)

    async def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        """执行单次搜索，限速时重试，额度耗尽时轮换 key，最终失败降级 DuckDuckGo。"""
        n_results = max_results or settings.max_search_results
        pool = _get_key_pool()

        results: list[SearchResult] = []
        # 最多尝试所有可用 key 次数
        max_key_attempts = len(pool._keys) + 1
        for _ in range(max_key_attempts):
            try:
                results = await self._primary_search(query, n_results)
                break
            except TavilyRateLimitError:
                logger.warning("Tavily 限速重试耗尽，降级 DDG: query='%s'", query[:50])
                break
            except TavilyQuotaError as exc:
                # 用 exc.key_idx（请求发出时的 index），而非 pool.current_idx，
                # 确保并发场景下多个任务失败只触发一次真正的轮换。
                new_key = pool.rotate(exc.key_idx)
                if new_key is None:
                    logger.warning("Tavily 所有 key 耗尽，降级 DDG: query='%s'", query[:50])
                    break
                # 继续循环，TavilyProvider.search() 下次调用会取 pool.current_key

        # 零结果降级：Tavily 返回空 → DuckDuckGo 补充
        if not results and self._current_provider == "tavily":
            logger.warning("Tavily 零结果，DDG 降级: query='%s'", query[:50])
            results = await self.providers["duckduckgo"].search(query, max_results=n_results)

        engine = self._current_provider if results else "duckduckgo(fallback)"
        logger.info("搜索完成 [%s]: query='%s', results=%d", engine, query[:50], len(results))
        return results

    async def probe_key_pool(self) -> None:
        """发 1 次轻量探针确认当前 Tavily key 可用；quota 耗尽则提前 rotate。

        在批量搜索前调用，消除 N 个并发请求同时撞到耗尽 key 时的级联串行重试。
        探针固定用 basic + 无 raw_content，不消耗 advanced credit。
        """
        if self._current_provider != "tavily":
            return
        pool = _get_key_pool()
        for _ in range(len(pool._keys) + 1):
            key_idx = pool.current_idx
            api_key = pool.current_key
            if not api_key:
                logger.warning("Tavily key probe: 所有 key 已耗尽")
                return
            try:
                from tavily import AsyncTavilyClient
                client = AsyncTavilyClient(api_key=api_key)
                await client.search(
                    "probe",
                    max_results=1,
                    search_depth="basic",
                    include_raw_content=False,
                    include_answer=False,
                )
                logger.info("Tavily key probe 通过: key %d/%d 可用", key_idx + 1, len(pool._keys))
                return
            except Exception as exc:
                try:
                    from tavily.errors import ForbiddenError as _F, InvalidAPIKeyError as _I
                    if isinstance(exc, (_F, _I)):
                        new_key = pool.rotate(key_idx)
                        if new_key is None:
                            return
                        continue
                except ImportError:
                    pass
                # 非 quota 错误（网络抖动/限速）—— 保留当前 key，让后续请求正常重试
                logger.warning("Tavily key probe 失败（非 quota 错误，保留当前 key）: %s", exc)
                return

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
