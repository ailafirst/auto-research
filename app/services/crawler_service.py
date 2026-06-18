"""网页抓取与内容清洗服务。"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any
from urllib.parse import urlparse

import certifi
import httpx
from bs4 import BeautifulSoup, Tag

from app.core.config import settings
from app.models.source import CrawledDocument

logger = logging.getLogger(__name__)

# 黑名单域名
BLACKLISTED_DOMAINS: set[str] = {
    "facebook.com", "twitter.com", "x.com", "instagram.com",
    "youtube.com", "tiktok.com",
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]


def _is_blacklisted(url: str) -> bool:
    domain = urlparse(url).netloc.lower()
    return any(b in domain for b in BLACKLISTED_DOMAINS)


def _extract_text(html: str) -> str:
    """从 HTML 提取可读正文。"""
    soup = BeautifulSoup(html, "lxml")

    # 移除无用标签
    for tag in soup(["script", "style", "nav", "footer", "header",
                      "aside", "noscript", "iframe", "form", "svg",
                      "button", "input"]):
        tag.decompose()

    # 定位主要内容区域
    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find(class_=re.compile(r"(content|article|post|entry|main)", re.I))
        or soup.find(id=re.compile(r"(content|article|post|entry|main)", re.I))
        or soup.body
    )

    if main is None:
        return ""

    # 提取文本
    texts: list[str] = []
    for el in main.find_all(
        ["p", "h1", "h2", "h3", "h4", "h5", "h6",
         "li", "blockquote", "td", "th", "pre", "code"],
        recursive=True,
    ):
        if isinstance(el, Tag):
            text = el.get_text(strip=True)
            if text and len(text) > 15:
                texts.append(text)

    return "\n\n".join(texts)


def _extract_title(html: str, url: str) -> str:
    """提取页面标题。"""
    soup = BeautifulSoup(html, "lxml")

    # Open Graph
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return str(og_title["content"])

    # <title>
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        title = title_tag.string.strip()
        for sep in [" | ", " — ", " – ", " - ", " :: "]:
            if sep in title:
                return title.split(sep)[0].strip()
        return title

    # <h1>
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)

    # 从 URL 推断
    path = urlparse(url).path.strip("/").replace("/", " — ").replace("-", " ")
    return path if path else urlparse(url).netloc


class CrawlerService:
    """网页抓取服务。"""

    def __init__(self) -> None:
        self.timeout = settings.request_timeout
        self.max_size = 5 * 1024 * 1024  # 5MB

    async def fetch(self, url: str, client: httpx.AsyncClient | None = None) -> CrawledDocument:
        """抓取单个网页。"""
        import random
        import time

        start_time = time.monotonic()

        if _is_blacklisted(url):
            return CrawledDocument(
                url=url, title="", content="",
                error="Domain is blacklisted",
                fetch_time=time.monotonic() - start_time,
            )

        close_client = client is None
        if client is None:
            client = httpx.AsyncClient(
                follow_redirects=True,
                timeout=self.timeout,
                headers={"User-Agent": random.choice(USER_AGENTS)},
                verify=certifi.where(),
            )

        try:
            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return CrawledDocument(
                    url=url, title="", content="",
                    error=f"Not HTML: {content_type}",
                    fetch_time=time.monotonic() - start_time,
                )

            html = response.text
            if len(html.encode("utf-8")) > self.max_size:
                html = html[:self.max_size]

            title = _extract_title(html, url)
            content = _extract_text(html)

            if not content.strip():
                return CrawledDocument(
                    url=url, title=title, content="",
                    error="No extractable content",
                    fetch_time=time.monotonic() - start_time,
                )

            logger.info("抓取成功: %s, 标题=%s, 长度=%d", url, title, len(content))
            return CrawledDocument(
                url=url, title=title, content=content,
                text_length=len(content),
                fetch_time=time.monotonic() - start_time,
            )

        except httpx.TimeoutException:
            return CrawledDocument(
                url=url, title="", content="",
                error="Timeout", fetch_time=time.monotonic() - start_time,
            )
        except httpx.HTTPStatusError as exc:
            return CrawledDocument(
                url=url, title="", content="",
                error=f"HTTP {exc.response.status_code}",
                fetch_time=time.monotonic() - start_time,
            )
        except Exception as exc:
            return CrawledDocument(
                url=url, title="", content="",
                error=str(exc), fetch_time=time.monotonic() - start_time,
            )
        finally:
            if close_client:
                await client.aclose()

    async def batch_fetch(
        self, urls: list[str], max_concurrent: int | None = None,
    ) -> list[CrawledDocument]:
        """批量抓取。"""
        sem = asyncio.Semaphore(max_concurrent or settings.max_concurrent_fetches)

        async def _fetch(url: str) -> CrawledDocument:
            async with sem:
                async with httpx.AsyncClient(
                    follow_redirects=True, timeout=self.timeout,
                    verify=certifi.where(),
                ) as client:
                    return await self.fetch(url, client=client)

        tasks = [_fetch(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        docs: list[CrawledDocument] = []
        for r in results:
            if isinstance(r, CrawledDocument):
                docs.append(r)
            elif isinstance(r, Exception):
                logger.warning("抓取异常: %s", r)

        return docs
