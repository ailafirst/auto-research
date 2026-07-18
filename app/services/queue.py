"""任务队列 — arq (Redis broker)。降级安全：不可用时回退进程内执行。"""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

_pool: Any | None = None
_tried = False


async def get_arq_pool() -> Any | None:
    """返回 arq 连接池单例；不可用返回 None（调用方回退 asyncio.create_task）。"""
    global _pool, _tried
    if _pool is not None:
        return _pool
    if _tried:
        return None
    _tried = True

    if not settings.redis_url:
        logger.info("未配置 REDIS_URL，任务队列禁用，回退进程内执行")
        return None
    try:
        from arq import create_pool
        from arq.connections import RedisSettings
        _pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
        logger.info("arq 任务队列已连接: %s", settings.redis_url)
    except Exception as exc:
        logger.warning("arq 队列不可用，回退进程内执行: %s", exc)
        _pool = None
    return _pool


async def close_arq_pool() -> None:
    global _pool, _tried
    if _pool is not None:
        try:
            close = getattr(_pool, "aclose", None) or getattr(_pool, "close", None)
            if close:
                await close()
        except Exception:
            pass
        _pool = None
    _tried = False
