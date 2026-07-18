"""分布式限流 — Redis 令牌桶。

进程内 asyncio.Semaphore 只在单进程内有效；多 worker 下 N 个进程各自持有一份，
真实并发 = N × 本地上限，会打爆下游（LLM / Tavily）。本模块用 Redis 令牌桶做
「跨进程全局速率上限」，令牌桶算法用 Lua 脚本保证「读-refill-扣减」原子执行。

降级安全：Redis 不可用 / 未配置 → acquire() 直接放行，由调用方的本地信号量兜底，
绝不因限流层故障阻断流程。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.core.config import settings
from app.services.cache_service import get_cache

logger = logging.getLogger(__name__)

# 令牌桶 Lua：按距上次的时间差补充令牌，够则扣 1 放行，不够则返回需等待秒数。
# 原子执行避免多 worker 并发时的读改写竞争。返回 {allowed(0/1), wait_seconds(str)}。
_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])
if tokens == nil then
  tokens = capacity
  ts = now
end

local delta = math.max(0, now - ts) / 1000.0
tokens = math.min(capacity, tokens + delta * rate)

local allowed = 0
local wait = 0.0
if tokens >= requested then
  tokens = tokens - requested
  allowed = 1
else
  wait = (requested - tokens) / rate
end

redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
redis.call('PEXPIRE', key, math.ceil((capacity / rate) * 1000) + 2000)
return {allowed, tostring(wait)}
"""


class RedisRateLimiter:
    """基于 Redis 令牌桶的分布式限流器（降级安全）。"""

    def __init__(self) -> None:
        self._cache = get_cache()
        self._script: Any | None = None
        client = getattr(self._cache, "_client", None)
        if self._cache.enabled and client is not None:
            try:
                self._script = client.register_script(_TOKEN_BUCKET_LUA)
            except Exception as exc:
                logger.warning("限流脚本注册失败，退回本地信号量: %s", exc)

    @property
    def enabled(self) -> bool:
        return self._script is not None and self._cache.enabled

    async def acquire(
        self,
        bucket: str,
        rate: float,
        burst: int,
        max_wait: float = 30.0,
    ) -> bool:
        """获取一个令牌；不足则按需 sleep 直到可用或累计等待超 max_wait。

        返回 True 表示确实等待/限流过，False 表示立即放行或已降级。
        Redis 不可用时直接放行（返回 False），由调用方本地信号量兜底。
        """
        if not self.enabled:
            return False

        key = f"dr:{settings.cache_version}:rl:{bucket}"
        waited = 0.0
        while True:
            try:
                now = int(time.time() * 1000)
                res = await self._script(keys=[key], args=[rate, burst, now, 1])
            except Exception as exc:
                logger.debug("限流调用失败，放行: %s", exc)
                return False

            allowed = int(res[0]) == 1
            wait = float(res[1])
            if allowed:
                return waited > 0

            sleep_for = min(max(wait, 0.005), 0.5)
            if waited + sleep_for > max_wait:
                logger.warning("限流累计等待超 %.1fs，放行 bucket=%s", max_wait, bucket)
                return False
            await asyncio.sleep(sleep_for)
            waited += sleep_for


_limiter_singleton: RedisRateLimiter | None = None


def get_rate_limiter() -> RedisRateLimiter:
    """返回进程级单例 RedisRateLimiter。"""
    global _limiter_singleton
    if _limiter_singleton is None:
        _limiter_singleton = RedisRateLimiter()
    return _limiter_singleton
