"""缓存服务 — 基于 Redis，降级安全。

设计原则：缓存是加速旁路，绝不能因其故障中断研究流程。
- REDIS_URL 为空或 CACHE_ENABLED=false → 整体禁用，所有方法变为 no-op / 返回未命中。
- 任何 Redis 操作异常 → 记 debug 日志并当作「未命中 / 写入忽略」，向上层透明。
- 进程级单例（get_cache()），跨 worker 天然共享同一份缓存。

key 规范：dr:{cache_version}:{kind}:{sha1(参数)}。改 CACHE_VERSION 可整体失效旧缓存。
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from app.core.config import mask_dsn, settings

logger = logging.getLogger(__name__)


class CacheService:
    """Redis 缓存封装，全部操作降级安全。"""

    def __init__(self) -> None:
        self._enabled = bool(settings.cache_enabled and settings.redis_url)
        self._client: Any | None = None
        self._stats: dict[str, int] = {"hit": 0, "miss": 0, "error": 0}
        # _enabled 是「按配置该不该用缓存」，构造后不再变；_degraded 是「当前连不上」，
        # 由 ping() 双向切换。两者分开是因为降级必须可恢复：容器化部署下 Redis 会独立
        # 重启（restart: unless-stopped）并换 IP，若降级是单向的，长驻的 api/worker
        # 就会永久放弃缓存直到人工重启——redis-py 本身下一条命令就会自动重连。
        self._degraded = False

        if not self._enabled:
            logger.info("缓存未启用（REDIS_URL 为空或 CACHE_ENABLED=false）")
            return

        try:
            import redis.asyncio as aioredis

            # socket 超时设小值：Redis 抖动时快速失败并旁路，不拖慢主流程
            self._client = aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            logger.info("缓存已启用: %s", mask_dsn(settings.redis_url))
        except ImportError:
            logger.warning("redis 未安装，缓存禁用（pip install redis）")
            self._enabled = False
        except Exception as exc:
            logger.warning("Redis 初始化失败，缓存禁用: %s", exc)
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled and self._client is not None and not self._degraded

    def key(self, kind: str, *parts: Any) -> str:
        """生成命名空间化的缓存 key。"""
        raw = "|".join(str(p) for p in parts)
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        return f"dr:{settings.cache_version}:{kind}:{digest}"

    async def ping(self) -> bool:
        """连通性探测，双向切换降级状态。

        失败时降级为旁路，避免后续每次操作都抛错；恢复时自动解除降级——
        /health 会周期性调到这里，所以 Redis 重启后无需人工干预即可自愈。
        """
        if not self._enabled or self._client is None:
            return False
        try:
            ok = bool(await self._client.ping())
        except Exception as exc:
            if not self._degraded:
                logger.warning("Redis ping 失败，缓存降级为旁路: %s", exc)
            self._degraded = True
            return False

        if ok and self._degraded:
            logger.info("Redis 已恢复，缓存重新启用: %s", mask_dsn(settings.redis_url))
            self._degraded = False
        return ok

    # ── 单值读写 ────────────────────────────────────────────────────────────────

    async def get_json(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        try:
            raw = await self._client.get(key)
        except Exception as exc:
            self._stats["error"] += 1
            logger.debug("缓存读取失败（旁路）: %s", exc)
            return None
        if raw is None:
            self._stats["miss"] += 1
            return None
        try:
            value = json.loads(raw)
            self._stats["hit"] += 1
            return value
        except Exception:
            self._stats["miss"] += 1
            return None

    async def set_json(self, key: str, value: Any, ttl: int) -> None:
        if not self.enabled:
            return
        try:
            await self._client.set(key, json.dumps(value, ensure_ascii=False), ex=ttl)
        except Exception as exc:
            logger.debug("缓存写入失败（忽略）: %s", exc)

    # ── 批量读写（embedding 用）─────────────────────────────────────────────────

    async def mget_json(self, keys: list[str]) -> list[Any | None]:
        """批量读取，返回与 keys 等长的列表，未命中/损坏项为 None。"""
        if not self.enabled or not keys:
            return [None] * len(keys)
        try:
            raws = await self._client.mget(keys)
        except Exception as exc:
            self._stats["error"] += 1
            logger.debug("缓存批量读取失败（旁路）: %s", exc)
            return [None] * len(keys)

        out: list[Any | None] = []
        for raw in raws:
            if raw is None:
                self._stats["miss"] += 1
                out.append(None)
                continue
            try:
                out.append(json.loads(raw))
                self._stats["hit"] += 1
            except Exception:
                self._stats["miss"] += 1
                out.append(None)
        return out

    async def mset_json(self, mapping: dict[str, Any], ttl: int) -> None:
        """批量写入（带 per-key TTL，用 pipeline，MSET 不支持 TTL）。"""
        if not self.enabled or not mapping:
            return
        try:
            pipe = self._client.pipeline()
            for k, v in mapping.items():
                pipe.set(k, json.dumps(v, ensure_ascii=False), ex=ttl)
            await pipe.execute()
        except Exception as exc:
            logger.debug("缓存批量写入失败（忽略）: %s", exc)

    # ── 观测与生命周期 ──────────────────────────────────────────────────────────

    def log_stats(self) -> None:
        h, m, e = self._stats["hit"], self._stats["miss"], self._stats["error"]
        total = h + m
        rate = (h / total * 100) if total else 0.0
        logger.info(
            "缓存统计: 命中 %d / 未命中 %d（命中率 %.1f%%），错误 %d",
            h, m, rate, e,
        )

    async def aclose(self) -> None:
        if self._client is None:
            return
        try:
            close = getattr(self._client, "aclose", None) or getattr(self._client, "close", None)
            if close:
                await close()
        except Exception:
            pass


_cache_singleton: CacheService | None = None


def get_cache() -> CacheService:
    """返回进程级单例 CacheService。"""
    global _cache_singleton
    if _cache_singleton is None:
        _cache_singleton = CacheService()
    return _cache_singleton
