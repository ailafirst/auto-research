"""向量数据库服务 — 基于 Qdrant。"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from app.core.config import settings
from app.core.exceptions import VectorStoreError
from app.models.source import EvidenceChunk

logger = logging.getLogger(__name__)

# 单次 upsert 的 point 数上限。256 × 1024 维向量 + chunk 全文 payload 约 3-5MB，
# 远低于 Qdrant 默认的请求体上限，同时批次数量不至于多到让往返开销变得显著。
_UPSERT_BATCH_SIZE = 256


class VectorStoreService:
    """向量数据库服务，管理文档切片与检索。"""

    def __init__(self) -> None:
        self.collection_name = settings.qdrant_collection
        self.embedding_dim = settings.embedding_dim
        self._collection_initialized = False

        # 构造期不做 I/O：远程可达性在首次 _ensure_collection 时才知道，不可达则就地
        # 回退内存模式。qdrant_mode 显式选择而非「配了 url 就用远程」，避免历史部署
        # （.env 里一直留着 QDRANT_URL 却没起服务）在升级后平白多一次连接超时。
        if settings.qdrant_mode == "remote":
            self._is_in_memory = False
            self.client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )
            logger.info("Qdrant 使用远程模式: %s", settings.qdrant_url)
        else:
            self._is_in_memory = True
            self.client = AsyncQdrantClient(location=":memory:")
            logger.info("Qdrant 使用内存模式")

    @property
    def mode(self) -> str:
        """实际生效的模式（远程回退后会变成 memory），供 /health 如实上报。"""
        return "memory" if self._is_in_memory else "remote"

    def _fallback_to_memory(self, reason: Exception) -> None:
        """远程不可达时就地降级为内存实例，本次进程不再重试远程。"""
        logger.warning(
            "Qdrant 远程不可用（%s），回退内存模式；本次运行的向量不会持久化", reason
        )
        self.client = AsyncQdrantClient(location=":memory:")
        self._is_in_memory = True
        self._collection_initialized = False

    async def _ensure_collection(self) -> None:
        """确保集合存在。"""
        if self._collection_initialized:
            return

        if not self._is_in_memory:
            try:
                await self.client.get_collections()
            except Exception as exc:
                self._fallback_to_memory(exc)

        try:
            collections = await self.client.get_collections()
            exists = any(
                c.name == self.collection_name
                for c in collections.collections
            )

            if not exists:
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dim,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info("向量集合已创建: %s", self.collection_name)
            else:
                logger.info("向量集合已存在: %s", self.collection_name)

            await self._ensure_payload_indexes()
            self._collection_initialized = True

        except Exception as exc:
            logger.error("Qdrant 初始化失败: %s", exc)
            raise VectorStoreError(f"Qdrant 初始化失败: {exc}") from exc

    async def _ensure_payload_indexes(self) -> None:
        """建 payload 索引（幂等，已存在时 Qdrant 直接返回成功）。

        两个字段都必须索引，否则退化成全集合扫描：
          task_id     —— 每次 search 都带这个过滤条件（见 search()）。内存模式下
                         集合随进程新建、数据量小，没索引也无感；remote 模式集合
                         长期累积，不索引会随任务数线性变慢。
          created_at  —— 过期清理按它做范围删除。Qdrant 维护者在讨论 #5441 里明确
                         强调「务必给 timestamp 字段建 payload 索引以保证删除速度」。
        """
        for field, schema in (
            ("task_id", models.PayloadSchemaType.KEYWORD),
            ("created_at", models.PayloadSchemaType.INTEGER),
        ):
            try:
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception as exc:
                # 索引建不上不该阻断写入——检索和删除只是变慢，不是不可用
                logger.warning("payload 索引 %s 创建失败（将退化为全扫描）: %s", field, exc)

    async def store_chunks(self, chunks: list[EvidenceChunk],
                           embeddings: list[list[float]]) -> list[str]:
        """存储切片及其向量到 Qdrant。"""
        await self._ensure_collection()

        if len(chunks) != len(embeddings):
            raise VectorStoreError("chunks 与 embeddings 数量不匹配")

        vector_ids: list[str] = []
        points: list[models.PointStruct] = []

        # 写入时刻，供过期清理做范围删除。用 Unix 秒（整数）而非 ISO 字符串：
        # 整数 payload 索引的范围查询最直接，也不受时区/格式解析影响。
        now_ts = int(time.time())

        for chunk, vector in zip(chunks, embeddings):
            point_id = str(uuid.uuid4())
            vector_ids.append(point_id)

            points.append(models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "task_id": chunk.task_id,
                    "source_id": chunk.source_id,
                    "url": chunk.url,
                    "title": chunk.title,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "created_at": now_ts,
                },
            ))

        # 分批写入。内存模式下不经 HTTP，一次塞多少都无所谓；远程模式则是一个
        # POST 请求，实测一个任务能产出 2000+ chunk，1024 维向量加 chunk 全文
        # payload 序列化后有几十 MB，超过 Qdrant 的请求体上限直接 400 Bad Request
        # ——而调用方把它当成「Qdrant 不可用」静默跳过整个证据构建，任务照常完成，
        # 只是报告没有任何检索证据支撑。
        try:
            for start in range(0, len(points), _UPSERT_BATCH_SIZE):
                await self.client.upsert(
                    collection_name=self.collection_name,
                    points=points[start:start + _UPSERT_BATCH_SIZE],
                )
            logger.info(
                "已存储 %d 个 Chunk 到 Qdrant（%d 批）",
                len(points),
                (len(points) + _UPSERT_BATCH_SIZE - 1) // _UPSERT_BATCH_SIZE,
            )
            return vector_ids

        except Exception as exc:
            logger.error("Qdrant 存储失败: %s", exc)
            raise VectorStoreError(f"向量存储失败: {exc}") from exc

    async def search(
        self,
        query_vector: list[float],
        task_id: str | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """检索最相关的 Chunk。"""
        await self._ensure_collection()

        k = top_k or settings.rag_top_k

        query_filter = None
        if task_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="task_id",
                        match=models.MatchValue(value=task_id),
                    )
                ],
            )

        try:
            # qdrant-client >= 1.7 removed search(), use query_points() instead
            response = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=k,
                query_filter=query_filter,
                with_payload=True,
            )

            return [
                {
                    "score": r.score,
                    "text": r.payload.get("text", ""),
                    "url": r.payload.get("url", ""),
                    "title": r.payload.get("title", ""),
                    "chunk_index": r.payload.get("chunk_index", 0),
                    "source_id": r.payload.get("source_id", ""),
                }
                for r in response.points
            ]

        except Exception as exc:
            logger.error("Qdrant 检索失败: %s", exc)
            raise VectorStoreError(f"向量检索失败: {exc}") from exc

    async def delete_task_chunks(self, task_id: str) -> int:
        """删除某任务的所有 Chunk。"""
        await self._ensure_collection()

        try:
            result = await self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="task_id",
                                match=models.MatchValue(value=task_id),
                            )
                        ],
                    ),
                ),
            )
            logger.info("已删除任务 %s 的 Chunk", task_id)
            return 0  # Qdrant 不直接返回删除数量

        except Exception as exc:
            logger.error("Qdrant 删除失败: %s", exc)
            raise VectorStoreError(f"向量删除失败: {exc}") from exc

    async def delete_expired(self, ttl_days: int) -> int:
        """删除超过 ttl_days 的证据向量，返回删除前的点数与删除后的差值。

        采用 Qdrant 官方三种滑动时间窗策略里的 filter-and-delete。官方对它的告警是
        「不适合每天百万级删除的时序场景」——本项目单任务约 2000 chunk、日增最多几
        个任务，比那个量级小三四个数量级，用 shard/collection rotation 属于过度设计。

        删除产生的墓碑点由 Qdrant 的 vacuum optimizer 异步回收（默认阈值 20%，即
        删除比例超过 20% 才触发段重整），所以磁盘空间不会立刻下降，属预期行为。
        """
        if ttl_days <= 0:
            return 0
        if self._is_in_memory:
            # 内存实例随进程消亡，没有累积问题，清理无意义
            return 0

        await self._ensure_collection()
        cutoff = int(time.time()) - ttl_days * 86400

        try:
            before = (await self.client.count(self.collection_name, exact=True)).count
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="created_at",
                                range=models.Range(lt=cutoff),
                            )
                        ],
                    ),
                ),
            )
            after = (await self.client.count(self.collection_name, exact=True)).count
            removed = max(before - after, 0)
            logger.info(
                "向量过期清理完成: 删除 %d 个点（保留 %d 个，TTL=%d 天）",
                removed, after, ttl_days,
            )
            return removed
        except Exception as exc:
            # 清理失败不该影响正常服务，下次 cron 会重试
            logger.error("向量过期清理失败: %s", exc)
            return 0

    async def health_check(self) -> bool:
        """检查 Qdrant 是否可用。"""
        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """关闭连接。"""
        await self.client.close()
