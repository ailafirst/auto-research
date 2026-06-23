"""向量数据库服务 — 基于 Qdrant。"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from app.core.config import settings
from app.core.exceptions import VectorStoreError
from app.models.source import EvidenceChunk

logger = logging.getLogger(__name__)


class VectorStoreService:
    """向量数据库服务，管理文档切片与检索。"""

    def __init__(self) -> None:
        self.collection_name = settings.qdrant_collection
        self.embedding_dim = settings.embedding_dim
        self._collection_initialized = False
        self._is_in_memory = True
        self.client = AsyncQdrantClient(location=":memory:")
        logger.info("Qdrant 使用内存模式")

    async def _ensure_collection(self) -> None:
        """确保集合存在。"""
        if self._collection_initialized:
            return

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

            self._collection_initialized = True

        except Exception as exc:
            logger.error("Qdrant 初始化失败: %s", exc)
            raise VectorStoreError(f"Qdrant 初始化失败: {exc}") from exc

    async def store_chunks(self, chunks: list[EvidenceChunk],
                           embeddings: list[list[float]]) -> list[str]:
        """存储切片及其向量到 Qdrant。"""
        await self._ensure_collection()

        if len(chunks) != len(embeddings):
            raise VectorStoreError("chunks 与 embeddings 数量不匹配")

        vector_ids: list[str] = []
        points: list[models.PointStruct] = []

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
                },
            ))

        try:
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            logger.info("已存储 %d 个 Chunk 到 Qdrant", len(points))
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
