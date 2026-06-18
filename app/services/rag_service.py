"""RAG 服务 — 文本切片、Embedding 生成与检索。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.core.config import settings
from app.core.exceptions import VectorStoreError
from app.models.source import CrawledDocument, EvidenceChunk
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class TextChunker:
    """文本切片器。"""

    def __init__(self, chunk_size: int | None = None, overlap: int | None = None) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.overlap = overlap or settings.chunk_overlap

    def chunk_text(self, text: str, source_id: str, task_id: str,
                   url: str, title: str) -> list[EvidenceChunk]:
        """将文本分割为多个 Chunk。"""
        # 先按段落分割
        paragraphs = text.split("\n\n")
        chunks: list[EvidenceChunk] = []
        current_chunk = ""
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk = (current_chunk + "\n\n" + para).strip()
            else:
                # 保存当前 chunk
                if current_chunk:
                    chunks.append(EvidenceChunk(
                        task_id=task_id,
                        source_id=source_id,
                        url=url,
                        title=title,
                        chunk_index=chunk_index,
                        text=current_chunk,
                    ))
                    chunk_index += 1

                # 保留 overlap 部分
                words = current_chunk.split()
                overlap_text = ""
                if words and self.overlap > 0:
                    overlap_words = words[-min(len(words), self.overlap // 5):]
                    overlap_text = " ".join(overlap_words)

                current_chunk = (overlap_text + "\n\n" + para).strip()

        # 最后一个 chunk
        if current_chunk:
            chunks.append(EvidenceChunk(
                task_id=task_id,
                source_id=source_id,
                url=url,
                title=title,
                chunk_index=chunk_index,
                text=current_chunk,
            ))

        return chunks

    def chunk_document(self, doc: CrawledDocument, task_id: str,
                       source_id: str) -> list[EvidenceChunk]:
        """对已抓取文档进行切片。"""
        if not doc.content:
            return []
        return self.chunk_text(doc.content, source_id, task_id, doc.url, doc.title)


class RAGService:
    """RAG 服务 — 整合切片、Embedding 与检索。"""

    def __init__(self, vector_store: VectorStoreService | None = None) -> None:
        self.chunker = TextChunker()
        self.vector_store = vector_store or VectorStoreService()
        self._embedding_model = None

    async def _get_embedding_model(self):
        """延迟加载 Embedding 模型。"""
        if self._embedding_model is None:
            try:
                from fastembed import TextEmbedding
                self._embedding_model = TextEmbedding(
                    model_name=settings.embedding_model,
                    max_length=512,
                )
                logger.info("Embedding 模型已加载: %s", settings.embedding_model)
            except ImportError:
                logger.error("fastembed 未安装，请执行: pip install fastembed")
                raise VectorStoreError("fastembed 未安装")
            except Exception as exc:
                logger.error("Embedding 模型加载失败: %s", exc)
                raise VectorStoreError(f"Embedding 模型加载失败: {exc}") from exc

        return self._embedding_model

    async def build_evidence(
        self,
        documents: list[CrawledDocument],
        task_id: str,
    ) -> list[EvidenceChunk]:
        """将抓取的文档切片 + 向量化 + 入库。"""
        all_chunks: list[EvidenceChunk] = []

        for i, doc in enumerate(documents):
            if not doc.content or doc.error:
                continue
            source_id = f"src_{task_id}_{i:04d}"
            chunks = self.chunker.chunk_document(doc, task_id, source_id)
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("没有有效的 Chunk 可入库")
            return []

        # 生成向量并存储
        try:
            model = await self._get_embedding_model()
            texts = [c.text for c in all_chunks]

            # fastembed.embed() 是同步生成器，在默认线程池中运行
            def _gen_embeddings() -> list[list[float]]:
                return [emb.tolist() for emb in model.embed(texts)]

            embeddings = await asyncio.to_thread(_gen_embeddings)

            # 存储到 Qdrant
            vector_ids = await self.vector_store.store_chunks(all_chunks, embeddings)

            # 回写 vector_id
            for chunk, vid in zip(all_chunks, vector_ids):
                chunk.vector_id = vid

            logger.info("证据构建完成: %d chunks, %d docs", len(all_chunks), len(documents))
            return all_chunks

        except Exception as exc:
            logger.error("证据构建失败: %s", exc)
            raise VectorStoreError(f"证据构建失败: {exc}") from exc

    async def retrieve_evidence(
        self,
        query: str,
        task_id: str | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """检索与查询相关的证据。"""
        try:
            model = await self._get_embedding_model()
            # 生成查询向量（同步生成器，在默认线程池运行）
            def _gen_query_emb() -> list[float]:
                for emb in model.embed([query]):
                    return emb.tolist()
                return []

            query_vector = await asyncio.to_thread(_gen_query_emb)
            if not query_vector:
                return []

            return await self.vector_store.search(
                query_vector=query_vector,
                task_id=task_id,
                top_k=top_k,
            )

        except Exception as exc:
            logger.error("证据检索失败: %s", exc)
            return []
