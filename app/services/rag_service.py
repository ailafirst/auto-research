"""RAG 服务 — 文本切片、Embedding 生成与检索。"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from app.core.config import settings
from app.core.exceptions import VectorStoreError
from app.models.source import CrawledDocument, EvidenceChunk
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class TextChunker:
    """文本切片器。"""

    # 句子边界：中英文句末标点 + 换行。用于切分超长"段落"——raw_content 提取常丢失
    # 段落边界，使 split("\n\n") 产出多段黏连的巨块，需在句子处二次切分。
    _SENT_BOUNDARY = re.compile(r"(?<=[。！？!?；;\n])")

    def __init__(self, chunk_size: int | None = None, overlap: int | None = None) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.overlap = overlap or settings.chunk_overlap

    def _split_oversized(self, para: str) -> list[str]:
        """将超过 chunk_size 的段落按句子边界切分为 ≤chunk_size 的片段。

        目的：让每个 chunk 都落在 embedding 的 512-token 窗口内，使向量真正代表
        全文（而非前 1/3 的模糊平均）。切分只在句末标点/换行处发生，绝不切断句子；
        仅当单句本身超长（无标点长串，罕见）时才硬切兜底。正常段落原样返回。
        """
        if len(para) <= self.chunk_size:
            return [para]
        sentences = [s for s in self._SENT_BOUNDARY.split(para) if s.strip()]
        pieces: list[str] = []
        buf = ""
        for sent in sentences:
            if len(buf) + len(sent) <= self.chunk_size:
                buf += sent
            else:
                if buf:
                    pieces.append(buf.strip())
                if len(sent) > self.chunk_size:
                    # 单句超长：按 chunk_size 硬切（兜底，正常文本不会触发）
                    for i in range(0, len(sent), self.chunk_size):
                        pieces.append(sent[i:i + self.chunk_size].strip())
                    buf = ""
                else:
                    buf = sent
        if buf.strip():
            pieces.append(buf.strip())
        return pieces

    def chunk_text(self, text: str, source_id: str, task_id: str,
                   url: str, title: str) -> list[EvidenceChunk]:
        """将文本分割为多个 Chunk。"""
        # 先按段落分割，再将超长段落（提取丢失边界的巨块）按句子边界二次切分
        paragraphs = [
            sub
            for para in text.split("\n\n")
            for sub in self._split_oversized(para.strip())
            if sub.strip()
        ]
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

                # 保留 overlap 部分（按字符截取，兼容中文——中文无空格，旧的
                # current_chunk.split() 会把整段视为一个词，使 overlap 退化为整个
                # 上一 chunk，导致每个中文 chunk 被撑大近一倍）
                overlap_text = current_chunk[-self.overlap:] if self.overlap > 0 else ""

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
        self._model_lock = asyncio.Lock()
        # SentenceTransformer/PyTorch 实例非线程安全：并发 encode 会在共享模型上
        # 死锁。串行化所有 encode（单次很快，300 chunk≈2.6s，串行代价可忽略）。
        self._encode_lock = asyncio.Lock()

    # ── Embedding ──────────────────────────────────────────────────────────────

    async def _get_fastembed_model(self):
        """延迟加载 fastembed 本地 Embedding 模型（单次加载后缓存在实例上）。"""
        if self._embedding_model is None:
            async with self._model_lock:
                if self._embedding_model is None:
                    try:
                        from fastembed import TextEmbedding
                        self._embedding_model = TextEmbedding(
                            model_name=settings.embedding_model,
                            max_length=512,
                        )
                        logger.info("fastembed 模型已加载: %s", settings.embedding_model)
                    except ImportError:
                        logger.error("fastembed 未安装，请执行: pip install fastembed")
                        raise VectorStoreError("fastembed 未安装")
                    except Exception as exc:
                        logger.error("fastembed 模型加载失败: %s", exc)
                        raise VectorStoreError(f"fastembed 模型加载失败: {exc}") from exc
        return self._embedding_model

    async def _get_st_model(self):
        """延迟加载 sentence-transformers 模型（bge-m3 等大型本地模型）。
        用 Lock 防止并发子问题同时触发多次模型加载（double-checked locking）。
        """
        if self._embedding_model is None:
            async with self._model_lock:
                if self._embedding_model is None:
                    try:
                        import torch
                        from sentence_transformers import SentenceTransformer
                        device = "cuda" if torch.cuda.is_available() else "cpu"

                        def _load():
                            m = SentenceTransformer(
                                settings.embedding_model,
                                trust_remote_code=True,
                                device=device,
                            )
                            # 限制序列长度：检索 chunk 约 800 字（≈512 token），默认 8192
                            # 会让长段落 chunk 的 encode 显存/耗时爆炸（实测 400chunk
                            # 84.6s/10.3GB → 6.5s/2.9GB）。与 rerank 的 512 截断一致。
                            m.max_seq_length = 512
                            return m

                        self._embedding_model = await asyncio.to_thread(_load)
                        logger.info(
                            "sentence-transformers 模型已加载: %s (device=%s)",
                            settings.embedding_model, device,
                        )
                    except ImportError:
                        raise VectorStoreError("sentence-transformers 未安装，请执行: pip install sentence-transformers")
                    except Exception as exc:
                        raise VectorStoreError(f"sentence-transformers 模型加载失败: {exc}") from exc
        return self._embedding_model

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """统一 Embedding 入口，根据 embedding_provider 路由。"""
        if settings.embedding_provider == "api":
            return await self._embed_via_api(texts)
        if settings.embedding_provider == "st":
            return await self._embed_via_st(texts)
        return await self._embed_via_fastembed(texts)

    async def _embed_via_st(self, texts: list[str]) -> list[list[float]]:
        """sentence-transformers 本地推理（支持 GPU）。"""
        model = await self._get_st_model()

        def _encode() -> list[list[float]]:
            import torch
            # 大批量前释放 CUDA 缓存，防止 reserved-but-unallocated 内存导致 OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch_size = 32  # 实测 bs=32 已达 encode 吞吐饱和，峰值<5GB；再大无增益
            vecs = model.encode(texts, normalize_embeddings=True, batch_size=batch_size)
            return vecs.tolist()

        # 串行化：防止 analyst 多子问题并发检索时同时 encode 同一模型而死锁
        async with self._encode_lock:
            return await asyncio.to_thread(_encode)

    async def _embed_via_fastembed(self, texts: list[str]) -> list[list[float]]:
        model = await self._get_fastembed_model()

        def _gen() -> list[list[float]]:
            return [emb.tolist() for emb in model.embed(texts)]

        return await asyncio.to_thread(_gen)

    async def _embed_via_api(self, texts: list[str]) -> list[list[float]]:
        """通过 LiteLLM 调用 API embedding（text-embedding-3-small 等）。"""
        try:
            from litellm import aembedding
        except ImportError:
            raise VectorStoreError("litellm 未安装，无法使用 api embedding")

        kwargs: dict[str, Any] = {
            "model": settings.embedding_model,
            "input": texts,
            "api_key": settings.llm_api_key,
        }
        if settings.llm_base_url:
            kwargs["api_base"] = settings.llm_base_url

        try:
            response = await aembedding(**kwargs)
            return [item["embedding"] for item in response.data]
        except Exception as exc:
            logger.error("API embedding 调用失败: %s", exc)
            raise VectorStoreError(f"API embedding 失败: {exc}") from exc

    # ── 证据构建 ───────────────────────────────────────────────────────────────

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

        try:
            # A4: 拼接标题为上下文，帮助 embedding 理解 chunk 所属主题
            texts = [
                f"{c.title}\n\n{c.text}" if c.title else c.text
                for c in all_chunks
            ]
            embeddings = await self._embed(texts)

            # 过滤 NaN 向量（空文本或零范数文本会产生 NaN，Qdrant 写入时会报错）
            import math
            valid = [(c, e) for c, e in zip(all_chunks, embeddings)
                     if e and not any(math.isnan(v) for v in e)]
            if len(valid) < len(all_chunks):
                logger.warning("过滤 %d 个 NaN 向量（共 %d chunks）",
                               len(all_chunks) - len(valid), len(all_chunks))
            if not valid:
                logger.warning("所有向量均为 NaN，跳过入库")
                return all_chunks
            valid_chunks, valid_embeddings = zip(*valid)

            vector_ids = await self.vector_store.store_chunks(
                list(valid_chunks), list(valid_embeddings)
            )
            for chunk, vid in zip(valid_chunks, vector_ids):
                chunk.vector_id = vid

            logger.info(
                "证据构建完成: %d chunks, %d docs (provider=%s)",
                len(all_chunks), len(documents), settings.embedding_provider,
            )
            return all_chunks

        except Exception as exc:
            logger.error("证据构建失败: %s", exc)
            raise VectorStoreError(f"证据构建失败: {exc}") from exc

    # ── 证据检索 ───────────────────────────────────────────────────────────────

    async def retrieve_evidence(
        self,
        query: str,
        task_id: str | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """检索与查询相关的证据。"""
        try:
            vectors = await self._embed([query])
            if not vectors or not vectors[0]:
                return []

            return await self.vector_store.search(
                query_vector=vectors[0],
                task_id=task_id,
                top_k=top_k,
            )

        except Exception as exc:
            logger.error("证据检索失败: %s", exc)
            return []


_rag_service_singleton: RAGService | None = None


def get_rag_service() -> RAGService:
    """返回进程级单例 RAGService，避免重复加载 embedding 模型。"""
    global _rag_service_singleton
    if _rag_service_singleton is None:
        _rag_service_singleton = RAGService()
    return _rag_service_singleton


# ── Reranker ───────────────────────────────────────────────────────────────────

_reranker_singleton: Any | None = None
_reranker_lock = asyncio.Lock()
# CrossEncoder 同为非线程安全：串行化 predict，避免并发重排死锁
_rerank_predict_lock = asyncio.Lock()


async def _get_reranker() -> Any:
    """延迟加载 CrossEncoder reranker 单例（sentence-transformers）。
    用 Lock 防止并发子问题同时触发多次模型加载。
    """
    global _reranker_singleton
    if _reranker_singleton is not None:
        return _reranker_singleton
    async with _reranker_lock:
        if _reranker_singleton is None:   # double-checked locking
            model_name = settings.reranker_model

            def _load():
                import torch
                from sentence_transformers import CrossEncoder
                device = "cuda" if torch.cuda.is_available() else "cpu"
                return CrossEncoder(model_name, device=device)

            _reranker_singleton = await asyncio.to_thread(_load)
            logger.info("Reranker 已加载: %s", model_name)
    return _reranker_singleton


async def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """用 CrossEncoder 对 chunks 重排序，返回前 top_k 条。"""
    if not chunks:
        return chunks
    k = top_k or settings.reranker_top_k
    reranker = await _get_reranker()
    # 截断上限覆盖整个 chunk（切分后 ≤chunk_size≈800 字符），避免只按前半段打分
    pairs = [(query, c.get("text", "")[:1024]) for c in chunks]

    def _predict() -> list[float]:
        return reranker.predict(pairs).tolist()

    async with _rerank_predict_lock:
        scores = await asyncio.to_thread(_predict)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    logger.info(
        "Rerank: %d → %d chunks，top分: %.3f",
        len(chunks), min(k, len(chunks)), ranked[0][0] if ranked else 0,
    )
    return [c for _, c in ranked[:k]]
