"""模型服务 —— 独立 FastAPI 进程，集中装载 bge-m3(embed) + bge-reranker-v2-m3(rerank)
+ opus-mt(translate)，供多个 worker 通过 HTTP 共享同一份模型。

背景：三个模型本是各 worker 进程内的懒加载单例；多 worker 部署下每个进程各自
建一份 CUDA context + 加载权重（实测 commit ~6.9GB/进程），2 个 worker 曾在这台
机器上把系统提交内存(commit)顶穿导致其一被 Windows 硬毙（bash.exe.stackdump 那次）。
把三个模型收敛到本进程后，worker 只发 HTTP 请求，全程不 import torch（实测
commit 6.9GB → 0.75GB），worker 数量因此可以真正横向扩展。

复用 app.services.rag_service / app.services.translation_service 里已有的懒加载 +
锁 + fp16 逻辑，本文件只是把它们包一层 HTTP —— 加载/精度/并发串行的实现只有一处。

启动:
    uvicorn app.model_server:app --host 0.0.0.0 --port 8100

worker/API 侧启用（.env）:
    MODEL_SERVICE_URL=http://localhost:8100

服务不可达时，调用方（rag_service._embed_raw / rerank_chunks / translation_service.translate）
会自动回退本地加载，本服务不是单点故障。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)


# ── Schemas ──────────────────────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    vectors: list[list[float]]


class RerankRequest(BaseModel):
    query: str
    texts: list[str]


class RerankResponse(BaseModel):
    scores: list[float]


class TranslateRequest(BaseModel):
    text: str


class TranslateResponse(BaseModel):
    translation: str


class HealthResponse(BaseModel):
    status: str
    device: str
    dtype: str
    models: dict[str, bool]


# ── Lifespan：启动即预热三个模型，避免第一个真实请求承担冷启动 ──────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("模型服务启动中... dtype=%s", settings.model_dtype)

    from app.services.rag_service import _get_reranker, get_rag_service

    rag = get_rag_service()
    await rag._get_st_model()
    logger.info("Embedding 模型预热完成: %s", settings.embedding_model)

    if settings.reranker_enabled:
        await _get_reranker()
        logger.info("Reranker 模型预热完成: %s", settings.reranker_model)

    if settings.xling_enabled:
        from app.services.translation_service import get_translation_service
        # 用 _translate_local 而非 translate()：本进程即模型持有者，若 .env 里
        # 也配了 MODEL_SERVICE_URL（worker 与本服务通常共享同一份 .env），
        # 公开的 translate() 会先尝试走 HTTP —— 转头调用自己，必须绕开。
        await get_translation_service()._translate_local("预热")
        logger.info("翻译模型预热完成: %s", settings.translation_model)

    logger.info("模型服务就绪")
    yield
    logger.info("模型服务关闭")


app = FastAPI(title="Deep Research Model Service", lifespan=lifespan)


# ── 路由 ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    import torch

    from app.services.rag_service import _reranker_singleton, get_rag_service
    from app.services.translation_service import _translation_singleton

    rag = get_rag_service()
    return HealthResponse(
        status="ok",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=settings.model_dtype,
        models={
            "embed": rag._embedding_model is not None,
            "rerank": _reranker_singleton is not None,
            "translate": _translation_singleton is not None
            and _translation_singleton._model is not None,
        },
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    if not req.texts:
        return EmbedResponse(vectors=[])
    from app.services.rag_service import get_rag_service

    try:
        vectors = await get_rag_service()._embed_via_st(req.texts)
    except Exception as exc:
        logger.error("embed 失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return EmbedResponse(vectors=vectors)


@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest) -> RerankResponse:
    if not req.texts:
        return RerankResponse(scores=[])
    from app.services.rag_service import _rerank_scores

    try:
        scores = await _rerank_scores(req.query, req.texts)
    except Exception as exc:
        logger.error("rerank 失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RerankResponse(scores=scores)


@app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest) -> TranslateResponse:
    from app.services.translation_service import get_translation_service

    try:
        # _translate_local：本进程是模型持有者，不经 translate() 的服务路由判断
        result = await get_translation_service()._translate_local(req.text)
    except Exception as exc:
        logger.error("翻译失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return TranslateResponse(translation=result)
