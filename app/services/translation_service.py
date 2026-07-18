"""翻译服务 — 本地 MT 模型（opus-mt），用于跨语言检索的 query 译写。

把中文 query 译为英文作为第二路检索，解决"中文问题→英文 gold"的 dense 跨语言
对齐失效。走独立本地小模型（opus-mt，~77M/20ms），不占用主 LLM 并发闸门。
"""

from __future__ import annotations

import asyncio
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class TranslationService:
    """本地 MT 翻译服务（懒加载、串行推理）。"""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._load_lock = asyncio.Lock()
        # Marian/PyTorch 实例非线程安全：并发 generate 会死锁，串行化（单次 ~20ms）
        self._predict_lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:   # double-checked locking
                return

            def _load():
                import torch
                from transformers import MarianMTModel, MarianTokenizer
                device = "cuda" if torch.cuda.is_available() else "cpu"
                tok = MarianTokenizer.from_pretrained(settings.translation_model)
                model = MarianMTModel.from_pretrained(
                    settings.translation_model
                ).to(device).eval()
                return tok, model, device

            self._tokenizer, self._model, self._device = await asyncio.to_thread(_load)
            logger.info(
                "翻译模型已加载: %s (device=%s)",
                settings.translation_model, self._device,
            )

    async def translate(self, text: str) -> str:
        """翻译单条文本；失败时返回空串（调用方据此降级为单路检索）。

        model_service_url 配置时优先走远程模型服务；不可达时自动回退本地加载。
        """
        if not text or not text.strip():
            return ""

        if settings.model_service_url:
            try:
                return await self._translate_via_service(text)
            except Exception as exc:
                logger.warning("模型服务 translate 调用失败，回退本地加载: %s", exc)

        return await self._translate_local(text)

    async def _translate_local(self, text: str) -> str:
        """本地 MT 推理 —— 被 translate() 的本地路径与模型服务的 /translate
        端点共用（模型服务进程即是这份翻译模型单例的持有者，不应再转发 HTTP）。"""
        await self._ensure_loaded()

        def _run() -> str:
            import torch
            inp = self._tokenizer(
                [text], return_tensors="pt", padding=True,
                truncation=True, max_length=128,
            ).to(self._device)
            with torch.no_grad():
                out = self._model.generate(**inp, max_length=128, num_beams=4)
            return self._tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()

        async with self._predict_lock:
            try:
                return await asyncio.to_thread(_run)
            except Exception as exc:
                logger.warning("翻译失败，降级为单路检索: %s", exc)
                return ""

    async def _translate_via_service(self, text: str) -> str:
        """通过模型服务 HTTP 调用远程翻译（复用 rag_service 的共享 HTTP 客户端）。"""
        from app.services.rag_service import _get_http_client
        client = _get_http_client()
        resp = await client.post("/translate", json={"text": text})
        resp.raise_for_status()
        return resp.json()["translation"]


_translation_singleton: TranslationService | None = None


def get_translation_service() -> TranslationService:
    """返回进程级单例 TranslationService，避免重复加载 MT 模型。"""
    global _translation_singleton
    if _translation_singleton is None:
        _translation_singleton = TranslationService()
    return _translation_singleton
