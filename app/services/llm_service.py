"""LLM 调用服务 — 基于 LiteLLM 统一接口。"""

from __future__ import annotations

import logging
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.exceptions import LLMServiceError

logger = logging.getLogger(__name__)


class LLMService:
    """统一的 LLM 调用服务，支持多种模型提供商。"""

    def __init__(self) -> None:
        self.provider = settings.llm_provider
        self.model = settings.llm_model
        self.api_key = settings.llm_api_key
        self.base_url = settings.llm_base_url
        self.max_tokens = settings.llm_max_tokens
        self.temperature = settings.llm_temperature

    def _get_model_name(self) -> str:
        """获取完整的模型名称（含 provider 前缀）。

        对于自定义 OpenAI 兼容 API（如设置了 base_url），使用 openai/ 前缀。
        """
        if "/" in self.model:
            return self.model
        # 自定义 base_url 的服务用 openai/ 前缀
        if self.base_url or self.provider.lower() in ("openai", "xiaomi", "custom"):
            return f"openai/{self.model}"
        return f"{self.provider}/{self.model}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """调用 LLM 进行对话。"""
        try:
            import litellm

            kwargs: dict[str, Any] = {
                "model": self._get_model_name(),
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
            }

            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["api_base"] = self.base_url
            if response_format:
                kwargs["response_format"] = response_format

            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content or ""

            # Token 统计
            if hasattr(response, "usage"):
                logger.debug(
                    "LLM 调用完成 — model=%s, input_tokens=%d, output_tokens=%d",
                    self.model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

            return content

        except Exception as exc:
            logger.error("LLM 调用失败: %s", exc)
            raise LLMServiceError(str(exc), provider=self.provider) from exc

    async def chat_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type,
        temperature: float | None = None,
    ) -> Any:
        """调用 LLM 并返回结构化输出（JSON mode）。"""
        content = await self.chat(
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )

        try:
            import json
            data = json.loads(content)
            return response_model(**data)
        except (json.JSONDecodeError, Exception) as exc:
            logger.error("结构化输出解析失败: %s, content=%s", exc, content[:200])
            raise LLMServiceError(f"结构化输出解析失败: {exc}", provider=self.provider) from exc
