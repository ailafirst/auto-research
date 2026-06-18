"""API 请求/响应 Schema。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """统一错误响应。"""
    detail: str
    code: str = "INTERNAL_ERROR"
    task_id: str | None = None


class HealthResponse(BaseModel):
    """健康检查响应。"""
    status: str = "ok"
    version: str = "0.1.0"
    qdrant_connected: bool = False
    qdrant_mode: str = "remote"  # remote / memory / unavailable
