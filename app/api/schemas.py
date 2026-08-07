"""API 请求/响应 Schema。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """统一错误响应。"""
    detail: str
    code: str = "INTERNAL_ERROR"
    task_id: str | None = None


class DependencyStatus(BaseModel):
    """单个依赖的探测结果。"""
    name: str
    ok: bool
    detail: str = ""
    skipped: bool = False        # 按配置本就不该存在，不计入健康判定
    mode: str | None = None      # 目前仅 qdrant 使用（memory / remote）


class HealthResponse(BaseModel):
    """健康检查响应。

    HTTP 恒为 200：进程活着就该返回 200，依赖缺失通过 status=degraded 表达。
    容器 healthcheck 据此判存活，运维和前端据 dependencies 判就绪。
    """
    status: str = "ok"           # ok / degraded
    version: str = "0.1.0"
    qdrant_connected: bool = False
    qdrant_mode: str = "remote"  # remote / memory / unavailable
    failed: list[str] = Field(default_factory=list)
    dependencies: list[DependencyStatus] = Field(default_factory=list)
