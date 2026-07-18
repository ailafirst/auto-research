"""任务管理服务 — SQLite（系统记录/冷）+ Redis（热进度）。

冷热分层：
- 持久记录（任务、状态、报告）落 SQLite → 重启/崩溃不丢、可查询、可分页。
- 高频进度快照写 Redis → 跨进程 /status 实时读，且为将来 pub/sub 推送留钩子。
公开方法签名与旧版一致，routes_research 与 CLI 无需改动。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.core.config import settings
from app.core.exceptions import TaskNotFoundError
from app.models.task import (
    ResearchTask,
    TaskDetailResponse,
    TaskProgressResponse,
    TaskReportResponse,
    TaskStatusResponse,
)
from app.services.cache_service import get_cache
from app.services.db import TaskRow, get_sessionmaker, init_db

logger = logging.getLogger(__name__)

_PROGRESS_TTL = 7 * 24 * 3600   # 热进度键保留 7 天


class TaskService:
    """任务管理服务（SQLite + Redis）。"""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        # data_dir 仅为兼容旧签名；实际路径由 DATABASE_URL 决定
        self._sm: Any | None = None

    async def _ensure(self) -> None:
        await init_db()
        if self._sm is None:
            self._sm = get_sessionmaker()

    # ── Redis 热进度 ─────────────────────────────────────────────────────────
    def _prog_key(self, task_id: str) -> str:
        return get_cache().key("taskprog", task_id)

    async def _write_progress(self, task: ResearchTask) -> None:
        await get_cache().set_json(self._prog_key(task.task_id), {
            "status": task.status,
            "progress": task.progress,
            "progress_message": task.progress_message,
            "current_round": task.current_round,
            "updated_at": task.updated_at,
        }, _PROGRESS_TTL)

    async def _read_progress(self, task_id: str) -> dict[str, Any] | None:
        return await get_cache().get_json(self._prog_key(task_id))

    # ── Redis 过程详情快照（中间产物，供前端「过程透明」）─────────────────────
    def _prog_detail_key(self, task_id: str) -> str:
        return get_cache().key("taskprogd", task_id)

    async def write_progress_detail(self, task_id: str, snapshot: dict[str, Any]) -> None:
        """写研究过程中间产物快照到 Redis 热层（量大高频，不落 SQLite）。
        Redis 不可用时静默旁路（CacheService 已处理），不阻断研究流程。"""
        await get_cache().set_json(self._prog_detail_key(task_id), snapshot, _PROGRESS_TTL)

    async def _read_progress_detail(self, task_id: str) -> dict[str, Any] | None:
        return await get_cache().get_json(self._prog_detail_key(task_id))

    # ── SQLite 持久 ──────────────────────────────────────────────────────────
    async def _save(self, task: ResearchTask) -> None:
        payload = json.dumps(task.to_dict(), ensure_ascii=False)
        async with self._sm() as s:
            row = await s.get(TaskRow, task.task_id)
            if row is None:
                row = TaskRow(task_id=task.task_id)
                s.add(row)
            row.status = task.status
            row.query = task.query
            row.created_at = task.created_at
            row.updated_at = task.updated_at
            row.data = payload
            await s.commit()

    async def _load(self, task_id: str) -> ResearchTask | None:
        async with self._sm() as s:
            row = await s.get(TaskRow, task_id)
        if row is None:
            return None
        return ResearchTask(**json.loads(row.data))

    # ── 公开接口（签名不变）─────────────────────────────────────────────────
    async def create_task(self, query: str, **kwargs: Any) -> ResearchTask:
        await self._ensure()
        task = ResearchTask(
            query=query,
            max_rounds=kwargs.get("max_rounds", settings.max_rounds),
            language=kwargs.get("language", "zh-CN"),
            report_type=kwargs.get("report_type", "deep"),
            search_depth=kwargs.get("search_depth", "advanced"),
            top_k=kwargs.get("top_k", settings.rag_top_k),
            enable_fact_check=kwargs.get("enable_fact_check", True),
        )
        await self._save(task)
        await self._write_progress(task)
        logger.info("任务已创建: task_id=%s, query='%s'", task.task_id, query[:50])
        return task

    async def get_task(self, task_id: str) -> ResearchTask:
        await self._ensure()
        task = await self._load(task_id)
        if task is None:
            raise TaskNotFoundError(task_id)
        # 叠加 Redis 热进度（在途任务的最新 tick，作为实时/兜底来源）
        prog = await self._read_progress(task_id)
        if prog:
            task.status = prog.get("status", task.status)
            task.progress = prog.get("progress", task.progress)
            task.progress_message = prog.get("progress_message", task.progress_message)
            task.current_round = prog.get("current_round", task.current_round)
        return task

    async def update_task(self, task_id: str, **updates: Any) -> ResearchTask:
        await self._ensure()
        task = await self._load(task_id)
        if task is None:
            raise TaskNotFoundError(task_id)
        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)
        task.updated_at = datetime.now().isoformat()
        await self._save(task)              # 冷：持久记录
        await self._write_progress(task)    # 热：进度快照
        return task

    async def get_task_status(self, task_id: str) -> TaskStatusResponse:
        task = await self.get_task(task_id)
        return TaskStatusResponse(
            task_id=task.task_id,
            query=task.query,
            status=task.status,
            progress=task.progress,
            progress_message=task.progress_message,
            current_round=task.current_round,
            max_rounds=task.max_rounds,
        )

    async def get_task_detail(self, task_id: str) -> TaskDetailResponse:
        task = await self.get_task(task_id)
        return TaskDetailResponse(
            task_id=task.task_id,
            query=task.query,
            status=task.status,
            progress=task.progress,
            progress_message=task.progress_message,
            current_round=task.current_round,
            max_rounds=task.max_rounds,
            research_plan=task.research_plan,
            final_report=task.final_report,
            fact_check_result=task.fact_check_result,
            created_at=task.created_at,
            updated_at=task.updated_at,
            error_message=task.error_message,
        )

    async def get_task_report(self, task_id: str) -> TaskReportResponse:
        task = await self.get_task(task_id)
        return TaskReportResponse(
            task_id=task.task_id,
            status=task.status,
            report=task.final_report,
            error_message=task.error_message,
        )

    async def get_task_progress(self, task_id: str) -> TaskProgressResponse:
        """状态（热进度）+ 过程详情快照（Redis）+ 报告（完成时）三合一，
        供前端单端点轮询做「过程透明」可视化。"""
        task = await self.get_task(task_id)
        d = await self._read_progress_detail(task_id) or {}
        return TaskProgressResponse(
            task_id=task.task_id,
            status=task.status,
            progress=task.progress,
            progress_message=task.progress_message,
            current_round=task.current_round,
            max_rounds=task.max_rounds,
            research_strategy=d.get("research_strategy", {}),
            sub_questions=d.get("sub_questions", []),
            search_queries=d.get("search_queries", []),
            search_summaries=d.get("search_summaries", []),
            sources=d.get("sources", []),
            crawled_count=d.get("crawled_count", 0),
            evidence_count=d.get("evidence_count", 0),
            citation_registry=d.get("citation_registry", []),
            sub_answers=d.get("sub_answers", []),
            fact_check=d.get("fact_check", {}),
            fact_check_passed=d.get("fact_check_passed", True),
            follow_up_queries=d.get("follow_up_queries", []),
            rounds=d.get("rounds", []),
            final_report=task.final_report,
            error_message=task.error_message,
            updated_at=task.updated_at,
        )

    async def list_tasks(self, limit: int = 200) -> list[TaskStatusResponse]:
        """列出任务（从 SQLite，按创建时间倒序分页）——修复旧版「只返回内存」。"""
        await self._ensure()
        async with self._sm() as s:
            rows = (
                await s.execute(
                    select(TaskRow).order_by(TaskRow.created_at.desc()).limit(limit)
                )
            ).scalars().all()
        out: list[TaskStatusResponse] = []
        for r in rows:
            try:
                t = ResearchTask(**json.loads(r.data))
            except Exception:
                continue
            out.append(TaskStatusResponse(
                task_id=t.task_id,
                query=t.query,
                status=t.status,
                progress=t.progress,
                progress_message=t.progress_message,
                current_round=t.current_round,
                max_rounds=t.max_rounds,
            ))
        return out
