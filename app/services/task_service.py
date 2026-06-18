"""任务管理服务 — 创建、查询、管理研究任务。"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.exceptions import TaskNotFoundError
from app.models.task import (
    ResearchTask,
    TaskDetailResponse,
    TaskReportResponse,
    TaskStatusResponse,
)

logger = logging.getLogger(__name__)


class TaskService:
    """任务管理服务。

    使用内存存储 + JSON 文件持久化，后续可升级为数据库。
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._tasks: dict[str, ResearchTask] = {}
        self._data_dir = Path(data_dir or "output/tasks")
        self._data_dir.mkdir(parents=True, exist_ok=True)

    async def create_task(self, query: str, **kwargs: Any) -> ResearchTask:
        """创建新研究任务。"""
        task = ResearchTask(
            query=query,
            max_rounds=kwargs.get("max_rounds", settings.max_rounds),
            language=kwargs.get("language", "zh-CN"),
            report_type=kwargs.get("report_type", "deep"),
            search_depth=kwargs.get("search_depth", "advanced"),
            top_k=kwargs.get("top_k", settings.rag_top_k),
            enable_fact_check=kwargs.get("enable_fact_check", True),
        )
        self._tasks[task.task_id] = task
        self._persist_task(task)
        logger.info("任务已创建: task_id=%s, query='%s'", task.task_id, query[:50])
        return task

    async def get_task(self, task_id: str) -> ResearchTask:
        """获取任务。"""
        task = self._tasks.get(task_id)
        if not task:
            # 尝试从文件恢复
            task = self._load_task(task_id)
            if not task:
                raise TaskNotFoundError(task_id)
        return task

    async def update_task(self, task_id: str, **updates: Any) -> ResearchTask:
        """更新任务。"""
        task = await self.get_task(task_id)
        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)
        task.updated_at = datetime.now().isoformat()
        self._persist_task(task)
        return task

    async def get_task_status(self, task_id: str) -> TaskStatusResponse:
        """获取任务状态。"""
        task = await self.get_task(task_id)
        return TaskStatusResponse(
            task_id=task.task_id,
            status=task.status,
            progress=task.progress,
            progress_message=task.progress_message,
            current_round=task.current_round,
            max_rounds=task.max_rounds,
        )

    async def get_task_detail(self, task_id: str) -> TaskDetailResponse:
        """获取任务详情。"""
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
            created_at=task.created_at,
            updated_at=task.updated_at,
            error_message=task.error_message,
        )

    async def get_task_report(self, task_id: str) -> TaskReportResponse:
        """获取任务报告。"""
        task = await self.get_task(task_id)
        return TaskReportResponse(
            task_id=task.task_id,
            status=task.status,
            report=task.final_report,
            error_message=task.error_message,
        )

    async def list_tasks(self) -> list[TaskStatusResponse]:
        """列出所有任务。"""
        return [
            TaskStatusResponse(
                task_id=t.task_id,
                status=t.status,
                progress=t.progress,
                progress_message=t.progress_message,
                current_round=t.current_round,
                max_rounds=t.max_rounds,
            )
            for t in self._tasks.values()
        ]

    def _persist_task(self, task: ResearchTask) -> None:
        """持久化任务到 JSON 文件。"""
        try:
            path = self._data_dir / f"{task.task_id}.json"
            path.write_text(
                json.dumps(task.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("任务持久化失败: %s, error=%s", task.task_id, exc)

    def _load_task(self, task_id: str) -> ResearchTask | None:
        """从 JSON 文件恢复任务。"""
        try:
            path = self._data_dir / f"{task_id}.json"
            if not path.exists():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            task = ResearchTask(**data)
            self._tasks[task_id] = task
            return task
        except Exception as exc:
            logger.warning("任务恢复失败: %s, error=%s", task_id, exc)
            return None
