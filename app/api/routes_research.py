"""研究任务 API 路由。"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.models.task import (
    ResearchTaskCreate,
    TaskDetailResponse,
    TaskProgressResponse,
    TaskReportResponse,
    TaskStatusResponse,
)
from app.services.queue import get_arq_pool
from app.services.research_runner import run_research
from app.services.task_service import TaskService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/research", tags=["research"])
task_service = TaskService()


@router.post("", response_model=dict[str, str], status_code=201)
async def create_research_task(params: ResearchTaskCreate) -> dict[str, str]:
    """创建新的研究任务。

    优先入 arq 队列由独立 worker 池执行（解耦、背压、横向扩展）；
    队列不可用时回退进程内 asyncio.create_task（单机开发无需另起 worker）。
    """
    task = await task_service.create_task(
        query=params.query,
        max_rounds=params.max_rounds,
        language=params.language,
        report_type=params.report_type,
        search_depth=params.search_depth,
        top_k=params.top_k,
        enable_fact_check=params.enable_fact_check,
    )

    enqueued = False
    if settings.queue_enabled:
        pool = await get_arq_pool()
        if pool is not None:
            try:
                # _job_id=task_id：同一任务重复提交自动去重（幂等）
                await pool.enqueue_job("run_research_job", task.task_id, _job_id=task.task_id)
                enqueued = True
            except Exception as exc:
                logger.warning("入队失败，回退进程内执行: %s", exc)

    if not enqueued:
        asyncio.create_task(run_research(task.task_id))

    logger.info("任务派发 [%s]: %s", "queue" if enqueued else "inproc", task.task_id)
    return {"task_id": task.task_id, "status": task.status}


@router.get("/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """查询任务状态。"""
    try:
        return await task_service.get_task_status(task_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{task_id}/progress", response_model=TaskProgressResponse)
async def get_task_progress(task_id: str) -> TaskProgressResponse:
    """研究过程实时快照（状态 + 中间产物），供前端过程透明可视化轮询。"""
    try:
        return await task_service.get_task_progress(task_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{task_id}", response_model=TaskDetailResponse)
async def get_task_detail(task_id: str) -> TaskDetailResponse:
    """获取任务详细信息。"""
    try:
        return await task_service.get_task_detail(task_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{task_id}/report", response_model=TaskReportResponse)
async def get_task_report(task_id: str) -> TaskReportResponse:
    """获取任务报告。"""
    try:
        return await task_service.get_task_report(task_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("", response_model=list[TaskStatusResponse])
async def list_tasks() -> list[TaskStatusResponse]:
    """列出所有任务。"""
    return await task_service.list_tasks()
