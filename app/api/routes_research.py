"""研究任务 API 路由。"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from app.graph.builder import build_research_graph
from app.models.task import (
    ResearchTaskCreate,
    TaskDetailResponse,
    TaskReportResponse,
    TaskStatusResponse,
)
from app.services.task_service import TaskService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/research", tags=["research"])
task_service = TaskService()


@router.post("", response_model=dict[str, str], status_code=201)
async def create_research_task(params: ResearchTaskCreate) -> dict[str, str]:
    """创建新的研究任务。"""
    task = await task_service.create_task(
        query=params.query,
        max_rounds=params.max_rounds,
        language=params.language,
        report_type=params.report_type,
        search_depth=params.search_depth,
        top_k=params.top_k,
        enable_fact_check=params.enable_fact_check,
    )

    # 异步启动研究流程
    asyncio.create_task(_run_research(task.task_id))

    return {"task_id": task.task_id, "status": task.status}


@router.get("/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """查询任务状态。"""
    try:
        return await task_service.get_task_status(task_id)
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


async def _run_research(task_id: str) -> None:
    """在后台执行研究流程，使用流式获取实时进度。"""
    try:
        task = await task_service.get_task(task_id)

        # 构建 LangGraph 工作流
        research_graph = build_research_graph()

        # 状态映射：node_name → 友好名称
        NODE_LABELS = {
            "planner": "正在分析研究问题...",
            "retriever": "正在搜索相关资料...",
            "content_extractor": "正在抓取网页内容...",
            "source_evaluator": "正在评估信源质量...",
            "evidence_builder": "正在构建证据索引...",
            "analyst": "正在分析研究内容...",
            "fact_checker": "正在事实核查...",
            "report_writer": "正在生成研究报告...",
        }
        NODE_PROGRESS = {
            "planner": 10,
            "retriever": 25,
            "content_extractor": 40,
            "source_evaluator": 50,
            "evidence_builder": 60,
            "analyst": 70,
            "fact_checker": 80,
            "report_writer": 90,
        }

        def _build_state(round_num: int, extra: dict | None = None) -> dict:
            state = {
                "task_id": task_id,
                "query": task.query,
                "language": task.language,
                "max_rounds": task.max_rounds,
                "current_round": round_num,
                "status": "planning",
                "research_plan": {},
                "sub_questions": [],
                "search_queries": [],
                "search_results": [],
                "crawled_documents": [],
                "evaluated_sources": [],
                "evidence_chunks": [],
                "sub_answers": [],
                "fact_check_result": {},
                "fact_check_passed": True,
                "follow_up_queries": [],
                "final_report": "",
                "errors": [],
                "progress": 0,
                "progress_message": "",
                "created_at": task.created_at,
                "updated_at": task.updated_at,
            }
            if extra:
                state.update(extra)
            return state

        async def _run_round(current_state: dict) -> dict:
            """执行一轮研究，流式获取进度。"""
            final_state = current_state.copy()
            async for chunk in research_graph.astream(
                current_state, stream_mode="updates",
            ):
                # chunk 格式: {node_name: state_update}
                for node_name, state_update in chunk.items():
                    final_state.update(state_update)

                    label = NODE_LABELS.get(node_name, f"正在执行 {node_name}...")
                    pct = state_update.get("progress") or NODE_PROGRESS.get(node_name, 0)
                    msg = state_update.get("progress_message") or label

                    await task_service.update_task(
                        task_id,
                        status=node_name,
                        progress=pct,
                        progress_message=msg,
                    )
            return final_state

        # 第一轮研究
        initial_state = _build_state(1)
        final_state = await _run_round(initial_state)

        # 多轮补充研究
        current_round = 1
        while current_round < task.max_rounds:
            if final_state.get("fact_check_passed", True):
                break

            follow_up = final_state.get("follow_up_queries", [])
            if not follow_up:
                break

            current_round += 1

            await task_service.update_task(
                task_id,
                status="retrieving",
                current_round=current_round,
                progress_message=f"第 {current_round} 轮补充研究...",
            )

            new_state = _build_state(current_round, {
                "search_queries": follow_up[:5],
                "search_results": [],
                "crawled_documents": [],
                "evaluated_sources": [],
            })
            final_state = await _run_round(new_state)

        # 更新任务结果为完成
        report = final_state.get("final_report", "")
        await task_service.update_task(
            task_id,
            status="completed",
            progress=100,
            progress_message="研究完成",
            final_report=report,
        )

        logger.info("研究任务完成: task_id=%s, 轮数=%d", task_id, current_round)

    except asyncio.CancelledError:
        logger.warning("研究任务被取消: %s", task_id)
        await _fail_task(task_id, "任务被取消")
    except Exception as exc:
        logger.error("研究任务失败: task_id=%s, error=%s", task_id, exc)
        await _fail_task(task_id, str(exc))


async def _fail_task(task_id: str, error_message: str) -> None:
    """标记任务失败。"""
    try:
        await task_service.update_task(
            task_id,
            status="failed",
            progress=0,
            progress_message="研究失败",
            error_message=error_message,
        )
    except Exception as exc:
        logger.error("更新任务失败状态出错: %s", exc)
