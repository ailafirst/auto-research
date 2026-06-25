"""
Deep Research Agent — FastAPI 应用入口。

启动方式:
    # 开发模式
    uvicorn app.main:app --reload --port 8000

    # 生产模式
    uvicorn app.main:app --host 0.0.0.0 --port 8000

    # 直接运行
    python -m app.main
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes_research import router as research_router
from app.api.schemas import HealthResponse
from app.core.config import settings
from app.core.exceptions import DeepResearchError
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    setup_logging()
    logger.info("Deep Research Agent 启动中...")
    logger.info("环境: %s, LLM: %s/%s", settings.app_env, settings.llm_provider, settings.llm_model)

    # 检查 Qdrant 连接
    try:
        from app.services.vector_store import VectorStoreService
        vs = VectorStoreService()
        qdrant_ok = await vs.health_check()
        if qdrant_ok:
            logger.info("Qdrant 连接成功: %s", settings.qdrant_url)
        else:
            logger.warning("Qdrant 不可用: %s", settings.qdrant_url)
        await vs.close()
    except Exception as exc:
        logger.warning("Qdrant 连接检查失败: %s", exc)

    # 预热 embedding / reranker 模型
    # 首次加载 bge-m3 约 20-30s，bge-reranker 约 5-10s；在 lifespan 里提前加载，
    # 使第一个真实研究请求的 evidence_builder 和 analyst 节点不需要等待模型冷启动。
    try:
        from app.services.rag_service import get_rag_service, _get_reranker
        rag = get_rag_service()
        if settings.embedding_provider == "st":
            await rag._get_st_model()
        elif settings.embedding_provider == "fastembed":
            await rag._get_fastembed_model()
        logger.info("Embedding 模型预热完成: %s", settings.embedding_model)
        if settings.reranker_enabled:
            await _get_reranker()
            logger.info("Reranker 模型预热完成: %s", settings.reranker_model)
    except Exception as exc:
        logger.warning("模型预热失败（不影响启动）: %s", exc)

    yield

    logger.info("Deep Research Agent 关闭")


app = FastAPI(
    title="Deep Research Agent",
    description="基于 LangGraph、RAG 与 FastAPI 的自动化深度调研系统",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(research_router)


# 全局异常处理
@app.exception_handler(DeepResearchError)
async def deep_research_error_handler(request: Request, exc: DeepResearchError):
    return JSONResponse(
        status_code=400,
        content={"detail": exc.message, "code": exc.code},
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    logger.exception("未处理的异常: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误", "code": "INTERNAL_ERROR"},
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """健康检查端点。"""
    return HealthResponse(qdrant_connected=True, qdrant_mode="memory")


@app.get("/")
async def root():
    return {
        "name": "Deep Research Agent",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


def start_cli() -> None:
    """CLI 入口。"""
    import argparse

    parser = argparse.ArgumentParser(description="Deep Research Agent CLI")
    parser.add_argument("question", nargs="?", help="研究问题")
    parser.add_argument("--api", action="store_true", help="启动 API 服务")
    parser.add_argument("--host", default="0.0.0.0", help="API 监听地址")
    parser.add_argument("--port", type=int, default=8000, help="API 监听端口")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")

    args = parser.parse_args()

    if args.api:
        import uvicorn
        setup_logging()
        uvicorn.run("app.main:app", host=args.host, port=args.port, reload=False)
    elif args.question:
        import asyncio
        setup_logging()
        asyncio.run(_run_cli_research(args.question))
    else:
        parser.print_help()


async def _run_cli_research(question: str) -> None:
    """CLI 研究模式。"""
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    from app.services.task_service import TaskService

    console = Console()
    task_service = TaskService()

    console.print(f"\n[bold cyan]🔬 Deep Research Agent[/bold cyan]")
    console.print(f"[dim]研究问题: {question}[/dim]\n")

    task = await task_service.create_task(query=question)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task_pbar = progress.add_task("[cyan]研究中...", total=100)

        from app.graph.builder import build_research_graph
        research_graph = build_research_graph()

        initial_state = {
            "task_id": task.task_id,
            "query": question,
            "language": "zh-CN",
            "max_rounds": 2,
            "current_round": 1,
            "status": "planning",
            "research_plan": {},
            "sub_questions": [],
            "search_queries": [],
            "search_results": [],
            "search_summaries": [],
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
            "created_at": "",
            "updated_at": "",
        }

        final_state = await research_graph.ainvoke(initial_state)

        def _update_progress(state: dict) -> None:
            pct = state.get("progress", 0)
            msg = state.get("progress_message", "")
            progress.update(task_pbar, completed=pct, description=f"[cyan]{msg}")

        _update_progress(final_state)

    # 输出报告
    report = final_state.get("final_report", "")
    if report:
        output_path = f"output/{task.task_id}.md"
        import os
        os.makedirs("output", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        console.print(f"\n[green]✅ 研究完成![/green]")
        console.print(f"📄 报告已保存: [bold]{output_path}[/bold]")
        console.print(f"\n[bold]--- 报告预览 ---[/bold]\n")
        console.print(report[:2000])
        console.print(f"\n[dim]... (完整报告共 {len(report)} 字符)[/dim]")
    else:
        console.print(f"\n[red]❌ 报告生成失败[/red]")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        start_cli()
    else:
        import uvicorn
        setup_logging()
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
