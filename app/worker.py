"""arq worker — 研究任务的独立执行进程池。

启动一个 worker 进程：
    arq app.worker.WorkerSettings

启动多个（worker 池，横向扩展）——在多个终端各跑一次上面的命令，
它们共享同一个 Redis 队列、自动分摊任务。

每个 worker 进程内并发 max_jobs 个任务；全局 LLM 速率由 Redis 令牌桶（P1）跨进程限住。
"""

from __future__ import annotations

import logging
import os

from arq import cron
from arq.connections import RedisSettings

from app.core.config import settings
from app.core.logging import setup_logging
from app.services.research_runner import run_research

logger = logging.getLogger(__name__)


async def run_research_job(ctx: dict, task_id: str) -> None:
    """arq 作业：执行一次完整研究流程。"""
    pid = os.getpid()
    logger.info("worker[pid=%s] 领取任务: %s", pid, task_id)
    # 旁路记录「哪个 worker 进程领了这个任务」，供 benchmark 观测跨进程分摊（不污染进度消息）
    try:
        await ctx["redis"].set(f"dr:jobworker:{task_id}", str(pid), ex=3600)
    except Exception:
        pass
    await run_research(task_id)


async def cleanup_expired_vectors(ctx: dict) -> None:
    """定时清理过期的证据向量（仅 remote 模式有意义）。

    放在 worker 而不是 API：worker 本来就是长驻的后台进程，而 arq 的 cron 已经带了
    跨进程去重（CronJob unique=True），多个 worker 副本只会有一个真正执行。
    """
    if settings.qdrant_mode != "remote" or settings.vector_ttl_days <= 0:
        return

    from app.services.vector_store import VectorStoreService

    vs = VectorStoreService()
    try:
        await vs.delete_expired(settings.vector_ttl_days)
    finally:
        await vs.close()


async def on_startup(ctx: dict) -> None:
    setup_logging()
    from app.services.db import init_db
    await init_db()
    logger.info("arq worker 启动，数据库就绪")

    # worker 才是真正跑 RAG 的进程，模型服务不可达在这里后果最重：容器镜像不含
    # torch，rag_service 那条「回退本地加载」会直接 ImportError。启动即校验，把
    # 结论提前到领任务之前，而不是等第一个任务跑到 evidence_builder 才炸。
    from app.services.health_service import close_probe_store, verify_on_startup
    await verify_on_startup()
    # worker 没有 /health 端点，这次校验是一次性的，探测客户端用完即放。留着它只是
    # 让每个副本白白占一条到 Qdrant 的空闲连接——定时清理任务自带独立实例。
    await close_probe_store()


async def on_shutdown(ctx: dict) -> None:
    from app.services.db import close_db
    await close_db()
    logger.info("arq worker 关闭")


class WorkerSettings:
    """arq worker 配置。"""

    functions = [run_research_job]
    # unique=True：多个 worker 副本共享同一队列时，同一时刻只有一个真正执行
    cron_jobs = [
        cron(
            cleanup_expired_vectors,
            hour=settings.vector_cleanup_hour,
            minute=0,
            unique=True,
        )
    ]
    redis_settings = RedisSettings.from_dsn(settings.redis_url or "redis://localhost:6379/0")
    max_jobs = settings.worker_max_jobs      # 单进程内并发任务数
    job_timeout = settings.job_timeout       # 单任务超时（秒）
    on_startup = on_startup
    on_shutdown = on_shutdown
