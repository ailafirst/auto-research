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


async def on_startup(ctx: dict) -> None:
    setup_logging()
    from app.services.db import init_db
    await init_db()
    logger.info("arq worker 启动，数据库就绪")


async def on_shutdown(ctx: dict) -> None:
    from app.services.db import close_db
    await close_db()
    logger.info("arq worker 关闭")


class WorkerSettings:
    """arq worker 配置。"""

    functions = [run_research_job]
    redis_settings = RedisSettings.from_dsn(settings.redis_url or "redis://localhost:6379/0")
    max_jobs = settings.worker_max_jobs      # 单进程内并发任务数
    job_timeout = settings.job_timeout       # 单任务超时（秒）
    on_startup = on_startup
    on_shutdown = on_shutdown
