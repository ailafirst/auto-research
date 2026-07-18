"""数据库 — SQLAlchemy 2.0 async。

默认 SQLite（WAL 模式，支持单机多 worker 共享同一文件）；DATABASE_URL 换成
mysql+aiomysql://... 或 postgresql+asyncpg://... 即可迁移到多机，代码不变。

冷热分层里这是「冷层 / 系统记录（SoR）」：只承接低频里程碑写（建任务、状态跃迁、
完成写报告）；高频进度 tick 走 Redis（见 task_service）。
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import String, Text, event
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class TaskRow(Base):
    """任务持久记录。可查询列 + 完整 ResearchTask JSON（首版够用，规范化后置）。"""

    __tablename__ = "tasks"

    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), index=True, default="pending")
    query: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[str] = mapped_column(String(40), index=True, default="")
    updated_at: Mapped[str] = mapped_column(String(40), default="")
    data: Mapped[str] = mapped_column(Text, default="{}")   # 完整 ResearchTask.to_dict() 的 JSON


_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker | None = None
_initialized = False


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        url = settings.database_url
        # SQLite：确保文件目录存在，并每连接设置 WAL + busy_timeout（并发友好）
        if url.startswith("sqlite"):
            dbfile = url.split(":///")[-1]
            if dbfile and dbfile != ":memory:":
                Path(dbfile).parent.mkdir(parents=True, exist_ok=True)
        _engine = create_async_engine(url, future=True)
        if _engine.dialect.name == "sqlite":
            @event.listens_for(_engine.sync_engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _record):  # noqa: ANN001
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")     # 读写不互斥、多进程共享
                cur.execute("PRAGMA busy_timeout=5000")    # 写锁竞争时最多等 5s，避免 locked
                cur.execute("PRAGMA synchronous=NORMAL")
                cur.close()
        logger.info("数据库引擎已创建: %s", url)
    return _engine


def get_sessionmaker() -> async_sessionmaker:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _sessionmaker


async def init_db() -> None:
    """建表（幂等）。"""
    global _initialized
    if _initialized:
        return
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    _initialized = True
    logger.info("数据库表已就绪")


async def close_db() -> None:
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
