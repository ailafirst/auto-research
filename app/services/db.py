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
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import mask_dsn, settings

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
    # 完整 ResearchTask.to_dict() 的 JSON。MySQL 的 TEXT 上限是 64KB，而这一列装着
    # 研究计划 + 信源评估 + 报告全文——当前实测最大 11KB，但 20 信源的长任务逼近
    # 上限完全可能，超了会直接 "Data too long for column"。MySQL 上用 LONGTEXT；
    # SQLite 的 TEXT 本就无长度限制，保持不变。
    data: Mapped[str] = mapped_column(
        Text().with_variant(LONGTEXT, "mysql"), default="{}"
    )


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
        engine_kwargs: dict = {"future": True}
        if not url.startswith("sqlite"):
            # worker 常常几天没有任务，连接池里的连接会闲置超过 MySQL 默认
            # wait_timeout（8h）。服务端悄悄断开后客户端并不知情，池里拿到这条
            # 死连接执行首条 SQL 就会报 "Lost connection to MySQL server during
            # query"（errno 2013）——现象是空闲一段时间后提交的头一两个任务必失败，
            # 之后的任务又正常（坏连接被自动剔出池）。pre_ping 在每次取用前探活，
            # recycle 在连接活满 30 分钟后主动换新，双保险覆盖闲置和网络抖动两类断连。
            # 仅对非 sqlite 生效：sqlite 没有服务端超时这回事，若 DATABASE_URL 指向
            # :memory:，recycle 换连接等于清空整个内存库，纯粹有害无益。
            engine_kwargs["pool_pre_ping"] = True
            engine_kwargs["pool_recycle"] = 1800
        _engine = create_async_engine(url, **engine_kwargs)
        if _engine.dialect.name == "sqlite":
            @event.listens_for(_engine.sync_engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _record):  # noqa: ANN001
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")     # 读写不互斥、多进程共享
                cur.execute("PRAGMA busy_timeout=5000")    # 写锁竞争时最多等 5s，避免 locked
                cur.execute("PRAGMA synchronous=NORMAL")
                cur.close()
        logger.info("数据库引擎已创建: %s", mask_dsn(url))
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
