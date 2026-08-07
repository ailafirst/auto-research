"""把 SQLite 里的历史任务搬进 MySQL。

    python deployment/docker/migrate_sqlite_to_mysql.py \
        --sqlite output/deepresearch.db \
        --mysql "mysql+aiomysql://deepresearch:<MYSQL_PASSWORD>@127.0.0.1:3307/deepresearch"

口令取 deployment/docker/.env 里的 MYSQL_PASSWORD（该文件由 start-all.ps1 生成，
不入库）；3307 是 compose 映射到宿主机的回环端口，见同目录 compose.yaml。

只搬 tasks 一张表（当前 SoR 的全部内容）。幂等：按 task_id 做 upsert，重复执行不会
产生重复行，也可以在切换前跑一次、切换时再补跑一次增量。

目标表由 SQLAlchemy 按 app/services/db.py 的模型建（data 列在 MySQL 上是 LONGTEXT），
所以运行前 API 容器已经建过表也没关系，create_all 是幂等的。
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from app.services.db import Base, TaskRow  # noqa: E402

COLUMNS = ("task_id", "status", "query", "created_at", "updated_at", "data")


def read_sqlite(path: str) -> list[dict]:
    if not Path(path).exists():
        raise SystemExit(f"SQLite 文件不存在: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(f"SELECT {', '.join(COLUMNS)} FROM tasks").fetchall()
    except sqlite3.OperationalError as exc:
        raise SystemExit(f"读取 tasks 表失败: {exc}") from exc
    finally:
        conn.close()
    return [dict(r) for r in rows]


async def write_mysql(url: str, rows: list[dict]) -> tuple[int, int]:
    engine = create_async_engine(url, future=True)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        sm = async_sessionmaker(engine, expire_on_commit=False)
        inserted = updated = 0
        async with sm() as session:
            for row in rows:
                existing = await session.get(TaskRow, row["task_id"])
                if existing is None:
                    session.add(TaskRow(**row))
                    inserted += 1
                else:
                    for key, value in row.items():
                        setattr(existing, key, value)
                    updated += 1
            await session.commit()
        return inserted, updated
    finally:
        await engine.dispose()


async def main() -> None:
    parser = argparse.ArgumentParser(description="SQLite → MySQL 任务数据迁移")
    parser.add_argument("--sqlite", default="output/deepresearch.db")
    parser.add_argument("--mysql", required=True, help="mysql+aiomysql://user:pass@host:port/db")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写入")
    args = parser.parse_args()

    rows = read_sqlite(args.sqlite)
    biggest = max((len(r["data"] or "") for r in rows), default=0)
    print(f"SQLite 读到 {len(rows)} 行，data 列最大 {biggest:,} 字节")

    if args.dry_run:
        print("dry-run，未写入")
        return

    inserted, updated = await write_mysql(args.mysql, rows)
    print(f"MySQL 写入完成：新增 {inserted} 行，更新 {updated} 行")


if __name__ == "__main__":
    asyncio.run(main())
