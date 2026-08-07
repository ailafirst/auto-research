"""配置工具测试。"""

from __future__ import annotations

import pytest

from app.core.config import mask_dsn


@pytest.mark.parametrize(
    ("dsn", "expected"),
    [
        # 带口令的两种主力形态：MySQL 有用户名，Redis 只有口令
        (
            "mysql+aiomysql://deepresearch:s3cr3t@mysql:3306/deepresearch",
            "mysql+aiomysql://deepresearch:***@mysql:3306/deepresearch",
        ),
        ("redis://:s3cr3t@redis:6379/0", "redis://:***@redis:6379/0"),
        # 无口令的形态必须原样返回，否则 /health 会把可读信息也抹掉
        ("redis://redis:6379/0", "redis://redis:6379/0"),
        ("sqlite+aiosqlite:///output/deepresearch.db", "sqlite+aiosqlite:///output/deepresearch.db"),
        ("redis://user@redis:6379/0", "redis://user@redis:6379/0"),
        ("", ""),
    ],
)
def test_mask_dsn(dsn: str, expected: str) -> None:
    assert mask_dsn(dsn) == expected


def test_mask_dsn_password_containing_at_sign() -> None:
    """口令里含 @ 时按最后一个 @ 切分，不能把口令片段当成主机名漏出去。"""
    masked = mask_dsn("redis://:p@ss@redis:6379/0")
    assert masked == "redis://:***@redis:6379/0"
    assert "p@ss" not in masked
