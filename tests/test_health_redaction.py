"""/health 明细遮蔽测试。

/health 经 Nginx 公网可达，dependencies[].detail 里是内部主机名、端口、数据库用户名、
组件版本号和失败时的异常原文。默认必须一律不返回，只有带对令牌的请求才看得到。
"""

from __future__ import annotations

import pytest

from app.core.config import settings
from app.services.health_service import detail_allowed, redact_health


@pytest.fixture
def _token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "health_detail_token", "tok-abcdef123456")
    return "tok-abcdef123456"


def test_detail_denied_when_token_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """未配置令牌 = 默认关闭。此时任何请求头都不该开门，包括空串。"""
    monkeypatch.setattr(settings, "health_detail_token", "")
    assert detail_allowed(None) is False
    assert detail_allowed("") is False
    assert detail_allowed("随便什么值") is False


def test_detail_requires_exact_token(_token: str) -> None:
    assert detail_allowed(_token) is True
    assert detail_allowed(None) is False
    assert detail_allowed("") is False
    assert detail_allowed(_token + "x") is False
    # 前缀正确也必须拒绝——compare_digest 不能因为长度不同就被绕过
    assert detail_allowed(_token[:-1]) is False


def test_redact_removes_every_detail() -> None:
    raw = {
        "status": "degraded",
        "failed": ["database"],
        "qdrant_mode": "remote",
        "qdrant_connected": True,
        "dependencies": [
            {
                "name": "database",
                "ok": False,
                "detail": "OperationalError: (1045, \"Access denied for 'deepresearch'@'172.18.0.5'\")",
            },
            {"name": "redis", "ok": True, "detail": "redis://:***@redis:6379/0"},
            {"name": "queue", "ok": True, "skipped": True, "detail": "队列未启用"},
        ],
    }
    out = redact_health(raw)

    # detail 一个不留
    assert [d["detail"] for d in out["dependencies"]] == ["", "", ""]
    # 但判定所需的字段必须原样保留：前端状态条靠它们区分阻断与降级
    assert out["status"] == "degraded"
    assert out["failed"] == ["database"]
    assert [(d["name"], d["ok"]) for d in out["dependencies"]] == [
        ("database", False),
        ("redis", True),
        ("queue", True),
    ]
    assert out["dependencies"][2]["skipped"] is True
    # 内部主机名不能从任何角落漏出去
    assert "172.18.0.5" not in str(out)
    assert "redis:6379" not in str(out)


@pytest.mark.asyncio
async def test_health_endpoint_hides_detail_without_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """端到端：真正走一遍 HTTP，确认遮蔽接在了端点上而不只是存在于工具函数里。"""
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    monkeypatch.setattr(settings, "health_detail_token", "tok-abcdef123456")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        anon = (await ac.get("/health")).json()
        authed = (await ac.get("/health", headers={"X-Health-Token": "tok-abcdef123456"})).json()
        wrong = (await ac.get("/health", headers={"X-Health-Token": "nope"})).json()

    assert all(d["detail"] == "" for d in anon["dependencies"])
    assert all(d["detail"] == "" for d in wrong["dependencies"])
    # 带对令牌时至少有一项能给出 detail，否则这个测试即使遮蔽失效也会通过
    assert any(d["detail"] for d in authed["dependencies"])

    # 判定字段无论遮蔽与否都必须在。不去比较两次调用的 status 是否相等——每次请求
    # 都是一轮真实探测，依赖的可用性本来就会在两次调用之间变化，那种断言是 flaky 的。
    # 「遮蔽不改变判定」由 test_redact_removes_every_detail 用固定输入覆盖。
    for body in (anon, authed, wrong):
        assert body["status"] in ("ok", "degraded")
        assert isinstance(body["failed"], list)
        assert [d["name"] for d in body["dependencies"]] == [
            d["name"] for d in anon["dependencies"]
        ]


@pytest.mark.asyncio
async def test_probe_store_is_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    """探测客户端要复用 —— 这正是把 /health 从 ~40ms 降到 ~12ms 的原因。"""
    import app.services.health_service as hs
    import app.services.vector_store as vsmod

    built = []

    class _Stub:
        def __init__(self) -> None:
            built.append(self)
            self.mode = "remote"

        async def close(self) -> None:
            pass

    monkeypatch.setattr(vsmod, "VectorStoreService", _Stub)
    monkeypatch.setattr(settings, "qdrant_mode", "remote")
    monkeypatch.setattr(hs, "_probe_store", None)

    first = await hs._get_probe_store()
    second = await hs._get_probe_store()

    assert first is second
    assert len(built) == 1


@pytest.mark.asyncio
async def test_probe_store_rebuilt_after_memory_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """复用的代价是状态粘滞：首次探测撞上 Qdrant 没起来，实例会永久回退内存模式。

    必须丢弃重建，否则 Qdrant 恢复后 /health 仍旧上报 memory，直到有人重启进程。
    """
    import app.services.health_service as hs
    import app.services.vector_store as vsmod

    closed = []

    class _Fallen:
        mode = "memory"

        async def close(self) -> None:
            closed.append(self)

    class _Healthy:
        mode = "remote"

        async def close(self) -> None:
            pass

    stale = _Fallen()
    monkeypatch.setattr(vsmod, "VectorStoreService", _Healthy)
    monkeypatch.setattr(settings, "qdrant_mode", "remote")
    monkeypatch.setattr(hs, "_probe_store", stale)

    fresh = await hs._get_probe_store()

    assert fresh is not stale
    assert fresh.mode == "remote"
    assert closed == [stale], "旧实例必须关掉，否则连接池会泄漏"


@pytest.mark.asyncio
async def test_probe_store_kept_when_memory_is_the_configured_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """memory 是配置值而非回退结果时，不该被反复重建。"""
    import app.services.health_service as hs
    import app.services.vector_store as vsmod

    class _Stub:
        mode = "memory"

        async def close(self) -> None:
            pass

    stub = _Stub()
    monkeypatch.setattr(vsmod, "VectorStoreService", _Stub)
    monkeypatch.setattr(settings, "qdrant_mode", "memory")
    monkeypatch.setattr(hs, "_probe_store", stub)

    assert await hs._get_probe_store() is stub


def test_redact_does_not_mutate_input() -> None:
    """verify_on_startup 拿同一个 dict 写日志，遮蔽不能就地改坏它。"""
    raw = {
        "status": "ok",
        "failed": [],
        "dependencies": [{"name": "redis", "ok": True, "detail": "redis://:***@redis:6379/0"}],
    }
    redact_health(raw)
    assert raw["dependencies"][0]["detail"] == "redis://:***@redis:6379/0"
