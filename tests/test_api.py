"""API 路由测试。"""

from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.fixture
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient) -> None:
    """测试健康检查端点。"""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient) -> None:
    """测试根端点。"""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "docs" in data


@pytest.mark.asyncio
async def test_create_task_empty_query(client: AsyncClient) -> None:
    """测试空查询校验。"""
    response = await client.post("/api/research", json={"query": ""})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_task_too_short_query(client: AsyncClient) -> None:
    """测试过短查询校验。"""
    response = await client.post("/api/research", json={"query": "a"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_task_not_found(client: AsyncClient) -> None:
    """测试获取不存在的任务。"""
    response = await client.get("/api/research/nonexistent/status")
    assert response.status_code == 404
