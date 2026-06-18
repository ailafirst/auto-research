"""数据模型测试。"""

from __future__ import annotations

from app.models.source import CrawledDocument, FactCheckResult, SearchResult
from app.models.task import ResearchTask, ResearchTaskCreate


def test_research_task_defaults() -> None:
    """测试 ResearchTask 默认值。"""
    task = ResearchTask(query="测试问题")
    assert task.task_id.startswith("task_")
    assert task.status == "pending"
    assert task.max_rounds == 2
    assert task.language == "zh-CN"
    assert task.enable_fact_check is True
    assert task.progress == 0


def test_research_task_update_status() -> None:
    """测试状态更新。"""
    task = ResearchTask(query="测试")
    task.update_status("planning", progress=10, message="正在规划...")
    assert task.status == "planning"
    assert task.progress == 10
    assert task.progress_message == "正在规划..."


def test_research_task_create_validation() -> None:
    """测试创建请求校验。"""
    # 正常数据
    params = ResearchTaskCreate(query="测试研究问题")
    assert params.query == "测试研究问题"

    # 过短查询应报错
    try:
        ResearchTaskCreate(query="a")
        assert False, "应校验失败"
    except Exception:
        pass


def test_search_result_model() -> None:
    """测试搜索结果模型。"""
    result = SearchResult(
        title="测试标题",
        url="https://example.com/article",
        snippet="测试摘要",
        position=1,
    )
    assert result.title == "测试标题"
    assert result.source == "example.com"  # 自动提取域名
    assert result.position == 1


def test_crawled_document_model() -> None:
    """测试已抓取文档模型。"""
    doc = CrawledDocument(
        url="https://example.com",
        title="测试文档",
        content="测试内容" * 100,
    )
    assert doc.text_length == len("测试内容" * 100)


def test_crawled_document_with_error() -> None:
    """测试抓取失败的文档。"""
    doc = CrawledDocument(
        url="https://example.com/404",
        title="",
        content="",
        error="HTTP 404",
    )
    assert doc.error == "HTTP 404"
    assert doc.text_length == 0


def test_fact_check_result() -> None:
    """测试事实核查结果。"""
    result = FactCheckResult(passed=True)
    assert result.passed is True
    assert len(result.issues) == 0
    assert len(result.follow_up_queries) == 0
