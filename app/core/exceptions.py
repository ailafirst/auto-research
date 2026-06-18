"""统一异常定义。"""

from __future__ import annotations


class DeepResearchError(Exception):
    """Deep Research 基础异常。"""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR") -> None:
        self.message = message
        self.code = code
        super().__init__(message)


class LLMServiceError(DeepResearchError):
    """LLM 调用异常。"""
    def __init__(self, message: str, provider: str = "unknown") -> None:
        super().__init__(message, code="LLM_ERROR")
        self.provider = provider


class SearchServiceError(DeepResearchError):
    """搜索服务异常。"""
    def __init__(self, message: str, provider: str = "unknown") -> None:
        super().__init__(message, code="SEARCH_ERROR")
        self.provider = provider


class CrawlerError(DeepResearchError):
    """网页抓取异常。"""
    def __init__(self, message: str, url: str = "") -> None:
        super().__init__(message, code="CRAWLER_ERROR")
        self.url = url


class VectorStoreError(DeepResearchError):
    """向量数据库异常。"""
    def __init__(self, message: str) -> None:
        super().__init__(message, code="VECTOR_STORE_ERROR")


class TaskNotFoundError(DeepResearchError):
    """任务不存在异常。"""
    def __init__(self, task_id: str) -> None:
        super().__init__(f"任务不存在: {task_id}", code="TASK_NOT_FOUND")
        self.task_id = task_id


class TaskAlreadyExistsError(DeepResearchError):
    """任务已存在异常。"""
    def __init__(self, task_id: str) -> None:
        super().__init__(f"任务已存在: {task_id}", code="TASK_ALREADY_EXISTS")
        self.task_id = task_id


class ConfigurationError(DeepResearchError):
    """配置错误。"""
    def __init__(self, message: str) -> None:
        super().__init__(message, code="CONFIG_ERROR")
