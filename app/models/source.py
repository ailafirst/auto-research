"""来源与证据数据模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """搜索结果条目。"""
    title: str
    url: str
    snippet: str
    source: str = ""
    position: int = 0
    sub_question_id: str | None = None
    retrieved_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    # Tavily 专属字段（DuckDuckGo 时均为 None）
    raw_content: str | None = None       # 完整页面正文，可跳过爬虫
    tavily_score: float | None = None    # 查询相关度 0-1
    query_answer: str | None = None      # Tavily 对该次查询的摘要答案
    # 检索来源元数据（P0：域驱动路由 + 可追溯性，见 docs/AnySearch技术原理与项目借鉴调研.md §6）
    provider: str = ""       # 实际服务本次查询的引擎："tavily" / "duckduckgo" / "crossref" / "arxiv"
    search_route: str = "general"  # 命中的检索路由："general"（通用）/ "academic"（学术白名单）/ "academic-structured"（CrossRef/arXiv 结构化）
    # 差距 #4：结构化子域参数契约的落地字段（目前只有 CrossRef/arXiv 落地会填，其余 provider
    # 为 None，见 docs/检索路由蓝图.md §9.15/§9.16）。不是通用元数据，是"这条结果来自结构化
    # 数据源"的标志。citation_count 只有 CrossRef 会填（arXiv 不追踪引用，恒为 None）；
    # venue 对 CrossRef 是期刊名，对 arXiv 借用存放 primary_category（如 "cs.LG"），
    # 两者语义不同，下游按 provider 字段区分对待，不能直接当同一种"期刊"展示。
    doi: str | None = None
    citation_count: int | None = None
    venue: str | None = None
    published_year: int | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.source:
            from urllib.parse import urlparse
            self.source = urlparse(self.url).netloc


class CrawledDocument(BaseModel):
    """已抓取的网页文档。"""
    url: str
    title: str
    content: str  # 清洗后的正文
    text_length: int = 0
    fetch_time: float = 0.0
    error: str | None = None
    published_at: str | None = None
    author: str | None = None
    tavily_score: float | None = None    # 从 SearchResult 透传，用于 source_evaluator

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.content and not self.text_length:
            self.text_length = len(self.content)


class SourceEvaluation(BaseModel):
    """信源评估结果。"""
    url: str
    title: str
    relevance_score: float = 0.0
    credibility_score: float = 0.0
    freshness_score: float = 0.0
    final_score: float = 0.0
    accepted: bool = False
    reason: str = ""


class EvidenceChunk(BaseModel):
    """证据切片（Chunk）。"""
    chunk_id: str = Field(default_factory=lambda: f"chunk_{id(object())}")
    task_id: str
    source_id: str
    url: str
    title: str
    chunk_index: int
    text: str
    vector_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class SubAnswer(BaseModel):
    """子问题分析结果。"""
    sub_question_id: str
    question: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    evidence_gap: bool = False


class FactCheckIssue(BaseModel):
    """事实核查问题。"""
    type: str  # insufficient_evidence, contradiction, overclaim, citation_mismatch
    claim: str
    reason: str


class FactCheckResult(BaseModel):
    """事实核查结果。"""
    passed: bool = True
    issues: list[FactCheckIssue] = Field(default_factory=list)
    follow_up_queries: list[str] = Field(default_factory=list)
