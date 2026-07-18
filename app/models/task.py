"""任务数据模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ResearchTask(BaseModel):
    """研究任务实体。"""
    task_id: str = Field(default_factory=lambda: f"task_{uuid4().hex[:12]}")
    query: str
    status: str = "pending"  # pending | planning | retrieving | indexing | analyzing | fact_checking | writing | completed | failed
    max_rounds: int = 2
    current_round: int = 0
    language: str = "zh-CN"
    report_type: str = "deep"
    search_depth: str = "advanced"
    top_k: int = 6
    enable_fact_check: bool = True

    # 执行结果
    research_plan: dict[str, Any] | None = None
    final_report: str | None = None
    fact_check_result: dict[str, Any] | None = None
    error_message: str | None = None

    # 元数据
    progress: int = 0
    progress_message: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    # Token 统计
    total_tokens: int = 0
    llm_calls: int = 0

    def update_status(self, status: str, progress: int | None = None,
                      message: str | None = None) -> None:
        """更新任务状态。"""
        self.status = status
        self.updated_at = datetime.now().isoformat()
        if progress is not None:
            self.progress = progress
        if message is not None:
            self.progress_message = message

    def to_dict(self) -> dict[str, Any]:
        """转换为字典。"""
        return self.model_dump(exclude_none=True)


class ResearchTaskCreate(BaseModel):
    """创建研究任务的请求模型。"""
    query: str = Field(..., min_length=2, max_length=500,
                       description="研究问题")
    max_rounds: int = Field(default=2, ge=1, le=5,
                            description="最大研究轮数")
    language: str = Field(default="zh-CN", pattern="^(zh-CN|en)$",
                          description="输出语言")
    report_type: str = Field(default="deep",
                             pattern="^(summary|deep|comparison)$",
                             description="报告类型")
    search_depth: str = Field(default="advanced",
                              pattern="^(basic|advanced)$",
                              description="搜索深度")
    top_k: int = Field(default=6, ge=1, le=20,
                       description="RAG 检索片段数量")
    enable_fact_check: bool = Field(default=True,
                                    description="是否启用事实核查")


class TaskStatusResponse(BaseModel):
    """任务状态响应。"""
    task_id: str
    query: str = ""
    status: str
    progress: int = 0
    progress_message: str = ""
    current_round: int = 0
    max_rounds: int = 2


class TaskDetailResponse(BaseModel):
    """任务详情响应。"""
    task_id: str
    query: str
    status: str
    progress: int
    progress_message: str
    current_round: int
    max_rounds: int
    research_plan: dict[str, Any] | None = None
    search_results: list[dict[str, Any]] = []
    evaluated_sources: list[dict[str, Any]] = []
    # 完整产出（供 benchmark / 前端一次拉取，替代已废弃的 output/tasks/*.json）
    final_report: str | None = None
    fact_check_result: dict[str, Any] | None = None
    created_at: str
    updated_at: str
    error_message: str | None = None


class TaskReportResponse(BaseModel):
    """报告响应。"""
    task_id: str
    status: str
    report: str | None = None
    sources: list[str] = []
    error_message: str | None = None


class TaskProgressResponse(BaseModel):
    """研究过程实时快照 — 供前端做「过程透明」可视化，逐节点填充。

    冷热分层：这些中间产物量大且高频变化，只写 Redis 热层（见 task_service
    write_progress_detail），不落 SQLite；任务完成后前端改拉 /report 或 /detail。
    """
    task_id: str
    status: str
    progress: int = 0
    progress_message: str = ""
    current_round: int = 0
    max_rounds: int = 2
    # ── 研究过程中间数据（随节点推进逐步填充）──
    research_strategy: dict[str, Any] = {}          # {intent, domain, depth}
    sub_questions: list[dict[str, Any]] = []        # [{id, question, search_queries}]
    search_queries: list[str] = []
    search_summaries: list[dict[str, Any]] = []     # [{sq_id, answer}] Tavily 摘要
    sources: list[dict[str, Any]] = []              # evaluated_sources 元数据
    crawled_count: int = 0
    evidence_count: int = 0
    citation_registry: list[dict[str, Any]] = []    # [{id, title, url}]
    sub_answers: list[dict[str, Any]] = []          # [{sub_question_id, question, answer, citations, confidence, evidence_gap}]
    fact_check: dict[str, Any] = {}                 # {passed, issues:[{type,claim,reason}], follow_up_queries}
    fact_check_passed: bool = True
    follow_up_queries: list[str] = []
    rounds: list[dict[str, Any]] = []               # 已完成轮次摘要 [{round, fact_check_passed, issues_count, follow_up_queries}]
    # 完成时一并带上，前端可直接渲染报告
    final_report: str | None = None
    error_message: str | None = None
    updated_at: str = ""
