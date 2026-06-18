"""LangGraph 状态定义 — ResearchState。"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    """研究工作流的状态对象。"""

    # 任务信息
    task_id: str
    query: str
    language: str
    max_rounds: int
    current_round: int
    status: str

    # 研究计划
    research_plan: dict[str, Any]
    sub_questions: list[dict[str, Any]]
    search_queries: list[str]

    # 搜索与抓取
    search_results: list[dict[str, Any]]
    crawled_documents: list[dict[str, Any]]

    # 信源评估
    evaluated_sources: list[dict[str, Any]]

    # RAG 证据
    evidence_chunks: list[dict[str, Any]]

    # 分析结果
    sub_answers: list[dict[str, Any]]

    # 事实核查
    fact_check_result: dict[str, Any]
    fact_check_passed: bool
    follow_up_queries: list[str]

    # 报告
    final_report: str

    # 执行信息
    errors: Annotated[list[str], add_messages]
    progress: int
    progress_message: str
    created_at: str
    updated_at: str
