"""LangGraph 工作流构建器。

构建流程图:
START -> planner -> retriever -> content_extractor -> source_evaluator
        -> evidence_builder -> analyst -> fact_checker
        -> (证据不足 ? retriever : report_writer) -> END
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from app.graph.nodes import (
    analyst_node,
    content_extractor_node,
    evidence_builder_node,
    fact_checker_node,
    planner_node,
    report_writer_node,
    retriever_node,
    source_evaluator_node,
)
from app.graph.state import ResearchState

logger = logging.getLogger(__name__)


def build_research_graph() -> StateGraph:
    """构建 Deep Research 工作流图。"""
    workflow = StateGraph(ResearchState)

    # 注册节点
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("content_extractor", content_extractor_node)
    workflow.add_node("source_evaluator", source_evaluator_node)
    workflow.add_node("evidence_builder", evidence_builder_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("fact_checker", fact_checker_node)
    workflow.add_node("report_writer", report_writer_node)

    # 设置入口
    workflow.set_entry_point("planner")

    # 构建边
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "content_extractor")
    workflow.add_edge("content_extractor", "source_evaluator")
    workflow.add_edge("source_evaluator", "evidence_builder")
    workflow.add_edge("evidence_builder", "analyst")
    workflow.add_edge("analyst", "fact_checker")

    # 事实核查后直接生成报告（多轮研究由外部 _run_research 控制）
    workflow.add_edge("fact_checker", "report_writer")
    workflow.add_edge("report_writer", END)

    logger.info("LangGraph 研究流程已构建")

    return workflow.compile()
