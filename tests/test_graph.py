"""LangGraph 工作流节点测试。"""

from __future__ import annotations

import pytest

from app.graph.nodes import planner_node


@pytest.mark.asyncio
async def test_planner_node_basic() -> None:
    """测试规划节点基本功能。"""
    state = {
        "task_id": "test_001",
        "query": "人工智能的最新进展",
        "language": "zh-CN",
        "max_rounds": 2,
        "current_round": 1,
        "status": "planning",
        "user_hints": {},
        "research_strategy": {},
        "research_plan": {},
        "sub_questions": [],
        "search_queries": [],
        "search_results": [],
        "crawled_documents": [],
        "evaluated_sources": [],
        "evidence_chunks": [],
        "sub_answers": [],
        "fact_check_result": {},
        "fact_check_passed": True,
        "follow_up_queries": [],
        "final_report": "",
        "errors": [],
        "progress": 0,
        "progress_message": "",
        "created_at": "",
        "updated_at": "",
    }

    result = await planner_node(state)

    assert "research_plan" in result
    assert "sub_questions" in result
    assert "search_queries" in result
    assert len(result["sub_questions"]) >= 2
    assert result["progress"] > 0



