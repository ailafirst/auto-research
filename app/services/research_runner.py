"""研究流程执行器 — 从 routes_research 抽出，供 API（进程内回退）与 arq worker 共用。

控制流（多轮补充研究、引用修正）仍在这里的 while 循环里，不在 LangGraph 图内。
"""

from __future__ import annotations

import asyncio
import logging

from app.graph.builder import build_research_graph
from app.services.task_service import TaskService

logger = logging.getLogger(__name__)

# 独立的 TaskService 实例：内部状态已全在 SQLite + Redis，多实例安全共享
task_service = TaskService()


def _progress_snapshot(state: dict) -> dict:
    """从 graph state 提取供前端「过程透明」展示的快照。

    裁掉全文类大字段（search_results.raw_content、crawled_documents 正文、
    evidence_chunks 全文），只留元数据与计数，避免每个节点往 Redis 写大 JSON。
    evaluated_sources / sub_answers 本身已是元数据/结论文本，直接透传。
    """
    return {
        "research_strategy": state.get("research_strategy") or {},
        "sub_questions": [
            {
                "id": sq.get("id"),
                "question": sq.get("question"),
                "search_queries": sq.get("search_queries", []),
            }
            for sq in state.get("sub_questions") or []
        ],
        "search_queries": state.get("search_queries") or [],
        "search_summaries": state.get("search_summaries") or [],
        "sources": state.get("evaluated_sources") or [],
        "crawled_count": len(state.get("crawled_documents") or []),
        "evidence_count": len(state.get("evidence_chunks") or []),
        "citation_registry": state.get("citation_registry") or [],
        "sub_answers": state.get("sub_answers") or [],
        "fact_check": state.get("fact_check_result") or {},
        "fact_check_passed": state.get("fact_check_passed", True),
        "follow_up_queries": state.get("follow_up_queries") or [],
        # 已完成轮次的摘要（由 run_research 维护并塞进 state），供前端画轮次时间线
        "rounds": state.get("rounds_history") or [],
    }


def _round_summary(round_num: int, state: dict) -> dict:
    """一轮研究结束时的摘要，用于前端轮次时间线（为何触发下一轮一目了然）。"""
    fc = state.get("fact_check_result") or {}
    return {
        "round": round_num,
        "fact_check_passed": state.get("fact_check_passed", True),
        "issues_count": len(fc.get("issues", []) if isinstance(fc, dict) else []),
        "follow_up_queries": state.get("follow_up_queries") or [],
    }


def _merge_by_url(base: list[dict], extra: list[dict]) -> list[dict]:
    """按 url 取并集合并两份文档/信源列表（base 优先，新 url 追加）。"""
    seen = {d.get("url") for d in base if d.get("url")}
    merged = list(base)
    for d in extra:
        u = d.get("url")
        if u and u not in seen:
            seen.add(u)
            merged.append(d)
    return merged


async def run_research(task_id: str) -> None:
    """在后台执行研究流程，使用流式获取实时进度。"""
    try:
        task = await task_service.get_task(task_id)

        research_graph = build_research_graph()

        NODE_LABELS = {
            "planner": "正在分析研究问题...",
            "retriever": "正在搜索相关资料...",
            "content_extractor": "正在抓取网页内容...",
            "source_evaluator": "正在评估信源质量...",
            "evidence_builder": "正在构建证据索引...",
            "analyst": "正在分析研究内容...",
            "fact_checker": "正在事实核查...",
            "report_writer": "正在生成研究报告...",
        }
        NODE_PROGRESS = {
            "planner": 10,
            "retriever": 25,
            "content_extractor": 40,
            "source_evaluator": 50,
            "evidence_builder": 60,
            "analyst": 70,
            "fact_checker": 80,
            "report_writer": 90,
        }

        def _build_state(round_num: int, extra: dict | None = None) -> dict:
            user_hints: dict = {}
            if task.report_type != "deep":
                user_hints["report_type"] = task.report_type
            if task.search_depth != "advanced":
                user_hints["search_depth"] = task.search_depth

            state = {
                "task_id": task_id,
                "query": task.query,
                "language": task.language,
                "max_rounds": task.max_rounds,
                "current_round": round_num,
                "status": "planning",
                "user_hints": user_hints,
                "research_strategy": {},
                "research_plan": {},
                "sub_questions": [],
                "search_queries": [],
                "search_results": [],
                "search_summaries": [],
                "crawled_documents": [],
                "evaluated_sources": [],
                "evidence_chunks": [],
                "sub_answers": [],
                "fact_check_result": {},
                "fact_check_passed": True,
                "follow_up_queries": [],
                "citation_mismatches": [],
                "analyst_revision_done": False,
                "final_report": "",
                "errors": [],
                "progress": 0,
                "progress_message": "",
                "created_at": task.created_at,
                "updated_at": task.updated_at,
            }
            if extra:
                state.update(extra)
            return state

        async def _run_round(current_state: dict) -> dict:
            final_state = current_state.copy()
            async for chunk in research_graph.astream(
                current_state, stream_mode="updates",
            ):
                for node_name, state_update in chunk.items():
                    final_state.update(state_update)

                    label = NODE_LABELS.get(node_name, f"正在执行 {node_name}...")
                    pct = state_update.get("progress") or NODE_PROGRESS.get(node_name, 0)
                    msg = state_update.get("progress_message") or label

                    await task_service.update_task(
                        task_id,
                        status=node_name,
                        progress=pct,
                        progress_message=msg,
                        current_round=final_state.get("current_round", 1),
                    )
                    # 写过程中间产物快照（Redis 热层），供前端逐节点透明可视化
                    await task_service.write_progress_detail(
                        task_id, _progress_snapshot(final_state),
                    )
            return final_state

        async def _revise_citations(state: dict) -> dict:
            if not state.get("citation_mismatches") or state.get("analyst_revision_done"):
                return state
            if not state.get("fact_check_passed", True):
                return state
            from app.graph.nodes import analyst_node, fact_checker_node, report_writer_node
            n = len(state["citation_mismatches"])
            await task_service.update_task(
                task_id, status="analyst", progress=72,
                progress_message=f"修正 {n} 处引用错误...",
            )
            revised = await analyst_node({**state, "analyst_revision_done": True})
            state = {**state, **revised, "analyst_revision_done": True}
            await task_service.write_progress_detail(task_id, _progress_snapshot(state))

            await task_service.update_task(
                task_id, status="fact_checker", progress=82,
                progress_message="验证引用修正结果...",
            )
            fc = await fact_checker_node(state)
            state = {**state, **fc}
            await task_service.write_progress_detail(task_id, _progress_snapshot(state))

            await task_service.update_task(
                task_id, status="report_writer", progress=88,
                progress_message="重新生成修订报告...",
            )
            rw = await report_writer_node(state)
            return {**state, **rw}

        async def _step(node_func, st: dict, status: str, progress: int, message: str) -> dict:
            """执行单个节点并推送进度/快照（增量补证中直接编排节点，不走整图）。"""
            await task_service.update_task(
                task_id, status=status, progress=progress,
                progress_message=message, current_round=st.get("current_round", 1),
            )
            out = await node_func(st)
            merged = {**st, **out}
            await task_service.write_progress_detail(task_id, _progress_snapshot(merged))
            return merged

        async def _supplement_round(
            base_state: dict, follow_up_by_sq: dict, failed_ids: list[str], round_num: int,
        ) -> dict:
            """增量补证（方案②）：只对核查未通过的子问题重做「检索→分析→核查」，
            其余子问题复用旧答案；Citation Registry 跨轮追加、CID 稳定；证据向量库按
            task_id 累积。相比原来的「整图重跑」，避免重复已通过的工作、不回退好答案。
            """
            from app.graph.nodes import (
                analyst_node,
                content_extractor_node,
                evidence_builder_node,
                fact_checker_node,
                retriever_node,
                source_evaluator_node,
            )

            all_sqs = base_state.get("sub_questions", [])
            failed_sqs: list[dict] = []
            for sq in all_sqs:
                if sq.get("id") in failed_ids:
                    fu = follow_up_by_sq.get(sq.get("id")) or sq.get("search_queries", [])
                    failed_sqs.append({**sq, "search_queries": fu})
            if not failed_sqs:
                return base_state

            flat_queries: list[str] = []
            for sq in failed_sqs:
                for q in sq.get("search_queries", []):
                    if q not in flat_queries:
                        flat_queries.append(q)

            sub_state = {
                **base_state,
                "current_round":      round_num,
                "sub_questions":      failed_sqs,
                "search_queries":     flat_queries,
                "search_results":     [],
                "search_summaries":   [],
                "crawled_documents":  [],
                "evaluated_sources":  [],
                "evidence_chunks":    [],
                "citation_mismatches": [],
                "analyst_revision_done": False,
                # 关键：带上已建立的 registry，analyst 在其上追加，CID 跨轮不重号
                "citation_registry":  base_state.get("citation_registry", []),
            }

            n = len(failed_sqs)
            sub_state = await _step(retriever_node, sub_state, "retriever", 30,
                                    f"第 {round_num} 轮定向补充检索（{n} 个待完善子问题）...")
            sub_state = await _step(content_extractor_node, sub_state, "content_extractor", 42,
                                    "抓取补充网页...")
            sub_state = await _step(source_evaluator_node, sub_state, "source_evaluator", 52,
                                    "评估补充信源...")
            sub_state = await _step(evidence_builder_node, sub_state, "evidence_builder", 62,
                                    "补充证据入库...")
            sub_state = await _step(analyst_node, sub_state, "analyst", 72,
                                    f"重新分析 {n} 个子问题...")
            sub_state = await _step(fact_checker_node, sub_state, "fact_checker", 82,
                                    "复核补充结论...")

            # 合并：失败子问题用新答案覆盖，其余复用旧答案（顺序与原 sub_questions 一致）
            new_answers = {a.get("sub_question_id"): a for a in sub_state.get("sub_answers", [])}
            merged_answers = [
                new_answers.get(a.get("sub_question_id"), a)
                for a in base_state.get("sub_answers", [])
            ]

            return {
                **base_state,
                "current_round":       round_num,
                "sub_questions":       all_sqs,
                "sub_answers":         merged_answers,
                # analyst 已在旧 registry 上追加，sub_state 的 registry 即并集
                "citation_registry":   sub_state.get("citation_registry", []),
                "crawled_documents":   _merge_by_url(base_state.get("crawled_documents", []),
                                                     sub_state.get("crawled_documents", [])),
                "evaluated_sources":   _merge_by_url(base_state.get("evaluated_sources", []),
                                                     sub_state.get("evaluated_sources", [])),
                # 核查结论取本轮（只覆盖曾失败的子问题）：全通过则收敛，否则携带新的 failed 集
                "fact_check_result":   sub_state.get("fact_check_result", {}),
                "fact_check_passed":   sub_state.get("fact_check_passed", True),
                "follow_up_queries":   sub_state.get("follow_up_queries", []),
                "follow_up_by_sq":     sub_state.get("follow_up_by_sq", {}),
                "failed_sub_questions": sub_state.get("failed_sub_questions", []),
                # 补证轮不再触发引用修正整图重析，交由 analyst 自身的引用纪律兜底
                "citation_mismatches": [],
                "analyst_revision_done": True,
            }

        # 轮次历史（供前端轮次时间线，避免多轮时步骤条「清零」的误解）
        rounds_history: list[dict] = []

        # 第一轮研究
        initial_state = _build_state(1)
        final_state = await _run_round(initial_state)
        final_state = await _revise_citations(final_state)
        rounds_history.append(_round_summary(1, final_state))
        final_state["rounds_history"] = rounds_history
        await task_service.write_progress_detail(task_id, _progress_snapshot(final_state))

        # 多轮增量补证（方案②）：只重做核查未通过的子问题，通过的复用；
        # failed 集为空即收敛停止（置信停止），封顶 max_rounds。
        current_round = 1
        while current_round < task.max_rounds:
            if final_state.get("fact_check_passed", True):
                break

            failed_ids = final_state.get("failed_sub_questions", [])
            follow_up_by_sq = final_state.get("follow_up_by_sq", {})
            if not failed_ids:
                break

            current_round += 1
            await task_service.update_task(
                task_id,
                status="retrieving",
                current_round=current_round,
                progress_message=f"第 {current_round} 轮增量补证（{len(failed_ids)} 个子问题）...",
            )

            final_state["rounds_history"] = rounds_history   # 供本轮快照展示已完成轮次
            final_state = await _supplement_round(
                final_state, follow_up_by_sq, failed_ids, current_round,
            )
            rounds_history.append(_round_summary(current_round, final_state))
            final_state["rounds_history"] = rounds_history
            await task_service.write_progress_detail(task_id, _progress_snapshot(final_state))

        # 若发生过增量补证，基于合并后的完整子答案重写报告（首轮报告已在图内生成）
        if current_round > 1:
            from app.graph.nodes import report_writer_node
            await task_service.update_task(
                task_id, status="report_writer", progress=92,
                progress_message="综合各轮结论重写报告...",
            )
            rw = await report_writer_node(final_state)
            final_state = {**final_state, **rw}
            final_state["rounds_history"] = rounds_history
            await task_service.write_progress_detail(task_id, _progress_snapshot(final_state))

        report = final_state.get("final_report", "")
        await task_service.update_task(
            task_id,
            status="completed",
            progress=100,
            progress_message="研究完成",
            final_report=report,
            fact_check_result=final_state.get("fact_check_result") or {},
        )

        logger.info("研究任务完成: task_id=%s, 轮数=%d", task_id, current_round)

    except asyncio.CancelledError:
        logger.warning("研究任务被取消: %s", task_id)
        await _fail_task(task_id, "任务被取消")
    except Exception as exc:
        logger.error("研究任务失败: task_id=%s, error=%s", task_id, exc)
        await _fail_task(task_id, str(exc))


async def _fail_task(task_id: str, error_message: str) -> None:
    """标记任务失败。"""
    try:
        await task_service.update_task(
            task_id,
            status="failed",
            progress=0,
            progress_message="研究失败",
            error_message=error_message,
        )
    except Exception as exc:
        logger.error("更新任务失败状态出错: %s", exc)
