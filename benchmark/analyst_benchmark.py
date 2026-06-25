"""Analyst / Fact Checker / Report Writer 基准测试

沿用 planner_benchmark / retriever_benchmark 的同一模式：
直接调用节点函数，所有 LLM 调用和搜索 API 均为真实调用，无任何 mock。

覆盖完整 8 节点流水线：
  planner → retriever → content_extractor → source_evaluator →
  evidence_builder → analyst → fact_checker → report_writer

专项度量 analyst 三项改进（#2/#3/#4）的真实输出：
  #2 Analyst 策略感知 ——  答案均长、深度符合率、置信度均值
  #3 Fact Checker 并发 —— issues 数、follow_up 数
  #4 Report Writer 注入 — 报告结构符合 report_type 规范

Trace 层次（本版新增）：
  A 输出侧    — report_len < 200 时立即打印完整内容 + 上下文摘要
  B LLM 调用  — monkey-patch litellm.acompletion，记录 prompt/completion tokens、耗时、输出长度
  C State 快照 — 每节点完成后记录关键字段大小，可视化数据流

用法：
  python benchmark/analyst_benchmark.py              # 全部 12 个任务
  python benchmark/analyst_benchmark.py 01 03 07    # 指定任务
  python benchmark/analyst_benchmark.py -c 3        # 并发数（默认 4）
"""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import io
import json
import sys

# Windows GBK 终端无法编码 ✓ 等 Unicode 字符，强制 UTF-8 输出
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

TASKS_FILE  = Path(__file__).parent / "tasks.json"
RESULTS_DIR = Path(__file__).parent / "results"

GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RED    = "\033[31m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def _g(t: str) -> str: return f"{GREEN}{t}{RESET}"
def _y(t: str) -> str: return f"{YELLOW}{t}{RESET}"
def _c(t: str) -> str: return f"{CYAN}{t}{RESET}"
def _r(t: str) -> str: return f"{RED}{t}{RESET}"
def _b(t: str) -> str: return f"{BOLD}{t}{RESET}"
def _d(t: str) -> str: return f"{DIM}{t}{RESET}"

# ── Trace B：每任务 LLM 调用日志（contextvar 标记当前 task_id）─────────────

_current_task_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "bench_task_id", default="unknown"
)
_llm_call_log: dict[str, list[dict]] = {}

# 深度规范答案长度区间
_DEPTH_RANGE: dict[str, tuple[int, int]] = {
    "shallow": (150, 300),
    "medium":  (300, 500),
    "deep":    (500, 800),
}


# ── Trace B：Monkey-patch litellm.acompletion ────────────────────────────────

def _patch_llm_tracing() -> None:
    """在 main() 开头调用一次；之后所有 LLM 调用自动记录 token 用量与输出内容。"""
    import litellm as _litellm  # noqa: PLC0415

    _orig = _litellm.acompletion

    async def _traced(**kwargs: object) -> object:  # type: ignore[misc]
        task_id = _current_task_id.get("unknown")
        # 尝试开启 MiMo 思维链输出（若端点不支持会被忽略）
        if "enable_thinking" not in kwargs:
            kwargs = {**kwargs, "enable_thinking": True}  # type: ignore[assignment]
        t0      = time.perf_counter()
        response = None
        exc_info: str | None = None
        try:
            response = await _orig(**kwargs)  # type: ignore[operator]
            return response
        except Exception as exc:
            exc_info = str(exc)[:200]
            raise
        finally:
            elapsed = round(time.perf_counter() - t0, 2)
            record: dict = {"elapsed": elapsed, "node": "?"}
            if response is not None and response.choices:
                msg     = response.choices[0].message
                content = msg.content or ""
                thinking = getattr(msg, "reasoning_content", None) or ""
                usage   = getattr(response, "usage", None)
                record.update({
                    "prompt_tokens":     getattr(usage, "prompt_tokens",     -1) if usage else -1,
                    "completion_tokens": getattr(usage, "completion_tokens", -1) if usage else -1,
                    "output_len":        len(content),
                    "output_preview":    content[:150] if len(content) < 300 else "",
                    "thinking_len":      len(thinking),
                    "thinking_preview":  thinking[:200] if thinking else "",
                })
            if exc_info:
                record["error"] = exc_info
            _llm_call_log.setdefault(task_id, []).append(record)

    _litellm.acompletion = _traced  # type: ignore[attr-defined]


# ── Trace C：节点后 State 快照 ───────────────────────────────────────────────

def _capture_state_snapshot(state: dict, after_node: str) -> dict:
    """每个节点完成后记录关键字段大小，用于数据流追踪与 report_writer prompt 估算。"""
    snap: dict = {}

    if after_node == "planner":
        snap["sub_q"]          = len(state.get("sub_questions", []))
        snap["search_queries"] = len(state.get("search_queries", []))

    elif after_node == "retriever":
        snap["search_results"] = len(state.get("search_results", []))

    elif after_node == "content_extractor":
        docs = state.get("crawled_documents", [])
        snap["docs"]          = len(docs)
        snap["content_chars"] = sum(len(d.get("content") or "") for d in docs)

    elif after_node == "source_evaluator":
        ev            = state.get("evaluated_sources", [])
        accepted      = [e for e in ev if e.get("accepted")]
        accepted_urls = {e["url"] for e in accepted}
        snap["evaluated"]      = len(ev)
        snap["accepted"]       = len(accepted)
        snap["accepted_chars"] = sum(
            len(d.get("content") or "")
            for d in state.get("crawled_documents", [])
            if d.get("url") in accepted_urls
        )

    elif after_node == "evidence_builder":
        chunks              = state.get("evidence_chunks", [])
        snap["chunks"]      = len(chunks)
        snap["chunk_chars"] = sum(len(c.get("text") or "") for c in chunks)

    elif after_node == "analyst":
        sas                        = state.get("sub_answers", [])
        lens                       = [len(sa.get("answer") or "") for sa in sas]
        snap["answers"]            = len(sas)
        snap["per_answer_chars"]   = lens
        snap["total_answer_chars"] = sum(lens)

    elif after_node == "fact_checker":
        fc     = state.get("fact_check_result", {})
        issues = fc.get("issues", [])
        issue_chars = sum(
            len(str(i.get("claim") or "")) + len(str(i.get("reason") or ""))
            for i in issues
        )
        sas       = state.get("sub_answers", [])
        sub_chars = sum(len(sa.get("answer") or "") for sa in sas)
        src_chars = sum(
            len(d.get("title") or "") + len(d.get("url") or "")
            for d in state.get("crawled_documents", [])
        )
        snap["issues"]              = len(issues)
        snap["issue_chars"]         = issue_chars
        snap["est_rw_prompt_chars"] = 460 + 200 + sub_chars + issue_chars + src_chars + 30

    elif after_node == "report_writer":
        report             = state.get("final_report", "")
        snap["report_len"] = len(report)
        if len(report) < 200:
            snap["SHORT"]        = True
            snap["full_content"] = report

    return snap


# ── Trace A：短输出立即告警 ──────────────────────────────────────────────────

def _short_output_alert(
    task_id: str,
    report: str,
    state_trace: dict,
    llm_calls: list[dict],
) -> None:
    """report_writer 输出 < 200 chars 时立即打印完整诊断，不等汇总。"""
    print(f"\n  {_r('⚠⚠⚠  [' + task_id + '] report_writer 输出异常短  ⚠⚠⚠')}")
    print(f"  输出内容 ({len(report)} chars): {repr(report)}")

    an_st = state_trace.get("analyst",      {})
    fc_st = state_trace.get("fact_checker", {})
    est   = fc_st.get("est_rw_prompt_chars")
    est_s = f"{est:,}" if isinstance(est, int) else "?"
    print(
        f"  sub_answers 总字符={an_st.get('total_answer_chars', '?')}  "
        f"issues={fc_st.get('issues', '?')}  est_prompt={est_s} chars"
    )
    rw_calls = [c for c in llm_calls if c.get("node") == "report_writer"]
    if rw_calls:
        print(f"  report_writer LLM 调用 ({len(rw_calls)} 次):")
        for i, c in enumerate(rw_calls, 1):
            pt  = c.get("prompt_tokens",     -1)
            ct  = c.get("completion_tokens", -1)
            el  = c.get("elapsed",            0)
            err = c.get("error",             "")
            pt_s = f"{pt:,}" if pt >= 0 else "?"
            ct_s = f"{ct:,}" if ct >= 0 else "?"
            print(
                f"    #{i}: prompt_tokens={pt_s}  completion_tokens={ct_s}  elapsed={el}s"
                + (f"  {_r('ERR: ' + err[:60])}" if err else "")
            )
    else:
        print(f"  {_y('未找到 report_writer LLM 调用记录（可能节点外部失败）')}")


# ── 状态构建 ──────────────────────────────────────────────────────────────────

def _build_state(task: dict) -> dict:
    now = datetime.now().isoformat()
    user_hints: dict = {}
    if task.get("report_type", "deep") != "deep":
        user_hints["report_type"] = task["report_type"]
    return {
        "task_id":           f"bench_{task['id']}",
        "query":             task["query"],
        "language":          task.get("language", "zh-CN"),
        "max_rounds":        1,
        "current_round":     1,
        "status":            "planning",
        "user_hints":        user_hints,
        "research_strategy": {},
        "research_plan":     {},
        "sub_questions":     [],
        "search_queries":    [],
        "search_results":    [],
        "search_summaries":  [],
        "crawled_documents": [],
        "evaluated_sources": [],
        "evidence_chunks":   [],
        "sub_answers":       [],
        "fact_check_result": {},
        "fact_check_passed": True,
        "follow_up_queries": [],
        "final_report":      "",
        "errors":            [],
        "progress":          0,
        "progress_message":  "",
        "created_at":        now,
        "updated_at":        now,
    }


# ── 单任务执行（全 8 节点，真实 API）─────────────────────────────────────────

async def run_task(task: dict) -> dict:
    from app.graph.nodes import (
        planner_node, retriever_node, content_extractor_node,
        source_evaluator_node, evidence_builder_node,
        analyst_node, fact_checker_node, report_writer_node,
    )

    task_id = task["id"]
    _llm_call_log[task_id] = []    # 初始化本任务 LLM 调用日志（Trace B）
    _current_task_id.set(task_id)  # 设置 contextvar，LLM tracer 通过此值归属调用

    base = {
        "benchmark_task_id":   task_id,
        "benchmark_task_name": task["name"],
        "query":               task["query"],
        "report_type":         task.get("report_type", "deep"),
        "run_at":              datetime.now().isoformat(),
    }
    state       = _build_state(task)
    timings:     dict[str, float] = {}
    state_trace: dict[str, dict]  = {}  # Trace C 快照容器

    def _early_return() -> dict:
        return {
            **base,
            "timings":     timings,
            "state_trace": state_trace,
            "llm_calls":   _llm_call_log.pop(task_id, []),
        }

    async def _step(name: str, node_fn) -> bool:
        call_idx_before = len(_llm_call_log.get(task_id, []))
        t0 = time.perf_counter()
        try:
            state.update(await node_fn(state))
        except Exception as exc:
            base["error"] = f"{name}: {exc}"
            return False
        timings[name] = round(time.perf_counter() - t0, 2)
        # 为本节点产生的 LLM 调用打标签（Trace B）
        for call in _llm_call_log.get(task_id, [])[call_idx_before:]:
            call["node"] = name
        # 记录节点后 state 快照（Trace C）
        state_trace[name] = _capture_state_snapshot(state, name)
        return True

    # ── 1–5：前置节点 ────────────────────────────────────────────────────────
    for name, fn in [
        ("planner",           planner_node),
        ("retriever",         retriever_node),
        ("content_extractor", content_extractor_node),
        ("source_evaluator",  source_evaluator_node),
        ("evidence_builder",  evidence_builder_node),
    ]:
        if not await _step(name, fn):
            return _early_return()

    strategy = state.get("research_strategy", {})
    depth    = strategy.get("depth", "medium")
    intent   = strategy.get("intent", "—")
    accepted = [e for e in state.get("evaluated_sources", []) if e.get("accepted")]

    # ── 6. Analyst ──────────────────────────────────────────────────────────
    if not await _step("analyst", analyst_node):
        return {**_early_return(), "strategy": strategy}

    sub_answers  = state.get("sub_answers", [])
    answer_lens  = [len(sa.get("answer", "")) for sa in sub_answers]
    confidences  = [sa.get("confidence", 0.0) for sa in sub_answers]
    gap_count    = sum(1 for sa in sub_answers if sa.get("evidence_gap"))
    lo, hi       = _DEPTH_RANGE.get(depth, (0, 9999))
    in_range_cnt = sum(1 for length in answer_lens if lo <= length <= hi)

    # ── 7. Fact Checker ─────────────────────────────────────────────────────
    if not await _step("fact_checker", fact_checker_node):
        return {**_early_return(), "strategy": strategy}

    fc        = state.get("fact_check_result", {})
    issues    = fc.get("issues", [])
    follow_up = fc.get("follow_up_queries", [])

    # ── 8. Report Writer ────────────────────────────────────────────────────
    if not await _step("report_writer", report_writer_node):
        return {**_early_return(), "strategy": strategy}

    report     = state.get("final_report", "")
    report_len = len(report)

    # Trace A：短输出立即告警（仅 report_writer，fact_checker 的短 JSON 是正常结果）
    llm_calls_snap = list(_llm_call_log.get(task_id, []))
    if report_len < 200:
        _short_output_alert(task_id, report, state_trace, llm_calls_snap)

    section_cnt = report.count("\n##")
    has_table   = "|" in report
    report_type = task.get("report_type", "deep")
    if report_type == "summary":
        fmt_ok = report_len < 3000
    elif report_type == "comparison":
        fmt_ok = has_table
    else:
        fmt_ok = section_cnt >= 3

    return {
        **base,
        "strategy": {"intent": intent, "depth": depth},
        "pipeline": {
            "sub_questions":  len(state.get("sub_questions", [])),
            "search_results": len(state.get("search_results", [])),
            "accepted_docs":  len(accepted),
            "chunks":         len(state.get("evidence_chunks", [])),
        },
        "analyst": {
            "n_answers":      len(sub_answers),
            "avg_len":        round(sum(answer_lens) / max(len(answer_lens), 1)),
            "depth_range":    f"{lo}–{hi}",
            "in_range":       in_range_cnt,
            "avg_confidence": round(sum(confidences) / max(len(confidences), 1), 3),
            "evidence_gaps":  gap_count,
        },
        "fact_checker": {
            "passed":      fc.get("passed", True),
            "n_issues":    len(issues),
            "n_follow_up": len(follow_up),
            "issues":      issues,   # 完整列表，用于统计 type 分布
        },
        "report_writer": {
            "report_len":  report_len,
            "section_cnt": section_cnt,
            "has_table":   has_table,
            "fmt_ok":      fmt_ok,
        },
        "timings":         timings,
        "elapsed_seconds": round(sum(timings.values()), 2),
        "state_trace":     state_trace,
        "llm_calls":       _llm_call_log.pop(task_id, []),
    }


# ── 单任务详情输出 ─────────────────────────────────────────────────────────────

def _print_task(task: dict, r: dict) -> None:
    header = f"[{task['id']}] {task['name']}"
    print(f"\n{'═' * 72}")
    print(f"  {_b(header)}")
    print(f"  {_d(task['query'])}")
    print(f"{'─' * 72}")

    if r.get("error"):
        print(f"  {_r('ERROR:')} {r['error']}")
        # 即使出错，仍打印已有 trace
        _print_state_trace(r)
        _print_llm_call_detail(r)
        return

    s   = r["strategy"]
    pip = r["pipeline"]
    an  = r["analyst"]
    fc  = r["fact_checker"]
    rw  = r["report_writer"]
    tm  = r["timings"]
    dc  = {"shallow": _g, "medium": _y, "deep": _c}.get(s["depth"], _b)

    print(f"  {dc(s['depth'])} · {s['intent']} · report_type={r['report_type']}\n")

    print(f"  {'节点':<20}  {'耗时':>6}  {'输出'}")
    print(f"  {'─' * 60}")
    print(f"  {'planner':<20}  {tm.get('planner', 0):>5.1f}s  {pip['sub_questions']} 个子问题")
    print(f"  {'retriever':<20}  {tm.get('retriever', 0):>5.1f}s  {pip['search_results']} 条结果")
    print(f"  {'content_extractor':<20}  {tm.get('content_extractor', 0):>5.1f}s")
    print(f"  {'source_evaluator':<20}  {tm.get('source_evaluator', 0):>5.1f}s  通过 {pip['accepted_docs']} 篇")
    print(f"  {'evidence_builder':<20}  {tm.get('evidence_builder', 0):>5.1f}s  {pip['chunks']} chunks")
    print()

    in_r_s  = _g(f"{an['in_range']}/{an['n_answers']}") if an["in_range"] == an["n_answers"] else _y(f"{an['in_range']}/{an['n_answers']}")
    gap_s   = f"  {_r(str(an['evidence_gaps']) + ' gap')}" if an["evidence_gaps"] else ""
    fc_mark = _g("通过") if fc["passed"] else _r("未通过")
    rw_len_s = _r(str(rw["report_len"])) if rw["report_len"] < 200 else str(rw["report_len"])
    fmt_mark = _g("✓") if rw["fmt_ok"] else _r("✗")

    print(f"  {'analyst':<20}  {tm.get('analyst', 0):>5.1f}s  "
          f"均长={an['avg_len']}（期望{an['depth_range']}）  "
          f"符合={in_r_s}  置信={an['avg_confidence']:.0%}{gap_s}")
    print(f"  {'fact_checker':<20}  {tm.get('fact_checker', 0):>5.1f}s  "
          f"{fc_mark}  issues={fc['n_issues']}  follow_up={fc['n_follow_up']}")
    print(f"  {'report_writer':<20}  {tm.get('report_writer', 0):>5.1f}s  "
          f"{rw_len_s} 字符  {rw['section_cnt']} 节  "
          f"{'含表格  ' if rw['has_table'] else ''}格式={fmt_mark}")
    print(f"\n  总耗时 {_b(str(r['elapsed_seconds']) + 's')}")

    _print_state_trace(r)
    _print_llm_call_detail(r)


def _print_state_trace(r: dict) -> None:
    """Trace C：打印数据流快照。"""
    st = r.get("state_trace")
    if not st:
        return

    print(f"\n  {_d('── 数据流 Trace (C) ──────────────────────────────────────────')}")

    p = st.get("planner", {})
    if p:
        print(f"  {'planner':<20}  {p.get('sub_q', '?')} 子问题  {p.get('search_queries', '?')} 搜索词")

    rv = st.get("retriever", {})
    if rv:
        print(f"  {'retriever':<20}  {rv.get('search_results', '?')} 条搜索结果")

    ce = st.get("content_extractor", {})
    if ce:
        print(f"  {'content_extractor':<20}  {ce.get('docs', '?')} 篇文档  "
              f"总内容 {ce.get('content_chars', 0):,} chars")

    se = st.get("source_evaluator", {})
    if se:
        print(f"  {'source_evaluator':<20}  通过 {se.get('accepted', '?')}/{se.get('evaluated', '?')} 篇  "
              f"有效内容 {se.get('accepted_chars', 0):,} chars")

    eb = st.get("evidence_builder", {})
    if eb:
        print(f"  {'evidence_builder':<20}  {eb.get('chunks', '?')} chunks  "
              f"{eb.get('chunk_chars', 0):,} chars")

    an_st = st.get("analyst", {})
    if an_st:
        per   = an_st.get("per_answer_chars", [])
        per_s = " / ".join(str(c) for c in per) if per else "?"
        total = an_st.get("total_answer_chars", 0)
        print(f"  {'analyst':<20}  {an_st.get('answers', '?')} 答案  "
              f"总 {total:,} chars  [每题: {per_s}]")

    fc_st = st.get("fact_checker", {})
    if fc_st:
        est   = fc_st.get("est_rw_prompt_chars")
        est_s = f"{est:,}" if isinstance(est, int) else "?"
        print(f"  {'fact_checker':<20}  {fc_st.get('issues', '?')} issues  "
              f"{fc_st.get('issue_chars', 0):,} chars")
        print(f"  {'→ rw prompt 预估':<20}  {est_s} chars")

    rw_st = st.get("report_writer", {})
    if rw_st.get("SHORT"):
        print(f"  {_r('report_writer')}        {_r('SHORT OUTPUT:')} {repr(rw_st.get('full_content', ''))}")


def _print_llm_call_detail(r: dict) -> None:
    """Trace B：打印本任务的 LLM 调用明细。"""
    calls = r.get("llm_calls")
    if not calls:
        return

    print(f"\n  {_d('── LLM 调用 Trace (B)  ' + str(len(calls)) + ' 次调用 ──────────────────────────────')}")
    print(f"  {'节点':<20}  {'输入tok':>9}  {'输出tok':>9}  {'耗时':>6}  {'输出长':>7}  {'思维链':>7}  备注")

    for call in calls:
        nd  = call.get("node",              "?")
        pt  = call.get("prompt_tokens",     -1)
        ct  = call.get("completion_tokens", -1)
        el  = call.get("elapsed",            0)
        ol  = call.get("output_len",        -1)
        tl  = call.get("thinking_len",      -1)
        err = call.get("error",             "")
        prv = call.get("output_preview",    "")
        tpv = call.get("thinking_preview",  "")

        pt_s = f"{pt:,}" if pt >= 0 else "?"
        ct_s = f"{ct:,}" if ct >= 0 else "?"
        ol_s = f"{ol:,}" if ol >= 0 else "?"
        tl_s = _y(f"{tl:,}") if tl > 0 else (_d("0") if tl == 0 else "?")

        # fact_checker 合法短 JSON（passed=true）不触发 SHORT 告警
        is_fc_valid_short = (nd == "fact_checker" and ol < 200
                             and '"passed"' in prv)
        flag = ""
        if err:
            flag = _r(f" ERR: {err[:50]}")
        elif ol >= 0 and ol < 200 and not is_fc_valid_short:
            flag = _r(f" ⚠SHORT! content={repr(prv)}")
        elif is_fc_valid_short:
            flag = _d(f" (passed/no-issues JSON)")
        elif tl > 0 and tpv:
            flag = _d(f" think={repr(tpv[:60])}")

        print(f"  {nd:<20}  {pt_s:>9}  {ct_s:>9}  {el:>5.1f}s  {ol_s:>7}  {tl_s:>7}{flag}")


# ── 汇总表 ────────────────────────────────────────────────────────────────────

def _print_summary(results: list[dict], all_tasks: list[dict]) -> None:
    task_map = {t["id"]: t for t in all_tasks}
    print(f"\n\n{'═' * 72}")
    print(f"  {_b('Analyst 基准测试汇总')}")
    print(f"{'═' * 72}")
    print(f"  {'ID':>4}  {'名称':<18}  {'深度':<8}  {'均长':>6}  {'置信':>6}  "
          f"{'符合':>6}  {'issues':>7}  {'格式':>5}  {'耗时':>7}")
    print(f"  {'─' * 70}")

    for r in results:
        tid  = r["benchmark_task_id"]
        name = (task_map.get(tid, {}).get("name", ""))[:18]

        if r.get("error"):
            print(f"  {tid:>4}  {name:<18}  {_r('ERROR: ' + r['error'][:40])}")
            continue

        s  = r["strategy"]
        an = r["analyst"]
        fc = r["fact_checker"]
        rw = r["report_writer"]
        dc     = {"shallow": _g, "medium": _y, "deep": _c}.get(s["depth"], str)
        in_r_c = _g if an["in_range"] == an["n_answers"] else _y
        iss_c  = _r if fc["n_issues"] > 3 else _y if fc["n_issues"] > 0 else _g
        fmt_s  = _g("✓") if rw["fmt_ok"] else _r("✗")
        len_s  = _r(str(rw["report_len"])) if rw["report_len"] < 200 else str(rw["report_len"])

        print(
            f"  {tid:>4}  {name:<18}  {dc(s['depth']):<8}  "
            f"{an['avg_len']:>6}  "
            f"{an['avg_confidence']:>6.0%}  "
            f"{in_r_c(str(an['in_range']) + '/' + str(an['n_answers'])):>6}  "
            f"{iss_c(str(fc['n_issues'])):>7}  "
            f"{fmt_s:>5}  "
            f"{r['elapsed_seconds']:>6.1f}s  "
            f"rw={len_s}chars"
        )

    errors = sum(1 for r in results if r.get("error"))
    print(f"  {'─' * 70}")
    print(f"\n  共 {len(results)} 个任务" + (f"，{_y(str(errors) + ' 个错误')}" if errors else ""))
    print(f"\n  列说明: 均长=答案均字符数  符合=深度范围内答案数  格式=report_type 结构检查  rw=报告字符数")
    print(f"  深度图例: {_g('shallow')}  {_y('medium')}  {_c('deep')}")


# ── Fact Checker Issue 分布统计 ───────────────────────────────────────────────

def _print_issue_distribution(results: list[dict]) -> None:
    """统计所有任务的 fact_checker issue 类型分布。"""
    from collections import Counter

    all_issues: list[dict] = []
    per_task: list[tuple[str, str, list[dict]]] = []

    for r in results:
        if r.get("error"):
            continue
        fc     = r.get("fact_checker", {})
        issues = fc.get("issues", [])
        all_issues.extend(issues)
        per_task.append((r["benchmark_task_id"], r["benchmark_task_name"], issues))

    print(f"\n\n{'═' * 72}")
    print(f"  {_b('Fact Checker Issue 类型分布')}")
    print(f"{'═' * 72}")

    if not all_issues:
        print(f"  {_g('所有任务无 issue（或 issues 字段未记录）')}")
        return

    dist  = Counter(i.get("type", "unknown") for i in all_issues)
    total = len(all_issues)

    print(f"  共 {total} 条 issue，来自 {len(per_task)} 个任务\n")
    print(f"  {'类型':<30}  {'数量':>6}  {'占比':>6}  {'触发补搜'}")
    print(f"  {'─' * 58}")

    # Must match fact_checker_node._SERIOUS_ISSUE_TYPES
    RETRY_TYPES = {"contradiction", "overclaim", "insufficient_evidence"}
    for itype, cnt in dist.most_common():
        will_retry = _r("是") if itype in RETRY_TYPES else _g("否")
        print(f"  {itype:<30}  {cnt:>6}  {cnt * 100 // total:>5}%  {will_retry}")

    # 每任务明细
    print(f"\n  {'─' * 58}")
    print(f"  {'ID':<4}  {'名称':<20}  {'total':>5}  类型分布")
    print(f"  {'─' * 58}")
    for tid, name, issues in per_task:
        if not issues:
            print(f"  {tid:<4}  {name:<20}  {_g('0'):>5}")
            continue
        tdist = Counter(i.get("type", "?") for i in issues)
        dist_str = "  ".join(f"{k}×{v}" for k, v in tdist.most_common())
        has_retry = any(t in RETRY_TYPES for t in tdist)
        n_s = _r(str(len(issues))) if has_retry else _y(str(len(issues)))
        print(f"  {tid:<4}  {name:<20}  {n_s:>5}  {dist_str}")

    retry_triggered = sum(
        1 for _, _, issues in per_task
        if any(i.get("type") in RETRY_TYPES for i in issues)
    )
    only_minor = sum(
        1 for _, _, issues in per_task
        if issues and not any(i.get("type") in RETRY_TYPES for i in issues)
    )
    print(f"\n  触发补搜（overclaim/contradiction/insufficient_evidence）：{_r(str(retry_triggered))} 个任务")
    print(f"  仅含 citation_mismatch（analyst 引用修正，无需补搜）：{_y(str(only_minor))} 个任务")


# ── 节点耗时汇总分析 ───────────────────────────────────────────────────────────

_NODE_META: list[tuple[str, str]] = [
    ("planner",           "LLM   "),
    ("retriever",         "Search"),
    ("content_extractor", "IO    "),
    ("source_evaluator",  "Rule  "),
    ("evidence_builder",  "Embed "),
    ("analyst",           "LLM   "),
    ("fact_checker",      "LLM   "),
    ("report_writer",     "LLM   "),
]


def _print_timing_analysis(results: list[dict], wall_elapsed: float, concurrency: int) -> None:
    """打印各节点耗时统计，分析 LLM / 非 LLM 时间占比与并发效率。"""
    ok = [r for r in results if not r.get("error") and r.get("timings")]
    if not ok:
        return

    node_times: dict[str, list[float]] = {n: [] for n, _ in _NODE_META}
    for r in ok:
        for node, _ in _NODE_META:
            v = r["timings"].get(node)
            if v is not None:
                node_times[node].append(v)

    serial_sums = [sum(r["timings"].values()) for r in ok]
    avg_serial  = sum(serial_sums) / len(serial_sums)
    total_work  = sum(serial_sums)

    print(f"\n\n{'═' * 72}")
    print(
        f"  {_b('节点耗时分析')}  "
        f"{_d(f'成功 {len(ok)} 任务 · 并发={concurrency} · 挂钟={wall_elapsed:.1f}s')}"
    )
    print(f"{'═' * 72}")
    print(f"  {'节点':<22}  {'类型':<6}  {'均值':>6}  {'最小':>5}  {'最大':>6}  {'串行占%':>7}")
    print(f"  {'─' * 62}")

    llm_avg = 0.0
    for node, ntype in _NODE_META:
        times = node_times[node]
        if not times:
            print(f"  {_d(f'{node:<22}')}  {ntype:<6}  {'—':>6}")
            continue
        avg = sum(times) / len(times)
        mn  = min(times)
        mx  = max(times)
        pct = avg / avg_serial * 100 if avg_serial > 0 else 0

        if "LLM" in ntype:
            llm_avg += avg
            row = _c(f"{node:<22}")
        elif "Embed" in ntype:
            row = _y(f"{node:<22}")
        elif "Search" in ntype:
            row = f"{node:<22}"
        else:
            row = _d(f"{node:<22}")

        print(f"  {row}  {ntype:<6}  {avg:>5.1f}s  {mn:>4.1f}s  {mx:>5.1f}s  {pct:>6.1f}%")

    print(f"  {'─' * 62}")
    llm_pct = llm_avg / avg_serial * 100 if avg_serial > 0 else 0
    non_llm = avg_serial - llm_avg

    print(f"  {'串行总时均值':<30}  {avg_serial:>5.1f}s  (100%)")
    print(f"  {_c('LLM 节点合计')} (planner+analyst+fc+writer)   "
          f"{_c(f'{llm_avg:.1f}s')}  {_c(f'({llm_pct:.0f}%)')}")
    print(f"  {'非 LLM 合计':<30}  {non_llm:>5.1f}s  ({100 - llm_pct:.0f}%)")

    theoretical = total_work / concurrency if concurrency > 0 else total_work
    efficiency  = theoretical / wall_elapsed * 100 if wall_elapsed > 0 else 0
    print(
        f"\n  并发效率  串行总量={total_work:.0f}s ÷ 并发{concurrency}"
        f" → 理论最短={theoretical:.0f}s  实际={wall_elapsed:.1f}s  "
        f"效率={_g(f'{efficiency:.0f}%') if efficiency >= 70 else _y(f'{efficiency:.0f}%')}"
    )


# ── Trace B 汇总：跨任务 LLM 调用统计 ────────────────────────────────────────

def _print_llm_trace(results: list[dict]) -> None:
    """打印跨任务 LLM 调用汇总：各节点 token 消耗、耗时、异常短输出列表。"""
    all_by_node: dict[str, list[dict]] = {}
    short_list:  list[dict]            = []

    for r in results:
        for call in r.get("llm_calls", []):
            node = call.get("node", "?")
            all_by_node.setdefault(node, []).append(call)
            ol  = call.get("output_len", 9999)
            prv = call.get("output_preview", "")
            is_fc_valid = (call.get("node") == "fact_checker"
                           and ol < 200 and '"passed"' in prv)
            if not call.get("error") and 0 <= ol < 200 and not is_fc_valid:
                short_list.append({**call, "_task": r.get("benchmark_task_id", "?")})

    if not all_by_node:
        return

    print(f"\n\n{'═' * 72}")
    print(f"  {_b('LLM 调用汇总 (Trace B)')}  {_d('各节点 token 消耗 / 输出长度 / 异常统计')}")
    print(f"{'═' * 72}")
    print(f"  {'节点':<20}  {'次数':>5}  {'avg输入tok':>10}  {'avg输出tok':>10}  "
          f"{'avg耗时':>8}  {'avg输出长':>9}  {'avg思维链':>9}  {'ERR':>4}")
    print(f"  {'─' * 82}")

    for node, _ in _NODE_META:
        calls = all_by_node.get(node, [])
        if not calls:
            continue
        valid = [c for c in calls if "error" not in c and c.get("prompt_tokens", -1) >= 0]
        errs  = sum(1 for c in calls if "error" in c)
        if not valid:
            print(f"  {node:<20}  {len(calls):>5}  {'—':>10}  {'—':>10}  {'—':>8}  {'—':>9}  {'—':>9}  "
                  f"{_r(str(errs)):>4}")
            continue
        avg_pt = sum(c["prompt_tokens"]     for c in valid) / len(valid)
        avg_ct = sum(c["completion_tokens"] for c in valid) / len(valid)
        avg_el = sum(c["elapsed"]           for c in calls) / len(calls)
        avg_ol = sum(c.get("output_len", 0) for c in calls) / len(calls)
        has_think = any(c.get("thinking_len", -1) >= 0 for c in valid)
        avg_tl = (sum(c.get("thinking_len", 0) for c in valid) / len(valid)) if has_think else -1
        tl_s   = _y(f"{avg_tl:,.0f}") if avg_tl > 0 else (_d("0") if avg_tl == 0 else _d("N/A"))
        err_s  = _r(f"{errs:>3}") if errs else f"{errs:>3}"
        print(f"  {node:<20}  {len(calls):>5}  {avg_pt:>10,.0f}  {avg_ct:>10,.0f}  "
              f"{avg_el:>7.1f}s  {avg_ol:>9,.0f}  {tl_s:>9}  {err_s}")

    print(f"  {'─' * 72}")

    if short_list:
        print(f"\n  {_r('⚠ 异常短输出汇总（output_len < 200 chars）：')}")
        for s in short_list:
            prv = s.get("output_preview", "")
            pt  = s.get("prompt_tokens",     "?")
            ct  = s.get("completion_tokens", "?")
            pt_s = f"{pt:,}" if isinstance(pt, int) and pt >= 0 else "?"
            ct_s = f"{ct:,}" if isinstance(ct, int) and ct >= 0 else "?"
            print(f"    task={s['_task']} node={s['node']}  "
                  f"output={s.get('output_len','?')}chars  "
                  f"prompt_tokens={pt_s}  completion_tokens={ct_s}  "
                  f"content={repr(prv)}")
    else:
        print(f"\n  {_g('所有 LLM 输出均正常（无短输出告警）')}")


# ── 并发执行 ───────────────────────────────────────────────────────────────────

async def _run_with_sem(task: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        result = await run_task(task)

    if result.get("error"):
        print(f"  {_r('✗')} [{task['id']}] {task['name']:<20}  {_r(result['error'][:50])}")
    else:
        s   = result["strategy"]
        an  = result["analyst"]
        tm  = result.get("timings", {})
        rw  = result["report_writer"]
        dc  = {"shallow": _g, "medium": _y, "deep": _c}.get(s["depth"], str)
        fmt_s = _g("✓") if rw["fmt_ok"] else _r("✗")
        llm_breakdown = (
            f"analyst={tm.get('analyst', 0):.0f}s "
            f"fc={tm.get('fact_checker', 0):.0f}s "
            f"writer={tm.get('report_writer', 0):.0f}s"
        )
        short_flag = f"  {_r('⚠SHORT=' + str(rw['report_len']) + 'chars')}" if rw["report_len"] < 200 else ""
        print(
            f"  {_g('✓')} [{task['id']}] {task['name']:<20}  "
            f"{s['intent']} / {dc(s['depth'])}  "
            f"均长={an['avg_len']} 置信={an['avg_confidence']:.0%} 格式={fmt_s}  "
            f"{_d(llm_breakdown)}  ({result['elapsed_seconds']:.1f}s){short_flag}"
        )
    return result


async def main(task_ids: list[str], concurrency: int) -> None:
    # Trace B：在所有 LLM 调用前 patch litellm.acompletion
    _patch_llm_tracing()

    # 降低 analyst 节点内部 LLM 并发，减少 benchmark 场景下的 API 限速
    import app.graph.nodes as _nodes
    _nodes._ANALYST_LLM_CONCURRENCY = 3

    all_tasks: list[dict] = json.loads(TASKS_FILE.read_text(encoding="utf-8"))

    if task_ids:
        selected = [t for t in all_tasks if t["id"] in task_ids]
        missing  = set(task_ids) - {t["id"] for t in selected}
        if missing:
            print(f"未找到任务: {missing}", file=sys.stderr)
    else:
        selected = all_tasks

    if not selected:
        print("没有可运行的任务。", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'═' * 72}")
    print(
        f"  {_b('Analyst 基准测试')}  "
        f"{_d(f'共 {len(selected)} 个任务 · 并发 {min(concurrency, len(selected))}')}"
    )
    print(f"  {_d('全 8 节点真实调用: planner→retriever→extractor→evaluator→evidence→analyst→fc→writer')}")
    print(f"  {_d('Trace A/B/C 已启用: 输出侧告警 / LLM token 追踪 / State 数据流快照')}")
    print(f"{'═' * 72}\n")

    sem        = asyncio.Semaphore(concurrency)
    wall_start = time.perf_counter()
    results_unordered: list[dict] = await asyncio.gather(
        *[_run_with_sem(t, sem) for t in selected]
    )
    wall_elapsed = time.perf_counter() - wall_start

    order   = {t["id"]: i for i, t in enumerate(selected)}
    results = sorted(results_unordered, key=lambda r: order.get(r["benchmark_task_id"], 999))

    task_map = {t["id"]: t for t in selected}
    for result in results:
        _print_task(task_map[result["benchmark_task_id"]], result)

    _print_summary(results, all_tasks)
    _print_issue_distribution(results)
    _print_timing_analysis(results, wall_elapsed, min(concurrency, len(selected)))
    _print_llm_trace(results)
    print(f"\n  总耗时 {wall_elapsed:.1f}s（并发 {min(concurrency, len(selected))}）")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"analyst_{ts}.json"
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"  结果已保存: {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyst 基准测试（全 8 节点真实调用）")
    parser.add_argument(
        "tasks", nargs="*",
        help="任务 ID（如 01 03 07），不填则运行全部 12 个任务",
    )
    parser.add_argument(
        "-c", "--concurrency", type=int, default=4,
        help="最大并发任务数（默认 4；analyst 内部已限制 LLM 并发=3，总峰值≤12次）",
    )
    args = parser.parse_args()
    asyncio.run(main(args.tasks, args.concurrency))
