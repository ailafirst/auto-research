"""单节点输出下限测试工具

对单个任务运行完整 8 节点流水线，限定指定节点的 max_tokens，
打印每次 LLM 调用的完整输出，用于人工判断最小有效输出长度。

用法：
  # 测试 analyst 节点在不同 max_tokens 下的输出质量
  python benchmark/test_analyst.py 03 --node analyst --max-tokens 500
  python benchmark/test_analyst.py 03 --node analyst --max-tokens 800
  python benchmark/test_analyst.py 03 --node analyst --max-tokens 1200

  # 测试 report_writer
  python benchmark/test_analyst.py 03 --node report_writer --max-tokens 1000
  python benchmark/test_analyst.py 04 --node report_writer --max-tokens 2000

  # 同时限制多个节点（逗号分隔）
  python benchmark/test_analyst.py 03 --node analyst:800,report_writer:1500

  # 不限制（观察自然输出分布）
  python benchmark/test_analyst.py 03
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
import time
from pathlib import Path

# Windows GBK 终端强制 UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

TASKS_FILE = Path(__file__).parent / "tasks.json"

GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RED    = "\033[31m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def _g(t): return f"{GREEN}{t}{RESET}"
def _y(t): return f"{YELLOW}{t}{RESET}"
def _c(t): return f"{CYAN}{t}{RESET}"
def _r(t): return f"{RED}{t}{RESET}"
def _b(t): return f"{BOLD}{t}{RESET}"
def _d(t): return f"{DIM}{t}{RESET}"

# ── 节点 max_tokens 覆盖表（由 CLI 参数填充）────────────────────────────────
_NODE_MAX_TOKENS: dict[str, int] = {}

# ── 当前节点标记（patch 时注入 max_tokens）──────────────────────────────────
_current_node: list[str] = ["?"]  # 用列表绕过闭包只读限制

# ── 每次 LLM 调用的完整记录 ──────────────────────────────────────────────────
_call_log: list[dict] = []


def _patch_llm() -> None:
    """Monkey-patch litellm.acompletion：注入 max_tokens + 记录完整输出。"""
    import litellm as _litellm

    _orig = _litellm.acompletion

    async def _traced(**kwargs):
        node = _current_node[0]
        # 开启 MiMo 思维链
        if "enable_thinking" not in kwargs:
            kwargs = {**kwargs, "enable_thinking": True}
        # 应用节点 max_tokens 限制
        if node in _NODE_MAX_TOKENS:
            kwargs = {**kwargs, "max_tokens": _NODE_MAX_TOKENS[node]}

        t0 = time.perf_counter()
        response = None
        error: str | None = None
        try:
            response = await _orig(**kwargs)
            return response
        except Exception as exc:
            error = str(exc)[:300]
            raise
        finally:
            elapsed = round(time.perf_counter() - t0, 2)
            rec: dict = {
                "node":             node,
                "elapsed":          elapsed,
                "applied_max":      int(kwargs.get("max_tokens", -1)),
                "prompt_tokens":    -1,
                "completion_tokens":-1,
                "output_len":       -1,
                "thinking_len":     -1,
                "output":           "",
                "thinking_preview": "",
                "error":            error or "",
            }
            if response is not None and response.choices:
                msg     = response.choices[0].message
                content = msg.content or ""
                thinking = getattr(msg, "reasoning_content", None) or ""
                usage   = getattr(response, "usage", None)
                rec["prompt_tokens"]     = getattr(usage, "prompt_tokens",     -1) if usage else -1
                rec["completion_tokens"] = getattr(usage, "completion_tokens", -1) if usage else -1
                rec["output_len"]        = len(content)
                rec["thinking_len"]      = len(thinking)
                rec["output"]            = content
                rec["thinking_preview"]  = thinking[:300]
                # ── Cache 命中字段（Anthropic 协议）──────────────────────────
                rec["cache_read_tokens"]     = (getattr(usage, "cache_read_input_tokens",     None) or 0) if usage else 0
                rec["cache_creation_tokens"] = (getattr(usage, "cache_creation_input_tokens", None) or 0) if usage else 0
                # OpenAI 协议备用
                details = getattr(usage, "prompt_tokens_details", None)
                rec["cached_tokens_oai"]     = (getattr(details, "cached_tokens", None) or 0) if details else 0
            _call_log.append(rec)

    _litellm.acompletion = _traced


def _print_call(rec: dict, call_idx: int) -> None:
    """打印单次 LLM 调用的完整输出，格式清晰便于人工判断。"""
    nd  = rec["node"]
    ol  = rec["output_len"]
    tl  = rec["thinking_len"]
    pt  = rec["prompt_tokens"]
    ct  = rec["completion_tokens"]
    el  = rec["elapsed"]
    mx  = rec["applied_max"]
    err = rec["error"]
    out = rec["output"]
    tpv = rec["thinking_preview"]

    mx_s  = _y(f" [max_tokens={mx}]") if mx > 0 else ""
    ol_s  = _r(f"{ol:,}") if ol >= 0 and ol < 100 else (f"{ol:,}" if ol >= 0 else "?")
    tl_s  = _y(f"{tl:,}") if tl > 0 else _d("0")
    pt_s  = f"{pt:,}" if pt >= 0 else "?"
    ct_s  = f"{ct:,}" if ct >= 0 else "?"

    bar = "═" * 68
    print(f"\n{bar}")
    print(f"  {_b(f'调用 #{call_idx + 1}')}  节点={_c(nd)}{mx_s}")
    print(f"  输入={pt_s}tok  输出={ct_s}tok  耗时={el}s  "
          f"content={ol_s}chars  思维链={tl_s}chars")
    print(bar)

    if err:
        print(f"  {_r('ERROR:')} {err}")
        return

    if tpv:
        print(f"  {_d('思维链片段:')} {_d(repr(tpv[:120]))}")
        print()

    if out:
        for line in out.splitlines():
            print(f"  {line}")
    else:
        print(f"  {_r('(空输出)')}")


async def run_task(task: dict) -> None:
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
    import app.graph.nodes as _nodes
    from app.core.config import settings as _s
    print(f"  [并发配置] analyst={_s.analyst_concurrency}  fact_checker={_s.fact_checker_concurrency}"
          f"  llm_global={_s.llm_max_concurrency}", flush=True)

    from datetime import datetime
    now = datetime.now().isoformat()
    user_hints: dict = {}
    if task.get("report_type", "deep") != "deep":
        user_hints["report_type"] = task["report_type"]

    state: dict = {
        "task_id":           f"test_{task['id']}",
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

    pipeline = [
        ("planner",          planner_node),
        ("retriever",        retriever_node),
        ("content_extractor",content_extractor_node),
        ("source_evaluator", source_evaluator_node),
        ("evidence_builder", evidence_builder_node),
        ("analyst",          analyst_node),
        ("fact_checker",     fact_checker_node),
        ("report_writer",    report_writer_node),
    ]

    call_idx_before = 0
    for name, fn in pipeline:
        _current_node[0] = name
        print(f"\n{'─' * 68}")
        print(f"  {_d('→')} 节点: {_b(name)}", end="", flush=True)
        t0 = time.perf_counter()
        try:
            result = await fn(state)
            state.update(result)
        except Exception as exc:
            print(f"  {_r('FAILED:')} {exc}")
            break
        elapsed = round(time.perf_counter() - t0, 2)
        print(f"  ({elapsed}s)")

        # 打印本节点产生的每次 LLM 调用
        new_calls = _call_log[call_idx_before:]
        call_idx_before = len(_call_log)
        for i, rec in enumerate(new_calls):
            _print_call(rec, len(_call_log) - len(new_calls) + i)

    # 最终摘要
    print(f"\n{'═' * 68}")
    print(f"  {_b('完成')}")
    report = state.get("final_report", "")
    print(f"  report_writer 输出: {_r(str(len(report))) if len(report) < 200 else _g(str(len(report)))} chars")
    if report:
        print(f"\n  {_d('报告前 300 chars:')}")
        for line in report[:300].splitlines():
            print(f"  {line}")
        if len(report) > 300:
            print(f"  {_d('...')}")

    print(f"\n  {_d('LLM 调用汇总（含 cache 命中）:')}")
    print(f"  {'节点':<20}  {'输入tok':>9}  {'cache创建':>10}  {'cache命中':>10}  "
          f"{'输出tok':>9}  {'耗时':>6}  {'content':>8}")
    total_cache_read = 0
    total_cache_create = 0
    for rec in _call_log:
        nd  = rec["node"]
        pt  = f"{rec['prompt_tokens']:,}"      if rec["prompt_tokens"]     >= 0 else "?"
        ct  = f"{rec['completion_tokens']:,}"  if rec["completion_tokens"] >= 0 else "?"
        ol  = rec["output_len"]
        mx  = rec["applied_max"]
        cr  = rec.get("cache_read_tokens", 0)
        cc  = rec.get("cache_creation_tokens", 0)
        oai = rec.get("cached_tokens_oai", 0)
        # 优先 Anthropic 字段，回退 OpenAI 字段
        hit     = cr  if cr  else oai
        created = cc
        total_cache_read   += hit
        total_cache_create += created
        ol_s  = _r(f"{ol:,}") if 0 <= ol < 100 else (f"{ol:,}" if ol >= 0 else "?")
        hit_s = _g(f"{hit:,}") if hit > 0 else _d("0")
        cc_s  = _y(f"{created:,}") if created > 0 else _d("0")
        mx_s  = f" [max={mx:,}]" if mx > 0 else ""
        print(f"  {nd:<20}  {pt:>9}  {cc_s:>10}  {hit_s:>10}  "
              f"{ct:>9}  {rec['elapsed']:>5.1f}s  {ol_s:>8}{mx_s}")

    analyst_calls = [r for r in _call_log if r["node"] == "analyst"]
    if len(analyst_calls) > 1:
        print(f"\n  {_b('Cache 命中率分析（analyst）:')}")
        hits    = sum(1 for r in analyst_calls if r.get("cache_read_tokens", 0) or r.get("cached_tokens_oai", 0))
        total   = len(analyst_calls)
        hit_tok = sum(r.get("cache_read_tokens", 0) or r.get("cached_tokens_oai", 0) for r in analyst_calls)
        cre_tok = sum(r.get("cache_creation_tokens", 0) for r in analyst_calls)
        print(f"  analyst 调用: {total} 次  命中: {_g(str(hits))} 次  未命中: {_y(str(total - hits))} 次")
        print(f"  cache 创建 tokens: {_y(str(cre_tok)):>8}  cache 读取 tokens: {_g(str(hit_tok))}")
        if total > 1:
            first_pt = analyst_calls[0]["prompt_tokens"]
            others   = [r["prompt_tokens"] for r in analyst_calls[1:] if r["prompt_tokens"] >= 0]
            print(f"  首次 prompt_tokens: {first_pt:,}  后续均值: "
                  f"{round(sum(others)/len(others)):,}" if others else "")


async def _warmup_models() -> None:
    """进程启动时预热 embedding / reranker 模型。
    消除 evidence_builder 和 analyst 节点的首次冷启动延迟（bge-m3 约 20-30s，
    bge-reranker 约 5-10s），使节点耗时只反映实际计算，不含模型加载。
    """
    from app.services.rag_service import get_rag_service, _get_reranker
    from app.core.config import settings as _s
    t0 = time.perf_counter()
    print(f"  {_d('预热模型中...')}", flush=True)
    rag = get_rag_service()
    if _s.embedding_provider == "st":
        await rag._get_st_model()
    elif _s.embedding_provider == "fastembed":
        await rag._get_fastembed_model()
    if _s.reranker_enabled:
        await _get_reranker()
    print(f"  {_d(f'模型预热完成（{time.perf_counter()-t0:.1f}s）')}", flush=True)


async def main(task_id: str, node_limits: str) -> None:
    # 解析节点限制（如 "analyst:800" 或 "analyst:800,report_writer:1500"）
    if node_limits:
        for pair in node_limits.split(","):
            pair = pair.strip()
            if ":" in pair:
                node, val = pair.split(":", 1)
                try:
                    _NODE_MAX_TOKENS[node.strip()] = int(val.strip())
                except ValueError:
                    print(f"[警告] 格式错误，忽略: {pair}", file=sys.stderr)

    _patch_llm()
    await _warmup_models()

    all_tasks = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    task = next((t for t in all_tasks if t["id"] == task_id), None)
    if task is None:
        print(f"找不到任务 {task_id!r}，可用: {[t['id'] for t in all_tasks]}", file=sys.stderr)
        sys.exit(1)

    print(f"{'═' * 68}")
    print(f"  {_b('test_analyst')} — 单任务输出质量测试")
    print(f"  任务: [{task['id']}] {task['name']}  ({task.get('report_type','?')} / {task.get('search_depth','?')})")
    if _NODE_MAX_TOKENS:
        print(f"  {_y('max_tokens 限制:')} {_NODE_MAX_TOKENS}")
    else:
        print(f"  {_d('无 max_tokens 限制（自然输出）')}")
    print(f"{'═' * 68}")

    await run_task(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="单任务输出质量测试，打印每次 LLM 完整输出，用于确定 short 阈值",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("task_id", help="任务 ID，如 03")
    parser.add_argument(
        "--node", dest="node_limits", default="",
        metavar="NODE:MAX_TOKENS[,NODE:MAX_TOKENS]",
        help="限制节点 max_tokens，如 analyst:800 或 analyst:800,report_writer:1500",
    )
    args = parser.parse_args()
    asyncio.run(main(args.task_id, args.node_limits))
