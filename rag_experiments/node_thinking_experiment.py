#!/usr/bin/env python
"""各节点「思维链 on/off」对比实验：质量 + 输出 token + 延迟。

对 planner / analyst / fact_checker / report_writer(draft) / report_writer(revise)
五个走 LLM 的节点，用**真实提示词** + 代表性夹具，各跑思维链 关/开 两种，测：
  · 质量：结构化节点看 JSON 是否完整 + 节点专属信号（fact_checker 看对已知错误的召回）
  · 成本：completion_tokens / reasoning_tokens / 延迟
  · 输出体量：用于给 max_tokens 留**冗余**（建议 = 关思维链下实测输出 max × 2，向上取整）

用途：决定每个节点是否关思维链、max_tokens 定多少（不卡刚好、留余量防截断）。

用法：
  D:/conda/envs/deepresearch/python.exe rag_experiments/node_thinking_experiment.py
  ... --runs 2
需要 .env 的 LLM_API_KEY（生产同一 mimo-v2.5 端点）。detect 节点已单独实验，不重复。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings  # noqa: E402
from app.graph.nodes import (  # noqa: E402
    _ANALYST_RESPONSE_FORMAT,
    _FACT_CHECKER_RESPONSE_FORMAT,
)

EXP_MAX_TOKENS = 8192  # 实验里给足，避免截断以测出"自然输出体量"


def _prompt(name: str) -> str:
    return (PROJECT_ROOT / "app" / "prompts" / f"{name}.md").read_text(encoding="utf-8")


# ── 共享夹具（固态电池主题）────────────────────────────────────────────────────
SOURCES = (
    "[C01] [固态电池技术进展](http://x/1)\n"
    "[C02] [动力电池市场综述](http://x/2)\n"
    "[C03] [A公司固态电池白皮书](http://x/3)"
)
EVIDENCE = (
    "[C01] 固态电池技术进展\n主流固态电池能量密度约 400 Wh/kg；行业预计 2027 年前后实现"
    "小规模量产，距离大规模量产仍有距离。\n\n---\n\n"
    "[C02] 动力电池市场综述\n2024 年全球动力电池出货量约 1000 GWh。\n\n---\n\n"
    "[C03] A公司固态电池白皮书\nA 公司固态电池样品循环寿命约 1000 次。"
)
DRAFT = (
    "# 固态电池发展研判\n\n固态电池是下一代储能的重要方向。其能量密度约 400 Wh/kg [C01]。"
    "行业预计固态电池将在 2025 年即可大规模量产 [C01]。\n\n2024 年全球动力电池出货约 3000 GWh "
    "[C02]。A 公司固态电池循环寿命可达 5000 次。预计到 2030 年市场规模将达 8000 亿美元。\n\n"
    "综合来看，固态电池将在三年内完全取代液态锂电池 [C01]。"
)
ERR_KEYS = ["2025", "3000", "5000", "8000", "完全取代"]  # fact_checker 召回用


# ── 各节点的 messages 构造（贴近生产）─────────────────────────────────────────
def msgs_planner():
    user = ("研究问题: 固态电池未来几年的发展与市场前景如何？\n输出语言: zh-CN\n\n"
            "请先分析问题（question_analysis），再生成研究计划。")
    return [{"role": "system", "content": _prompt("planner")},
            {"role": "user", "content": user}]


def msgs_analyst():
    sys_msg = (
        f"{_prompt('analyst')}\n\n---\n\n## 研究背景\n- 原始问题: 固态电池的发展与前景\n"
        f"- 研究意图: deep_investigation\n- 分析深度: medium\n- 研究领域: technology\n\n"
        f"## 完整子问题列表（共 1 题）\n1. [q1] 固态电池的量产时间线与循环寿命现状如何？\n\n"
        f"## Citation Registry\n{SOURCES}\n\n## 全部可用证据\n{EVIDENCE}"
    )
    user = ('当前子问题 ID: q1\n问题: 固态电池的量产时间线与循环寿命现状如何？\n\n'
            '字数要求: 500 字以内\n\n请仅针对当前子问题作答，引用使用上方 Citation Registry '
            '中的 ID（如 [C01]）。\n返回 JSON: {"sub_question_id":"q1","answer":"...",'
            '"citations":["C01"],"confidence":0.8,"evidence_gap":false}')
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]


def msgs_fact_checker():
    sub_answer = ("固态电池预计 2025 年即可大规模量产 [C01]；2024 年全球动力电池出货约 3000 GWh "
                  "[C02]；A 公司循环寿命可达 5000 次；2030 年市场规模将达 8000 亿美元；固态电池将"
                  "在三年内完全取代液态锂电池 [C01]。")
    user = (f"【研究总目标】判断固态电池量产时间、市场规模与替代节奏\n\n"
            f"子问题: 固态电池发展现状如何？\n置信度: 90%  引用: C01, C02, C03\n\n"
            f"分析结果:\n{sub_answer}\n\n## 可用来源内容\n{EVIDENCE}\n\n"
            f"请对以上分析结果进行事实核查，返回 JSON。")
    return [{"role": "system", "content": _prompt("fact_checker")},
            {"role": "user", "content": user}]


def msgs_report_draft():
    sub_answers_text = (
        "素材 1（子问题：量产时间线）\n置信度: 70%\n结论: 固态电池能量密度约 400 Wh/kg [C01]，"
        "行业预计 2027 年前后小规模量产 [C01]。\n可用引用: C01\n\n"
        "素材 2（子问题：市场规模）\n置信度: 60%\n结论: 2024 年全球动力电池出货约 1000 GWh [C02]。"
        "\n可用引用: C02\n\n"
        "素材 3（子问题：循环寿命）\n置信度: 65%\n结论: A 公司固态电池样品循环寿命约 1000 次 [C03]。"
        "\n可用引用: C03"
    )
    user = (f"# 用户要的答案（全文必须正面回应，结论前置）\n研究问题: 固态电池未来几年的发展与"
            f"市场前景如何？\n研究目标: 判断固态电池量产时间、市场规模与替代节奏\n\n"
            f"问题意图: deep_investigation  研究领域: technology  研究深度: medium\n"
            f"报告类型: deep（深度报告）\n\n## 分析素材\n\n{sub_answers_text}\n\n"
            f"## 事实核查结果\n无\n\n## 参考来源\n{SOURCES}\n\n"
            f"请综合上方素材，围绕研究目标写一篇结构自然、按主题分节的 Markdown 分析文章。")
    return [{"role": "system", "content": _prompt("report_writer")},
            {"role": "user", "content": user}]


def msgs_report_revise():
    problems_text = (
        "1. 原句：「行业预计固态电池将在 2025 年即可大规模量产 [C01]」\n   问题类型：conflict\n"
        "   修正要求：据 [C01] 应为 2027 年前后小规模量产，非 2025 大规模\n"
        "2. 原句：「2024 年全球动力电池出货约 3000 GWh [C02]」\n   问题类型：conflict\n"
        "   修正要求：据 [C02] 应为约 1000 GWh\n"
        "3. 原句：「A 公司固态电池循环寿命可达 5000 次」\n   问题类型：unsourced\n"
        "   修正要求：据 [C03] 应为约 1000 次，补 [C03]\n"
        "4. 原句：「预计到 2030 年市场规模将达 8000 亿美元」\n   问题类型：unsourced\n"
        "   修正要求：无来源，改为斜体并标注为通用知识推断、不加引用号\n"
        "5. 原句：「固态电池将在三年内完全取代液态锂电池 [C01]」\n   问题类型：overclaim\n"
        "   修正要求：证据不支持，改为审慎表述并去掉 [C01]"
    )
    user = (f"# 用户要的答案\n研究问题: 固态电池未来几年的发展与市场前景如何？\n"
            f"研究目标: 判断固态电池量产时间、市场规模与替代节奏\n\n"
            f"## 你的报告初稿\n{DRAFT}\n\n## 事实核对发现的问题（逐条修正，证据至上，其余保持不变）\n"
            f"{problems_text}\n\n## 合法引用编号（Citation Registry）\n{SOURCES}\n\n"
            f"请只针对上面每条问题做最小修正，其余段落保持初稿原样，输出完整的修订版 Markdown 报告。")
    return [{"role": "system", "content": _prompt("report_writer")},
            {"role": "user", "content": user}]


# ── 质量评估函数 ───────────────────────────────────────────────────────────────
def q_json_struct(content, required_keys):
    try:
        d = json.loads(content)
    except Exception:
        return False, "JSON坏/截断"
    miss = [k for k in required_keys if k not in d]
    return (not miss), ("缺" + ",".join(miss) if miss else "结构完整")


def q_planner(content):
    ok, note = q_json_struct(content, ["question_analysis", "research_goal", "sub_questions"])
    if not ok:
        return ok, note
    d = json.loads(content)
    n = len(d.get("sub_questions", []))
    return n >= 2, f"{n} 个子问题, goal={'有' if d.get('research_goal') else '无'}"


def q_analyst(content):
    ok, note = q_json_struct(content, ["answer", "citations", "confidence"])
    if not ok:
        return ok, note
    d = json.loads(content)
    cits = [str(c) for c in d.get("citations", [])]
    bad = [c for c in cits if c not in {"C01", "C02", "C03"}]
    return (len(d.get("answer", "")) > 40 and not bad), \
        f"答{len(d.get('answer',''))}字, 引用{cits}{' 越界!' if bad else ''}"


def q_fact_checker(content):
    ok, note = q_json_struct(content, ["passed", "issues"])
    if not ok:
        return ok, note
    d = json.loads(content)
    blob = json.dumps(d.get("issues", []), ensure_ascii=False)
    hit = sum(1 for k in ERR_KEYS if k in blob)
    return hit >= 4, f"passed={d.get('passed')}, 召回 {hit}/{len(ERR_KEYS)}, issues={len(d.get('issues',[]))}"


def q_report(content):
    n_cit = content.count("[C0")
    n_head = sum(1 for ln in content.splitlines() if ln.strip().startswith("#"))
    return len(content) > 300, f"{len(content)}字, {n_cit}处引用, {n_head}节标题"


def q_revise(content):
    # 修订应把 3000/5000/8000 这些错误数字纠正/删除 → 残留越少越好
    resid = [k for k in ["3000", "5000", "8000"] if k in content]
    labeled = "基于通用知识" in content
    return len(resid) <= 1, f"{len(content)}字, 残留错误数字{resid or '无'}, 标注={'有' if labeled else '无'}"


NODES = [
    ("planner",      msgs_planner,       {"type": "json_object"},          q_planner),
    ("analyst",      msgs_analyst,       _ANALYST_RESPONSE_FORMAT,         q_analyst),
    ("fact_checker", msgs_fact_checker,  _FACT_CHECKER_RESPONSE_FORMAT,    q_fact_checker),
    ("report_draft", msgs_report_draft,  None,                             q_report),
    ("report_revise", msgs_report_revise, None,                            q_revise),
]


def _model_name():
    m = settings.llm_model
    if "/" in m:
        return m
    if settings.llm_base_url or settings.llm_provider.lower() in ("openai", "xiaomi", "custom"):
        return f"openai/{m}"
    return f"{settings.llm_provider}/{m}"


def _usage(resp):
    u = resp.usage
    ct = getattr(u, "completion_tokens", 0) or 0
    pt = getattr(u, "prompt_tokens", 0) or 0
    det = getattr(u, "completion_tokens_details", None)
    rt = 0
    if isinstance(det, dict):
        rt = det.get("reasoning_tokens", 0) or 0
    elif det is not None:
        rt = getattr(det, "reasoning_tokens", 0) or 0
    return pt, ct, rt


async def run_one(messages, response_format, quality_fn, thinking, temperature=0.3):
    import litellm
    kw = dict(model=_model_name(), messages=messages, temperature=temperature,
              max_tokens=EXP_MAX_TOKENS, api_key=settings.llm_api_key or None,
              api_base=settings.llm_base_url or None, timeout=180,
              extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}})
    if response_format:
        kw["response_format"] = response_format
    t0 = time.perf_counter()
    try:
        resp = await litellm.acompletion(**kw)
        dt = time.perf_counter() - t0
        content = resp.choices[0].message.content or ""
        pt, ct, rt = _usage(resp)
        ok, note = quality_fn(content)
        return {"ok": ok, "note": note, "prompt_t": pt, "completion_t": ct,
                "reasoning_t": rt, "latency": round(dt, 1), "content": content, "err": ""}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "note": "", "prompt_t": 0, "completion_t": 0, "reasoning_t": 0,
                "latency": round(time.perf_counter() - t0, 1), "content": "", "err": f"{type(exc).__name__}: {exc}"}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    rows, samples = [], []
    for name, build, fmt, qfn in NODES:
        messages = build()
        for thinking in (False, True):
            runs = [await run_one(messages, fmt, qfn, thinking) for _ in range(args.runs)]
            ok = [r for r in runs if not r["err"]]
            base = ok[0] if ok else runs[0]
            agg = {
                "ok_rate": (sum(1 for r in ok if r["ok"]) / len(ok)) if ok else 0.0,
                "completion_max": max((r["completion_t"] for r in ok), default=0),
                "completion_avg": (sum(r["completion_t"] for r in ok) / len(ok)) if ok else 0,
                "reasoning_avg": (sum(r["reasoning_t"] for r in ok) / len(ok)) if ok else 0,
                "latency_avg": (sum(r["latency"] for r in ok) / len(ok)) if ok else 0,
            }
            rows.append({"node": name, "thinking": thinking, "note": base["note"],
                         "err": base["err"], **agg})
            samples.append((f"{name} | thinking={'ON' if thinking else 'OFF'}", base["content"]))

    # ── 表 ──────────────────────────────────────────────────────────────────
    hdr = (f"{'节点':<15}{'思维链':<7}{'质量ok':>7}{'输出avg':>8}{'输出max':>8}"
           f"{'思维avg':>8}{'延迟s':>7}  质量备注")
    lines = [f"各节点 思维链 on/off 对比  |  模型 {settings.llm_model}  |  runs={args.runs}",
             "", hdr, "-" * len(hdr)]
    off_max = {}
    for r in rows:
        if not r["thinking"]:
            off_max[r["node"]] = r["completion_max"]
        note = ("调用失败:" + r["err"][:36]) if r["err"] else r["note"]
        lines.append(
            f"{r['node']:<15}{'开' if r['thinking'] else '关':<7}"
            f"{r['ok_rate']*100:>6.0f}%{r['completion_avg']:>8.0f}{r['completion_max']:>8.0f}"
            f"{r['reasoning_avg']:>8.0f}{r['latency_avg']:>7.1f}  {note}"
        )

    # ── max_tokens 冗余建议（基于关思维链的实测输出 max × 2，向上取整到 512，下限 1024）──
    lines += ["", "建议 max_tokens（关思维链下 实测输出max × 2 冗余，向上取整 512，下限 1024）："]
    for node, cmax in off_max.items():
        rec = max(1024, math.ceil(cmax * 2 / 512) * 512) if cmax else 0
        lines.append(f"  {node:<15} 实测输出max {cmax:>5}  →  建议 max_tokens = {rec}")

    lines += ["",
              "读法：思维avg=reasoning_tokens（关思维链应为 0）；质量ok 若"
              "开关差不多 → 可安全关思维链省 token 并恢复温度；若关了质量掉 → 保留思维链。",
              "冗余：max_tokens 不卡实测刚好，留 2× 余量防长尾输出被截断（detect 曾因卡太紧截断）。"]

    out = "\n".join(lines)
    print(out)
    res_dir = PROJECT_ROOT / "rag_experiments" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "node_thinking_experiment.txt").write_text(out, encoding="utf-8")
    # 存样本供人工看生成质量（尤其 planner / report 这类难自动评的）
    sample_txt = "\n\n".join(f"===== {tag} =====\n{c[:1500]}" for tag, c in samples)
    (res_dir / "node_thinking_samples.txt").write_text(sample_txt, encoding="utf-8")
    print(f"\n[已保存] {res_dir / 'node_thinking_experiment.txt'}")
    print(f"[样本]   {res_dir / 'node_thinking_samples.txt'}")


if __name__ == "__main__":
    asyncio.run(main())
