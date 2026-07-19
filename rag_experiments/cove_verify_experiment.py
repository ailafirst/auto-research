#!/usr/bin/env python
"""CoVe 核对调用的成本/质量对比实验：思维链 on/off × 最大输出上限。

对同一份「含已知错误的报告初稿 + 证据」，用不同配置跑 CoVe 问题检出调用（方案丙里
「多的那一次」），比较：
  · 质量：检出问题数、对已知错误的召回率、JSON 是否完整（截断=坏）
  · 成本：prompt / completion / reasoning token、总 token、延迟

结论用于回填 app/graph/nodes.py 的 _COVE_DETECT_THINKING / _COVE_DETECT_MAX_TOKENS。

用法：
  D:/conda/envs/deepresearch/python.exe rag_experiments/cove_verify_experiment.py
  ... --runs 2 --max-tokens 256 512 1024
需要 .env 里的 LLM_API_KEY（与生产同一个 mimo-v2.5 端点）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings  # noqa: E402

VERIFIER_PROMPT = (PROJECT_ROOT / "app/prompts/report_verifier.md").read_text(encoding="utf-8")

_PROBLEMS_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "report_problems",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "problems": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "quote": {"type": "string"},
                            "issue": {"type": "string", "enum": ["conflict", "unsourced", "overclaim", "citation_mismatch"]},
                            "fix":   {"type": "string"},
                        },
                        "required": ["quote", "issue", "fix"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["problems"],
            "additionalProperties": False,
        },
    },
}

# ── 固定夹具：一份含已知错误的报告初稿 + 与之对照的证据 ──────────────────────────
SOURCES_TEXT = (
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
    "# 固态电池发展研判\n\n"
    "固态电池是下一代储能的重要方向。其能量密度约 400 Wh/kg [C01]，显著高于传统液态锂电池。"
    "行业预计固态电池将在 2025 年即可大规模量产 [C01]。\n\n"
    "从市场看，2024 年全球动力电池出货约 3000 GWh [C02]，增长强劲。A 公司固态电池循环寿命"
    "可达 5000 次，量产在即。预计到 2030 年市场规模将达 8000 亿美元。\n\n"
    "综合来看，固态电池将在三年内完全取代液态锂电池 [C01]。"
)
# 已知错误（key = 该错误句里的独特标记，出现在模型 quote 中即算召回）
KNOWN = [
    ("2025 年即可大规模量产（C01 实为 2027 小规模）", "2025"),
    ("2024 出货 3000 GWh（C02 实为 1000）", "3000"),
    ("循环寿命 5000 次且无引用（C03 实为 1000）", "5000"),
    ("2030 年 8000 亿美元，无任何来源", "8000"),
    ("三年内完全取代，过度外推", "完全取代"),
]
# 正确对照句「能量密度约 400 Wh/kg [C01]」不应被标记（误报观测点）
CONTROL_KEY = "400 Wh/kg"


def build_messages() -> list[dict]:
    user = (
        f"研究问题: 固态电池未来几年的发展与市场前景如何？\n"
        f"研究目标: 判断固态电池量产时间、市场规模与对液态电池的替代节奏\n\n"
        f"## Citation Registry（唯一合法引用编号）\n{SOURCES_TEXT}\n\n"
        f"## 可用证据正文（逐条核对依据）\n{EVIDENCE}\n\n"
        f"## 报告初稿（在其中定位有问题的具体断言）\n{DRAFT}\n\n"
        f"只挑出有问题的具体断言，每条给出 quote / issue / fix（严格 JSON）；无问题返回空列表。"
    )
    return [
        {"role": "system", "content": VERIFIER_PROMPT},
        {"role": "user", "content": user},
    ]


def _model_name() -> str:
    m = settings.llm_model
    if "/" in m:
        return m
    if settings.llm_base_url or settings.llm_provider.lower() in ("openai", "xiaomi", "custom"):
        return f"openai/{m}"
    return f"{settings.llm_provider}/{m}"


def _usage_tokens(resp) -> tuple[int, int, int]:
    u = resp.usage
    pt = getattr(u, "prompt_tokens", 0) or 0
    ct = getattr(u, "completion_tokens", 0) or 0
    det = getattr(u, "completion_tokens_details", None)
    rt = 0
    if isinstance(det, dict):
        rt = det.get("reasoning_tokens", 0) or 0
    elif det is not None:
        rt = getattr(det, "reasoning_tokens", 0) or 0
    return pt, ct, rt


async def one_call(messages, thinking: bool, max_tokens: int) -> dict:
    import litellm

    body = {"thinking": {"type": "enabled" if thinking else "disabled"}}
    t0 = time.perf_counter()
    err = ""
    content, pt, ct, rt = "", 0, 0, 0
    try:
        resp = await litellm.acompletion(
            model=_model_name(),
            messages=messages,
            temperature=0.1,
            max_tokens=max_tokens,
            api_key=settings.llm_api_key or None,
            api_base=settings.llm_base_url or None,
            response_format=_PROBLEMS_FORMAT,
            extra_body=body,
            timeout=120,
        )
        content = resp.choices[0].message.content or ""
        pt, ct, rt = _usage_tokens(resp)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    dt = time.perf_counter() - t0

    # 评估质量
    json_ok, n_problems, recall, false_pos = False, 0, 0.0, False
    if content:
        try:
            probs = json.loads(content).get("problems", [])
            json_ok = True
            n_problems = len(probs)
            quotes = " ".join(p.get("quote", "") for p in probs)
            hit = sum(1 for _, key in KNOWN if key in quotes)
            recall = hit / len(KNOWN)
            false_pos = CONTROL_KEY in quotes  # 把正确句也标了 = 误报
        except Exception:
            json_ok = False

    return {
        "thinking": thinking, "max_tokens": max_tokens,
        "json_ok": json_ok, "n_problems": n_problems,
        "recall": recall, "false_pos": false_pos,
        "prompt_t": pt, "completion_t": ct, "reasoning_t": rt,
        "latency_s": round(dt, 2), "err": err,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1, help="每个配置重复次数（取均值）")
    ap.add_argument("--max-tokens", type=int, nargs="+", default=[256, 512, 1024])
    args = ap.parse_args()

    messages = build_messages()
    configs = [(t, mt) for t in (False, True) for mt in args.max_tokens]

    rows: list[dict] = []
    for thinking, mt in configs:
        runs = [await one_call(messages, thinking, mt) for _ in range(args.runs)]
        ok = [r for r in runs if not r["err"]]
        base = runs[0]
        if ok:
            agg = {
                "recall":       sum(r["recall"] for r in ok) / len(ok),
                "n_problems":   sum(r["n_problems"] for r in ok) / len(ok),
                "prompt_t":     sum(r["prompt_t"] for r in ok) / len(ok),
                "completion_t": sum(r["completion_t"] for r in ok) / len(ok),
                "reasoning_t":  sum(r["reasoning_t"] for r in ok) / len(ok),
                "latency_s":    sum(r["latency_s"] for r in ok) / len(ok),
                "json_ok":      sum(1 for r in ok if r["json_ok"]) / len(ok),
                "false_pos":    any(r["false_pos"] for r in ok),
            }
        else:
            agg = {"recall": 0, "n_problems": 0, "prompt_t": 0, "completion_t": 0,
                   "reasoning_t": 0, "latency_s": 0, "json_ok": 0, "false_pos": False}
        rows.append({"thinking": thinking, "max_tokens": mt, "err": base["err"], **agg})

    # ── 输出表 ─────────────────────────────────────────────────────────────
    hdr = (f"{'思维链':<6}{'max_tok':>8}{'json_ok':>8}{'召回':>7}{'检出':>6}"
           f"{'prompt':>8}{'输出tok':>8}{'思维tok':>8}{'延迟s':>7}  备注")
    lines = [
        f"CoVe 核对调用对比实验  |  模型 {settings.llm_model}  |  runs={args.runs}",
        f"已知错误 {len(KNOWN)} 处，正确对照句 1 句（不应被标）",
        "",
        hdr,
        "-" * len(hdr),
    ]
    for r in rows:
        note = []
        if r["err"]:
            note.append("调用失败:" + r["err"][:40])
        if r["false_pos"]:
            note.append("误报正确句")
        lines.append(
            f"{'开' if r['thinking'] else '关':<6}{r['max_tokens']:>8}"
            f"{r['json_ok']*100:>7.0f}%{r['recall']*100:>6.0f}%{r['n_problems']:>6.1f}"
            f"{r['prompt_t']:>8.0f}{r['completion_t']:>8.0f}{r['reasoning_t']:>8.0f}"
            f"{r['latency_s']:>7.2f}  {' / '.join(note)}"
        )
    lines += [
        "",
        "读法：召回=检出多少已知错误；输出tok=completion_tokens（计费的输出，含思维tok）；",
        "思维tok=reasoning_tokens（关思维链应为 0）。理想：关思维链即可拿到高召回 + 低输出，",
        "且小 max_tokens 下 json_ok 仍 100%（开思维链时推理会吃光预算 → json_ok 掉、召回崩）。",
    ]
    out = "\n".join(lines)
    print(out)
    res_dir = PROJECT_ROOT / "rag_experiments" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "cove_verify_experiment.txt").write_text(out, encoding="utf-8")
    print(f"\n[已保存] {res_dir / 'cove_verify_experiment.txt'}")


if __name__ == "__main__":
    asyncio.run(main())
