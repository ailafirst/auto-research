"""验证 reranker 短板的成因：是"论文摘要粒度/格式"被压制，还是"论文真不相关"？

假设：全局 rerank 把 OpenAlex 论文压到 ~0 分，是因为把"整段密集摘要"当一个 passage
喂给 cross-encoder，对短的 meta 查询打分吃亏；若按句子粒度重排取最高分（模拟生产里
把文档切 chunk 再 rerank），论文的最佳句应显著高于整段摘要分。

  blob_score      : 整段"标题+摘要"对问题的 rerank 分（= 当前 ② 的做法）
  best_sent_score : 摘要切句后，各句 rerank 取最高（= 切 chunk 后论文能拿到的分）

若 best_sent 明显 > blob，且能越过"web top-10 分数线"，则结论是——
**别在 rerank 之后打补丁，应把论文和网页一样切 chunk 再排**（修输入，非修输出）。

用法：D:\\conda\\envs\\deepresearch\\python.exe rag_experiments\\rerank_granularity_experiment.py
"""

from __future__ import annotations

import asyncio
import re
import statistics as stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from source_strategy_experiment import (  # noqa: E402
    ACADEMIC_DOMAINS, openalex_search, tavily_search,
)
from rerank_enhance_experiment import rerank_scores  # noqa: E402

RESEARCH_QS = [
    {"id": "bci", "question": "2026 brain-computer interface latest research directions worth deep investment",
     "queries": ["brain-computer interface latest research directions 2026",
                 "most promising brain-computer interface research directions worth investing"]},
    {"id": "battery", "question": "solid-state battery latest research breakthroughs and most promising directions 2026",
     "queries": ["solid-state battery latest research breakthroughs 2026",
                 "most promising solid-state battery research directions"]},
]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 20]


async def run(q: dict, report: list[str]) -> None:
    question = q["question"]
    print(f"\n=== [{q['id']}] {question}")

    # web+academic 基线，确定 "top-10 分数线"（第10高分）
    web: list[dict] = []
    for sub in q["queries"]:
        web += await tavily_search(sub)
        web += await tavily_search(sub, include_domains=ACADEMIC_DOMAINS)
    web_texts = [w.get("text", "")[:1024] for w in web]
    web_scores = sorted(await rerank_scores(question, web_texts), reverse=True) if web_texts else []
    cutoff = web_scores[9] if len(web_scores) >= 10 else (web_scores[-1] if web_scores else 0.0)

    # OpenAlex 论文
    papers: list[dict] = []
    for sub in q["queries"]:
        papers += await openalex_search(sub)
    # 去重
    seen = set()
    papers = [p for p in papers if p["url"] and not (p["url"] in seen or seen.add(p["url"]))]

    # blob 分
    blob_scores = await rerank_scores(question, [p.get("text", "")[:1024] for p in papers])

    # 句粒度：拉平所有句子一次打分，再按 paper 取 max
    flat_sents: list[str] = []
    owner: list[int] = []
    for i, p in enumerate(papers):
        sents = split_sentences(p.get("text", ""))
        for s in sents:
            flat_sents.append(s[:512])
            owner.append(i)
    sent_scores = await rerank_scores(question, flat_sents) if flat_sents else []
    best_sent = [0.0] * len(papers)
    for idx, sc in zip(owner, sent_scores):
        best_sent[idx] = max(best_sent[idx], sc)

    n = len(papers)
    blob_pass = sum(1 for s in blob_scores if s >= cutoff)
    sent_pass = sum(1 for s in best_sent if s >= cutoff)

    report.append(f"\n## [{q['id']}] {question}\n")
    report.append(f"- web/academic top-10 分数线(cutoff) = **{cutoff:.3f}**；OpenAlex 论文 {n} 篇")
    report.append("")
    report.append("| 指标 | blob(整段摘要) | best_sent(切句取最高) |")
    report.append("|---|---|---|")
    if blob_scores and best_sent:
        report.append(f"| 平均分 | {stats.mean(blob_scores):.3f} | {stats.mean(best_sent):.3f} |")
        report.append(f"| 中位数 | {stats.median(blob_scores):.3f} | {stats.median(best_sent):.3f} |")
        report.append(f"| 最高分 | {max(blob_scores):.3f} | {max(best_sent):.3f} |")
    report.append(f"| **越过分数线的论文数** | **{blob_pass}/{n}** | **{sent_pass}/{n}** |")
    report.append("")
    # 展示提升最大的 3 篇
    gains = sorted(
        [(best_sent[i] - blob_scores[i], papers[i], blob_scores[i], best_sent[i]) for i in range(n)],
        key=lambda x: x[0], reverse=True,
    )[:3]
    report.append("- 提升最大的论文（blob→best_sent）：")
    for g, p, b, bs in gains:
        report.append(f"  - {b:.3f}→{bs:.3f}  《{(p.get('title') or '')[:70]}》")


async def main() -> None:
    import time
    ts = time.strftime("%Y%m%d_%H%M%S")
    report = [
        "# reranker 短板成因验证：粒度 vs 相关性",
        f"> 时间: {ts} | 仅测试",
        "> 若 best_sent(切句) 明显 > blob(整段) 且越过分数线，则短板是「粒度/格式」，应切 chunk 修输入",
    ]
    for q in RESEARCH_QS:
        await run(q, report)
    out = Path(__file__).resolve().parent / "results" / f"rerank_granularity_{ts}.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"\nWROTE {out}")


if __name__ == "__main__":
    asyncio.run(main())
