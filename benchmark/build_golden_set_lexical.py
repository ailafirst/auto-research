"""黄金集扩充 —— 补充"精确词锚点"问题（产品型号/版本号/DOI/法条编号）。

背景：黄金集构造刻意要求"不照抄原句"、拉大字面距离（build_golden_set.py 的
_GEN_TMPL 规则 3），这让黄金集系统性低估了精确字面匹配场景——120 题里这类问题
只有 1-2 条（H200/H100 那条），样本量不足以判断 BM25 混合检索对这类场景是否
真的有用（见 docs/RAG改进方案.md「Hybrid Sparse」否决记录里的讨论）。

本脚本**不重新爬取**、复用 golden_set.json 里已冻结的语料，只做两件事：
  1. 正则扫描每个任务语料，找含精确词锚点（产品型号/版本号/DOI/法条编号）且
     该锚点在本任务语料里"稀有"（出现 chunk 数少，唯一性强，才是 BM25 真正的
     目标场景，避免选到到处都提的通用词）的 chunk，排除已被用作现有 10 题
     gold 的 chunk。
  2. 复用 build_golden_set.py 同款 LLM 生成问题（同一份 prompt/规则），保证
     和原 120 题风格一致、同样不照抄原句。

追加进 golden_set.json 的 qa 列表，corpus 不变（不影响原 120 题的 gold_cids）。
新增条目带 anchor_terms 字段，供后续单独切出「锚点子集」跑对照评测。

用法：
  python benchmark/build_golden_set_lexical.py --scan        # 只扫描不生成，人工检查候选
  python benchmark/build_golden_set_lexical.py                # 扫描 + LLM 生成 + 写回
  python benchmark/build_golden_set_lexical.py --per-task 4   # 每任务候选上限（默认 4）
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

GOLDEN_DIR = Path(__file__).parent / "GoldenDataset"
GOLDEN_PATH = GOLDEN_DIR / "golden_set.json"

MIN_CHUNK_LEN = 200
MAX_OCCURRENCE = 1      # 锚点必须在本任务语料里唯一出现（严格单 gold，避免同一型号/DOI 出现在
                         # 两个 chunk 里造成"标了一个、另一个同样正确"的测量假象——见
                         # improvement_plan.md 里已记录的单 gold 低估问题，本次不能重犯）
PER_TASK_DEFAULT = 4
MAX_PER_SOURCE = 1       # 同一来源 URL 最多取 1 条，避免锚点题堆在同一篇文章

# 锚点正则：产品/型号（字母+数字组合）、显式版本号（v 前缀强制）、DOI、法条编号（中英）。
# 不用泛化的 \d+\.\d+：那会把百分比/价格/IP 段/TOC 章节号全当成锚点，噪声远大于信号
# （实测扫描：58 候选里九成是这类假阳性）。
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("product_code", re.compile(r"\b[A-Z]{1,5}[0-9]{2,4}[A-Za-z]{0,3}\b")),   # H100, GH200, RSP200DC
    ("version", re.compile(r"\bv\d+\.\d+(?:\.\d+)?\b")),                      # v2.5, v1.19.0（v 前缀强制）
    ("doi", re.compile(r"\b10\.\d{4,9}/[-._;/:A-Za-z0-9]+")),
    # 大小写严格匹配（不用 re.I）+ 单空格：避免匹配到期刊页面"article\n\n2456 Accesses"
    # 这类跨行噪声——那是"文章"泛指，不是法条引用。
    ("legal_en", re.compile(r"\bArticle [0-9]{1,3}(?:\([0-9]{1,2}\))?\b")),
    ("legal_zh", re.compile(r"第\s*\d+\s*条")),
]
# URL/引用文献/基金编号噪声：命中词附近这些 token 说明是链接、图片路径、参考文献编号、
# 论文基金资助号，不是正文里的产品/型号事实。
_NOISE_CTX = re.compile(
    r"http|www\.|\.com|\.png|\.svg|\.jpg|\.jpeg|cdn\.|base64|%[0-9A-Fa-f]{2}"
    r"|#ref|#B\d|PMC\d|et al|\[B\d|zhida_s|images/mobile"
    r"|Fund|Grant|Foundation|基金|资助|项目编号|Scholar",
    re.I,
)
_CTX_WINDOW = 50


# 人工审核剔除：正则命中但不是真正的"精确锚点"事实——引用编号残片（#ref-CRxx）、
# 付费墙/样板页面、原始数据表格行、纯新闻标题堆砌。逐条核对过上下文见
# scratchpad 审核记录，这里直接按 cid 拉黑，不靠更复杂的正则去猜。
_BLACKLIST_CIDS = {
    "src_golden_01_0011#1",   # ScienceDirect 付费墙页面，正文只剩 "Purchase PDF" 样板
    "src_golden_06_0006#37",  # 引用编号残片 "f-CR45 ..."
    "src_golden_06_0007#15",  # "EC95" 是 "Directive 95/46/EC" 被切分错位，非真实型号
    "src_golden_08_0007#11",  # 原始数据表格行片段，非自然语言正文
    "src_golden_08_0007#12",  # 同上，另一行数据表格片段（P2310/P393）
    "src_golden_06_0006#199", # Google Scholar 引用链接的 URL 查询串片段（"F13&volume=..."）
    "src_golden_10_0015#9",   # 新闻标题堆砌（"最新资讯" 列表），非论述正文
    "src_golden_12_0003#65",  # 引用编号残片 "f-CR112 ..."
    "src_golden_12_0010#61",  # 引用编号残片 "f-CR24 ..."
}


def _cid(c: dict) -> str:
    return f"{c.get('source_id', '')}#{c.get('chunk_index', 0)}"


# task08（电商推荐系统）语料主体是匿名客户/商品 ID 原始数据表（C1/P25 这类单字母+数字
# 编号），product_code 正则在这里几乎全是假阳性、且花样多到没法用通用规则一一排除
# （表格样式不统一：有的用 "|" 分隔，有的用换行/项目符号）——审核后判定该任务本身
# 缺真实"产品型号"内容，直接对这个任务关掉 product_code 识别，DOI 类锚点不受影响。
_SKIP_PRODUCT_CODE_TASKS = {"08"}


def _extract_anchors(text: str, task_id: str = "") -> dict[str, set[str]]:
    # 曾试过"chunk 里 product_code 命中数≥N 判定为表格转储"的通用阈值，但会连坐真实的
    # 多型号对比段落（如 "V100/A100/H100 ... MI100/MI200/MI300" 一句话提 6 个型号，
    # 阈值 5 就会把这条最典型的正例也排除）——放弃通用阈值，改成 task08 那样按任务
    # 特征精确关闭（该任务的"表格"实为客户/商品 ID 转储，与产品型号语义完全不同）。
    found: dict[str, set[str]] = {}
    for name, pat in _PATTERNS:
        if name == "product_code" and task_id in _SKIP_PRODUCT_CODE_TASKS:
            continue
        for m in pat.finditer(text):
            term = m.group(0).rstrip(").,]、，。")
            if name == "product_code" and not re.search(r"\d", term):
                continue
            # Nature/Springer 页面的脚注引用锚点固定形如 "CR<数字>"（href="#ref-CR45"
            # 残片），审核时连续在多任务里发现同一形态、零一条是真实型号，按形态整体排除。
            if name == "product_code" and re.fullmatch(r"CR\d{1,4}", term):
                continue
            lo, hi = max(0, m.start() - _CTX_WINDOW), min(len(text), m.end() + _CTX_WINDOW)
            if name in ("product_code", "version") and _NOISE_CTX.search(text[lo:hi]):
                continue
            found.setdefault(name, set()).add(term)
    return found


def scan_task(task: dict, existing_gold: set[str], per_task: int) -> list[dict]:
    """返回该任务的候选 chunk 列表：[{cid, chunk, anchors, url}]，按稀有度+来源多样性挑选。"""
    corpus = task["corpus"]
    task_id = task.get("id", "")
    # 全语料锚点 → chunk 出现次数统计（判断"稀有"）
    term_chunks: dict[str, set[str]] = {}
    chunk_anchors: dict[str, dict[str, set[str]]] = {}
    for c in corpus:
        text = c.get("text", "")
        if len(text) < MIN_CHUNK_LEN:
            continue
        anchors = _extract_anchors(text, task_id)
        if not anchors:
            continue
        cid = _cid(c)
        chunk_anchors[cid] = anchors
        for terms in anchors.values():
            for t in terms:
                term_chunks.setdefault(t.lower(), set()).add(cid)

    candidates = []
    for c in corpus:
        cid = _cid(c)
        if cid not in chunk_anchors or cid in existing_gold or cid in _BLACKLIST_CIDS:
            continue
        anchors = chunk_anchors[cid]
        # 只保留本任务语料里唯一出现（occurrence==1）的锚点词，避免同一型号/DOI 落在
        # 两个不同 chunk 里、question 却只能标一个 gold（重犯已记录过的单 gold 假象）。
        unique_terms = sorted({
            t for terms in anchors.values() for t in terms
            if len(term_chunks[t.lower()]) <= MAX_OCCURRENCE
        })
        if not unique_terms:
            continue
        candidates.append({
            "cid": cid, "url": c.get("url", ""), "title": c.get("title", ""),
            "text": c.get("text", ""), "anchor_terms": unique_terms, "rarity": 1,
        })

    candidates.sort(key=lambda x: x["rarity"])
    picked, per_source = [], {}
    for cand in candidates:
        if per_source.get(cand["url"], 0) >= MAX_PER_SOURCE:
            continue
        per_source[cand["url"]] = per_source.get(cand["url"], 0) + 1
        picked.append(cand)
        if len(picked) >= per_task:
            break
    return picked


_ANCHOR_GEN_SYS = "你是检索评测数据集构造助手，负责根据给定文本片段和指定关键词生成检索问题。"
_ANCHOR_GEN_TMPL = """根据下面的文本片段，生成一个中文问题，要求：
1. 答案必须明确、完整地出自该片段中的具体事实/数据/结论，不能依赖片段外的常识；
2. 问题里必须原样包含下面"指定关键词"里的至少一个（一字不改地保留，这是硬性要求，
   不能替换成别的说法、不能省略、不能只写中文释义不写原词）；
3. 除了保留指定关键词，其余部分要像真实用户的检索意图，自然提问，不要整句照抄原文；
4. 只输出问题本身一行，不要任何前缀、编号、引号或解释。

指定关键词（至少包含一个）：{terms}

文本片段：
{text}"""


async def _gen_anchor_question(llm, chunk: dict, anchor_terms: list[str]) -> dict | None:
    """锚点问题生成——与 build_golden_set._gen_question 的区别：强制问题里保留至少一个
    指定锚点词（原样出现），不能让"不要照抄原句"这条通用规则把锚点词也一起改写掉——
    那样生成出来的问题就没有 lexical anchor 了，起不到扩充"精确字面匹配"样本的作用。
    """
    text = chunk.get("text", "")[:1500]
    terms_str = "、".join(anchor_terms[:3])

    def _norm(s: str) -> str:
        return re.sub(r"[\s\-_.]", "", s).lower()

    norm_terms = [_norm(t) for t in anchor_terms]

    async def _one_try() -> str | None:
        try:
            out = await llm.chat(
                messages=[
                    {"role": "system", "content": _ANCHOR_GEN_SYS},
                    {"role": "user", "content": _ANCHOR_GEN_TMPL.format(terms=terms_str, text=text)},
                ],
                temperature=0.4, max_tokens=4096,
            )
        except Exception as exc:
            print(f"    锚点问题生成失败 {_cid(chunk)}: {exc}", file=sys.stderr)
            return None
        lines = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
        if not lines:
            return None
        q = lines[0].strip('"""「」 ').strip()
        if len(q) < 6 or q.startswith("无法"):
            return None
        return q

    for _attempt in range(2):   # 第一次没带上锚点词，重试一次，仍失败则放弃这条候选
        q = await _one_try()
        if q and any(nt in _norm(q) for nt in norm_terms if nt):
            return {"question": q, "gold_cids": [_cid(chunk)], "src_cid": _cid(chunk)}
    print(f"    锚点词始终未出现在问题里，放弃: {_cid(chunk)}", file=sys.stderr)
    return None


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", action="store_true", help="只扫描候选，不调 LLM、不写回")
    parser.add_argument("--per-task", type=int, default=PER_TASK_DEFAULT)
    args = parser.parse_args()

    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    print("=" * 88)
    print(f"  build_golden_set_lexical —— 精确锚点候选扫描（per_task={args.per_task}）")
    print("=" * 88)

    all_candidates: dict[str, list[dict]] = {}
    for task in data["tasks"]:
        existing_gold = {cid for qa in task["qa"] for cid in qa["gold_cids"]}
        picked = scan_task(task, existing_gold, args.per_task)
        all_candidates[task["id"]] = picked
        print(f"\n[{task['id']}] {task['name']}  候选 {len(picked)} 条")
        for cand in picked:
            print(f"  {cand['cid']}  锚点={cand['anchor_terms'][:5]}  rarity={cand['rarity']}")
            print(f"    {cand['text'][:80]!r}")

    total = sum(len(v) for v in all_candidates.values())
    print(f"\n候选总数: {total}（原 120 题 → 预计 {120 + total} 题）")

    if args.scan:
        return

    from app.services.llm_service import LLMService

    llm = LLMService()
    added, dropped = 0, 0
    for task in data["tasks"]:
        picked = all_candidates[task["id"]]
        if not picked:
            continue
        chunks = [{"text": c["text"], "source_id": c["cid"].rsplit("#", 1)[0],
                   "chunk_index": int(c["cid"].rsplit("#", 1)[1])} for c in picked]
        qa_raw = await asyncio.gather(
            *[_gen_anchor_question(llm, c, cand["anchor_terms"])
              for c, cand in zip(chunks, picked)],
            return_exceptions=True,
        )
        for cand, qa in zip(picked, qa_raw):
            if not isinstance(qa, dict):
                print(f"  [{task['id']}] 生成失败/放弃: {cand['cid']}", file=sys.stderr)
                dropped += 1
                continue
            qa["anchor_terms"] = cand["anchor_terms"]
            task["qa"].append(qa)
            added += 1
            print(f"  [{task['id']}] + {qa['question']}  (gold={qa['gold_cids']})")

    data["questions_per_task"] = "mixed（原 10/任务 + 锚点子集，见各 qa.anchor_terms）"
    data["lexical_anchor_added"] = added
    from datetime import datetime
    data["updated_at"] = datetime.now().isoformat()

    GOLDEN_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    total_qa = sum(len(t["qa"]) for t in data["tasks"])
    print(f"\n已写回 {GOLDEN_PATH}：新增 {added} 题（放弃 {dropped} 条锚点词始终未出现），"
          f"总计 {total_qa} 题")


if __name__ == "__main__":
    asyncio.run(main())
