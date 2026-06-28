"""Tavily raw_content（markdown）清洗 — 去除导航/footer/图墙等样板，净化检索语料。

Tavily 返回的 raw_content 是 markdown，原样入库会让两类噪声污染向量库：
  1. 链接/图片标记：`[赣](/article/list/area.html?subType=赣)`、`![](//.../flag.png)`
     —— URL 撑大字符数、污染 embedding，锚文本之外无检索价值。
  2. 整块样板：站点导航、页脚国旗/语言列表、参考文献区 —— 本体即链接堆，无正文。

策略（已在 12 任务黄金集语料上验证）：
  - 清洗：markdown → HTML → 取文本。`<a>` 天然只留锚文本丢 href，`<img>` 无文本
    自动消失，代码块/表格内容完整保留（正则做不到这点，代码块里的 `[x](y)` 会误删）。
  - 丢弃：仅当剥离标记后近乎塌缩才判样板（导航/图墙剥离后留存极低；表格/正文留存
    ~100%）。保守阈值，全语料丢 ~2%，零误杀数值表格。
"""
from __future__ import annotations

import re

import markdown
from bs4 import BeautifulSoup

_MD_EXT = ["tables", "fenced_code"]
_MULTI_NL = re.compile(r"\n{3,}")
_HSPACE = re.compile(r"[ \t]+")


def md_to_text(raw: str) -> str:
    """markdown → 纯文本：丢图片与链接 URL，保留锚文本/代码/表格正文。"""
    if not raw:
        return ""
    html = markdown.markdown(raw, extensions=_MD_EXT)
    soup = BeautifulSoup(html, "lxml")
    for img in soup.find_all("img"):
        img.decompose()
    text = soup.get_text("\n")
    text = _MULTI_NL.sub("\n\n", _HSPACE.sub(" ", text))
    return text.strip()


def is_boilerplate(raw: str, cleaned: str) -> bool:
    """剥离标记后近乎塌缩 → 判为导航/footer/图墙样板。

    raw 为原始 markdown，cleaned 为 md_to_text(raw)。导航/图墙留存率极低，
    表格/正文留存率接近 1，故用留存率 + 残留长度双阈值，保守丢弃。
    """
    if not cleaned:
        return True
    retained = len(cleaned) / len(raw) if raw else 0.0
    # 只按"塌缩"判样板：导航/图墙剥离后留存极低；短正文留存 ~100%（无标记可剥），
    # 故绝不能用独立的"残留过短"条件——那会误杀合法短正文。
    return retained < 0.25 and len(cleaned) < 150
