"""BM25 稀疏向量编码 + 查询锚点探测 —— 混合检索（settings.hybrid_enabled）的 sparse 侧。

混合检索按查询自适应触发，不是全局开关：has_lexical_anchor() 判断一条查询是否
含精确标识符（型号/版本号/DOI/法条编号），只有命中的查询才在 rag_service.py 里
走 dense+sparse 加性并集，其余查询走原有纯 dense 路径，不受影响。

分词：英文/数字走正则（保留 h200 / pipl / gdpr 这类精确锚点），中文走 jieba 搜索
模式。阶段一实测 jieba 对黄金集零贡献（可捞回的案例几乎全是"中文 query→英文
gold"，靠 P-XLing 英译 + ascii 分词），保留中文分词是为覆盖"中文 query→中文
gold 精确词"这类离线基准里样本极少、线上未必罕见的场景。

权重：文档侧存 BM25 的 TF 饱和项，IDF 交给 Qdrant 的 Modifier.IDF 按整个
collection 的文档频率在线算（score = Σ_t q_tf(t) · idf(t) · doc_tf_weight(t)）。
故 encode_query 每个词权重给 1.0，不在本地乘 IDF。

token → u32 下标用 md5 前 4 字节，跨进程稳定（Python 内置 hash 有随机化，不能用）。
"""

from __future__ import annotations

import hashlib
import re

import jieba

# jieba 首次调用会打印 building prefix dict 日志，且 initialize 非线程安全——
# 模块加载时静默预热一次。
jieba.setLogLevel(60)
jieba.initialize()

_ASCII_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[一-鿿]")

# 查询侧锚点探测：判断一条子问题是否"像是在问一个精确标识符"（产品型号/版本号/
# DOI/法条编号），命中才触发 hybrid（见 has_lexical_anchor）。正则与
# benchmark/build_golden_set_lexical.py 扫描语料用的同源，但这里扫的是用户/Planner
# 写的短查询文本，不是爬来的网页 chunk，不需要那边的 URL/引用残片/表格转储过滤。
#
# 实测依据（benchmark/GoldenDataset/improvement_plan.md「黄金集扩充后复核」）：
# 生产路径验证，含这类锚点的查询 hybrid 比纯 dense@40 高 11.9pp（0.857→0.976）；
# 不含锚点的查询 hybrid 反而 −1.7pp——这正是要按查询门控、不能全局开的原因。
#
# 边界用 (?<![A-Za-z0-9]) / (?![A-Za-z0-9])，不用 \b：Python re 的 \b 按 Unicode
# \w 判断词边界，中文字符也算 \w，"INT16量化""Article 9将""DOI为10.xxx的论文"这类
# 中英literal 直接粘连（中文技术问答里极常见，不像英文正文会留空格）时 \b 在英文/
# 数字与中文交界处不成立，整条正则直接匹配失败——过 42 条锚点验证题实测：用 \b 时
# 20/42 漏检，全部是这个模式。踩过的坑记下来，别再用 \b 写这类正则。
_ANCHOR_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"(?<![A-Za-z0-9])[A-Z]{1,5}[0-9]{2,4}[A-Za-z]{0,3}(?![A-Za-z0-9])"),  # H100, GH200
    re.compile(r"(?<![A-Za-z0-9])v[0-9]+\.[0-9]+(?:\.[0-9]+)?(?![A-Za-z0-9])"),  # v2.5（v 前缀强制）
    re.compile(r"(?<![A-Za-z0-9])10\.\d{4,9}/[-._;/:A-Za-z0-9]+"),               # DOI
    re.compile(r"(?<![A-Za-z])Article [0-9]{1,3}(?:\([0-9]{1,2}\))?(?![A-Za-z0-9])"),  # Article 9
    re.compile(r"第\s*\d+\s*条"),                             # 第14条
)


def has_lexical_anchor(text: str) -> bool:
    """query 是否含精确标识符锚点——命中才对这条查询触发 hybrid 检索。"""
    if not text:
        return False
    return any(p.search(text) for p in _ANCHOR_PATTERNS)

# BM25 标准参数：k1 控制词频饱和速度，b 控制文档长度归一化强度。
_BM25_K1 = 1.5
_BM25_B = 0.75

# 下标空间。Qdrant 稀疏向量下标要求 u32，取满量程；md5 前 32 位碰撞概率对
# 单任务几千 chunk 的词表可忽略。
_INDEX_SPACE = 2**32


def _token_index(token: str) -> int:
    return int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:4], "big") % _INDEX_SPACE


def tokenize(text: str) -> list[str]:
    """分词，返回小写 token 列表（含重复，供词频统计）。"""
    if not text:
        return []
    tokens = _ASCII_RE.findall(text.lower())
    if _CJK_RE.search(text):
        for w in jieba.lcut_for_search(text):
            w = w.strip()
            if w and _CJK_RE.search(w):
                tokens.append(w)
    return tokens


class Bm25Encoder:
    """把文本编码成 Qdrant 稀疏向量 (indices, values)。无状态，进程级单例即可。"""

    @staticmethod
    def doc_length(text: str) -> int:
        return len(tokenize(text))

    def encode_document(self, text: str, avgdl: float) -> tuple[list[int], list[float]]:
        """文档侧：BM25 TF 饱和项。avgdl 为所在批次的平均文档长度（token 数）。"""
        tokens = tokenize(text)
        if not tokens:
            return [], []
        dl = len(tokens)
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        norm = _BM25_K1 * (1.0 - _BM25_B + _BM25_B * dl / avgdl) if avgdl > 0 else _BM25_K1
        idx_to_val: dict[int, float] = {}
        for token, freq in tf.items():
            weight = freq * (_BM25_K1 + 1.0) / (freq + norm)
            # 不同 token 若 md5 撞到同一下标，取较大权重（保守，不叠加）
            i = _token_index(token)
            if weight > idx_to_val.get(i, 0.0):
                idx_to_val[i] = weight
        return list(idx_to_val.keys()), list(idx_to_val.values())

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        """查询侧：每个唯一 token 权重 1.0，IDF 由 Qdrant Modifier.IDF 施加。"""
        seen: dict[int, float] = {}
        for token in set(tokenize(text)):
            seen[_token_index(token)] = 1.0
        return list(seen.keys()), list(seen.values())


_encoder_singleton: Bm25Encoder | None = None


def get_bm25_encoder() -> Bm25Encoder:
    global _encoder_singleton
    if _encoder_singleton is None:
        _encoder_singleton = Bm25Encoder()
    return _encoder_singleton
