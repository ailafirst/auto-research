"""搜索服务 — 支持多种搜索引擎。"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.exceptions import SearchServiceError
from app.models.source import SearchResult
from app.services.academic_api import search_arxiv, search_crossref
from app.services.cache_service import get_cache

logger = logging.getLogger(__name__)

# 方案①：静态学术白名单域（Tavily include_domains 用）。命中不命中交给 reranker 自过滤，
# 不做问题类型判断。详见 docs/检索与证据.md。
ACADEMIC_ALLOWLIST: list[str] = [
    "arxiv.org", "biorxiv.org", "medrxiv.org", "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov", "nature.com",
    "science.org", "sciencedirect.com", "springer.com", "link.springer.com",
    "ieee.org", "ieeexplore.ieee.org", "acm.org", "dl.acm.org", "cell.com",
    "pnas.org", "frontiersin.org", "mdpi.com", "plos.org", "elifesciences.org",
    "jstor.org", "onlinelibrary.wiley.com", "tandfonline.com", "sagepub.com",
    "semanticscholar.org", "aps.org", "iop.org", "rsc.org", "acs.org", "oup.com",
]

# P0：域驱动检索路由（见 docs/AnySearch技术原理与项目借鉴调研.md §6 建议）。
# 原方案①对所有查询无条件触发学术双路，与问题实际领域无关，白白多打一次 Tavily 请求。
#
# science/technology 两个域下面的学术路由最初只收窄到 {"science"}，用真实 Tavily 请求
# 实测后证明是错的：本项目 Planner 的 domain 分类把「LLM推理加速」「LLM幻觉成因机制」
# 这类 AI/ML 研究型问题也归进 "technology"（而非 "science"）——但这类问题的学术白名单
# 一路命中率不低，tavily_score >= 0.5（source_evaluator 采纳阈值）的比例约 40%（如
# arxiv 论文直接命中 vLLM/PagedAttention、幻觉检测机制）。而纯工程实践类 technology
# 问题（微服务架构选型、电商推荐系统搭建）命中率经验证接近 0%，business/legal 领域
# 同样接近 0%。也就是说真正的预测变量是「是否研究/机制类问题」，不是粗粒度的 domain
# 字符串；domain 只是它的一个不完美代理。只留 science 会静默丢掉 technology 域下研究型
# 问题的真实证据，且退化到只测请求数/raw_content% 的基准测不出这类丢失。把 technology
# 一起纳入学术路由是当前证据支持的最小修正：不完美（仍会为工程类 technology 问题多打
# 一次通常零命中的请求），但避免了更严重的证据缺失。更精确的方案应改用 Planner 已产出的
# dimensions/intent（如 deep_investigation + technical）做二级判断，而非仅靠 domain——
# 这需要更大样本的实测，先不做，避免用另一个同样未经验证的启发式替换当前这个。
# domain 取值见 app/prompts/planner.md（11 个枚举，随差距 #1 第二轮扩容从 7 个增至此数）。
#
# 差距 #1（垂直源覆盖，见 docs/检索路由蓝图.md §5）：P0 v2 确认 business/legal/policy
# 三个域收益为零，根因是当时唯一的垂直源（学术白名单）本就不该覆盖这三个域——不是路由
# 判断错了，是候选池里压根没有对应的垂直源。这里比照 ACADEMIC_ALLOWLIST 的路数，给这三个
# 域各配一份权威域名白名单，域名选取原则是「一手信源优先于媒体转述」：监管/立法机构、
# 国际组织、交易所披露、判例库，而非新闻网站首页（新闻网站命中率高但多为二手转述，
# 本身已经在 general 路里能搜到，垂直路的增量价值在于 general 路搜不到的一手文献）。
#
# 是否真的有增量证据、还是重复 §3 的教训（只看请求数不看 tavily_score 命中率）——
# 上线前必须用真实请求测 business/legal/policy 各自的命中率，方法论与 science/technology
# 那次一致，不能因为「域名列表看起来权威」就跳过实测。结果记录见 docs/检索路由蓝图.md。
BUSINESS_ALLOWLIST: list[str] = [
    "sec.gov", "federalreserve.gov", "worldbank.org", "imf.org", "oecd.org",
    "bis.org", "wto.org", "statista.com", "crunchbase.com", "hbr.org",
    "mckinsey.com", "bloomberg.com", "reuters.com", "ft.com", "wsj.com",
]

LEGAL_ALLOWLIST: list[str] = [
    "eur-lex.europa.eu", "gdpr-info.eu", "law.cornell.edu", "justia.com",
    "courtlistener.com", "supremecourt.gov", "uscourts.gov", "wipo.int",
    "echr.coe.int", "un.org", "oyez.org", "congress.gov", "legislation.gov.uk",
]

POLICY_ALLOWLIST: list[str] = [
    "oecd.org", "worldbank.org", "imf.org", "un.org", "iea.org", "irena.org",
    "epa.gov", "brookings.edu", "rand.org", "cfr.org", "chathamhouse.org",
    "iisd.org", "unfccc.int",
]

# education 域最初测出命中率仅 25%（4 类代表性问题 3 类 0 命中）判定不达标暂不接入，
# 但那一版验证只测了中文查询。复核后发现同一批问题换成英文查询命中率跳到 73%（11/15），
# 比直接复用 ACADEMIC_ALLOWLIST 还高（47%，说明 eric.ed.gov 等教育研究专属源确实比泛学术
# 源更对口）——25% 的低命中率主要是语言不匹配的假象，不是这个域没有一手信源，也不是白名单
# 选错了类型。见 docs/检索路由蓝图.md §9.4。
# 按用户要求补到 10 个规模（本轮不测命中率，官方机构优先于第三方媒体/平台）：
# ies.ed.gov（美国教育科学研究所，官方）、britishcouncil.org（英国文化教育协会，半官方）。
EDUCATION_ALLOWLIST: list[str] = [
    "eric.ed.gov", "unesco.org", "ed.gov", "nces.ed.gov",
    "oecd.org", "edutopia.org", "brookings.edu", "rand.org",
    "ies.ed.gov", "britishcouncil.org",
]

# 本项目最终面向中文母语用户——"中文查询命中率低"不能止步于"跳过中文换英文变体"，
# 那对英文查询本来就搜不到的中国本土话题（如中国法规执行细则、国内产业政策）完全无效：
# 换个英文说法照样查不到 npc.gov.cn/发改委原文，不是语言问题，是白名单里压根没有对应国别
# 的一手源。改用中国官方信源白名单重测同一批中文查询（不改查询本身，只换白名单）：
# legal 100%（10/10，cac.gov.cn/moj.gov.cn/court.gov.cn 直接命中执法案例与法规原文）、
# policy 100%（10/10，ndrc.gov.cn/mofcom.gov.cn）、education 100%（5/5，moe.gov.cn）、
# business 50%（5/10——统计局数据类查询 5/5，但"AI芯片出口管制"这类查询命中率 0，
# 因为 csrc.gov.cn 是证券监管机构，不是产业政策权威源，白名单本身还可以再细化，
# 不是语言问题）。全面高于同一批查询打西方白名单的命中率（33%~40%，见 §7.3），
# 也远高于中文查询打纯英文语料库的 9%（见下方 LANGUAGE_GATED_ROUTES 注释）。
# 结论：中文查询本身没有"先天劣势"，之前测出的低命中率是白名单国别选错了，不是语言本身
# 的问题——纯英文语料库（academic）才是语言本身导致命中率低的情形，两者原因不同，
# 不能用同一个"跳过"策略处理。见 docs/检索路由蓝图.md §9.5。
# 第二轮规模扩容（见 docs/检索路由蓝图.md §9.8）：用真实 Tavily 请求测了一批新候选，
# 逐个域名判定去留，不是整批加入——spp.gov.cn（最高检）4/5、12348.gov.cn（中国法律
# 服务网）5/5，均达标收录；mps.gov.cn（公安部）本轮查询里从未出现在结果中，未验证，
# 暂不收录。
# 补到 10 个规模：mps.gov.cn（公安部，官方，此前未搜到结果，未做命中率验证，按
# 用户要求先补充规模）。
LEGAL_ALLOWLIST_ZH: list[str] = [
    "gov.cn", "npc.gov.cn", "flk.npc.gov.cn", "court.gov.cn",
    "wenshu.court.gov.cn", "cac.gov.cn", "moj.gov.cn",
    "spp.gov.cn", "12348.gov.cn", "mps.gov.cn",
]

# 第二轮扩容：mof.gov.cn（财政部）5/5、moa.gov.cn（农业农村部）5/5、nea.gov.cn（国家
# 能源局）5/5，全部达标收录；mohrss.gov.cn（人社部）本轮未在结果中出现，未验证，暂不
# 收录。见 docs/检索路由蓝图.md §9.8。
# 补到 10 个规模：mohrss.gov.cn（人社部，官方，此前未搜到结果，未做命中率验证）。
POLICY_ALLOWLIST_ZH: list[str] = [
    "gov.cn", "ndrc.gov.cn", "stats.gov.cn", "mee.gov.cn",
    "miit.gov.cn", "mofcom.gov.cn", "mof.gov.cn", "moa.gov.cn", "nea.gov.cn",
    "mohrss.gov.cn",
]

# 第二轮扩容：sse.com.cn（上交所）5/5、sasac.gov.cn（国资委）5/5，达标收录；
# samr.gov.cn（市场监管总局）0/5 明确不达标，未收录；safe.gov.cn/nfra.gov.cn/
# gsxt.gov.cn/szse.cn 本轮查询未搜到结果，未验证，暂不收录。见 docs/检索路由蓝图.md §9.8。
# 补到 10 个规模：safe.gov.cn（外汇管理局）、nfra.gov.cn（金融监管总局）、
# szse.cn（深交所），均为官方机构，此前未搜到结果，未做命中率验证。
BUSINESS_ALLOWLIST_ZH: list[str] = [
    "stats.gov.cn", "csrc.gov.cn", "pbc.gov.cn", "miit.gov.cn", "customs.gov.cn",
    "sse.com.cn", "sasac.gov.cn", "safe.gov.cn", "nfra.gov.cn", "szse.cn",
]

# 最初只有 moe.gov.cn 一个域名，命中率虽有 100%，但样本全部来自同一个域名，是单点
# 故障，不构成"规模"。补测教育部考试中心/学信网后扩到 3 个域，命中率仍有 93%（14/15，
# neea.edu.cn/chsi.com.cn 本轮样本里没测出命中，但作为教育部下属官方机构予以保留，
# 换更多真实查询时再观察其贡献）。见 docs/检索路由蓝图.md §9.6。
# 第二轮扩容：shanghairanking.cn（软科中国大学排名）5/5、icourse163.org（中国大学
# MOOC）4/4、xuetangx.com（学堂在线）1/1，均达标收录，从"仅官方机构"扩展到"官方+
# 权威第三方教育平台"；eol.cn/hie.edu.cn/nies.net.cn 本轮未搜到结果，未验证，暂不
# 收录。见 docs/检索路由蓝图.md §9.8。
# 补到 10 个规模：eol.cn（中国教育在线）、hie.edu.cn（中国高等教育学会，官方社团）、
# nies.net.cn（中国教育科学研究院，官方）、cdgdc.edu.cn（中国学位与研究生教育信息网，
# 官方），均此前未搜到结果，未做命中率验证。
EDUCATION_ALLOWLIST_ZH: list[str] = [
    "moe.gov.cn", "neea.edu.cn", "chsi.com.cn",
    "shanghairanking.cn", "icourse163.org", "xuetangx.com",
    "eol.cn", "hie.edu.cn", "nies.net.cn", "cdgdc.edu.cn",
]

# 项目最终面向中文用户，science/technology（对应 "academic" 路由）覆盖 8/12 基准任务，
# 是使用最频繁的域，此前却是唯一一个完全没有中文源的路由（0 个域名）——§9.2 测过 CNKI/
# 万方命中率仅 40% 且多为付费墙预览页，因此判定"换不出对应中文源"直接跳过。复测换一条
# 思路：不找"中文版学术论文"（本来就稀缺），改找中国官方科研机构本身发布的研究动态/
# 成果通报，命中率 80%（16/20）。中国科学院（cas.cn）和中国计算机学会（ccf.org.cn）
# 是主力贡献者；国家自然科学基金委（nsfc.gov.cn）、科技部（most.gov.cn）、CNKI/万方
# 命中率较低但仍保留——跟英文 ACADEMIC_ALLOWLIST 31 个域名里也不是每个域都同等强势
# 是一样的设计取舍：广撒网、让 reranker/source_evaluator 自过滤，不是每个域名都要求
# 单独达标。见 docs/检索路由蓝图.md §9.6。
# 第二轮扩容尝试过 cqvip.com（维普网，0/3）、cast.org.cn（中国科协，2/7=29%，低于
# security-zh 33% 的拒绝先例）——cast.org.cn 明确不达标未收录；cqvip.com 连同
# xueshu.baidu.com（百度学术，认可度高）、nstl.gov.cn（国家科技图书文献中心，官方）、
# paper.edu.cn（中国科技论文在线，教育部主管，官方）按用户要求补进规模，不看命中率，
# 官方两个优先于认可度高的两个第三方平台。见 docs/检索路由蓝图.md §9.8。
ACADEMIC_ALLOWLIST_ZH: list[str] = [
    "cas.cn", "ccf.org.cn", "nsfc.gov.cn", "most.gov.cn",
    "cnki.net", "wanfangdata.com.cn",
    "nstl.gov.cn", "paper.edu.cn", "xueshu.baidu.com", "cqvip.com",
]

# 差距 #1 第二轮扩容：对照 AnySearch 报告列出的 17 个域，之前只覆盖了 Planner 自身
# domain 分类体系里的 6 个（对应 7 个枚举中除 general 外的全部）。此轮参照 AnySearch
# 差距清单里提到的 finance/legal/health/code/ip/security，新增 health/code/ip/security
# 四个域（finance 已被 business 覆盖，legal 已有），domain 枚举需要同步从 7 扩到 11，
# 见 app/prompts/planner.md。四个新域均已用真实 Tavily 请求实测，方法论与之前一致：
#
# | 域 | EN 命中率 | ZH 命中率 |
# |---|---|---|
# | health | 100%（12/12） | 92%（11/12，nhc.gov.cn/nmpa.gov.cn/chinacdc.cn） |
# | code | 100%（12/12，几乎全部命中来自 github.com） | 见下方 CODE_ALLOWLIST_ZH（第二轮已接入） |
# | ip | 92%（11/12） | 100%（12/12，但只有 cnipa.gov.cn 一个域，规模偏薄） |
# | security | 100%（12/12） | 33%（4/12，不达标，当时未接入） |
#
# security 当时只测出 cert.org.cn 部分命中，候选的 cnvd.org.cn/cnnvd.org.cn 在结果里
# 从未出现过（要么域名本身有误，要么 Tavily 索引不到），判定不达标——第二轮换掉
# cnvd.org.cn/cnnvd.org.cn，只保留 cert.org.cn 并新增 anquanke.com（安全客）重测，
# 命中率 100%（10/10），已正式接入，见下方 SECURITY_ALLOWLIST_ZH。
# code 第二轮改用中文技术社区（cnblogs.com/csdn.net/juejin.cn）而非"官方文档中文版"
# 思路重测，命中率 90%（9/10），已接入，见下方 CODE_ALLOWLIST_ZH；此前"官方技术文档
# 本就没有中文版"的推断被证伪——中文用户搜代码问题时命中的不是官方文档的中文翻译，
# 而是中文技术社区的原创内容，跟 academic 最初"换不出中文源"判断错误是同一类教训。
#
# ip 域后被用户要求直接换掉（见 docs/检索路由蓝图.md §9.9）：cnipa.gov.cn 单点问题
# 换了 sbj.cnipa.gov.cn/pss-system.cnipa.gov.cn/ccopyright.com.cn/cnipr.com/
# iprdaily.cn 等候选源重测，中文知识产权权威源候选池本身见底（cnipr.com 测了 16 条
# 样本命中率仅 13%，两个 cnipa 子域名结果里从未出现），即使中文侧勉强凑到 3 个也远
# 达不到规模要求，遂放弃扩容 ip，改用 AnySearch 领域清单里的 environment（环境）
# 整域替换，EN/ZH 双侧首轮实测均为 100%，规模与质量都明显更好。
# 补到 10 个规模：nhs.uk（英国国民医疗服务体系，官方），此前未做命中率验证。
HEALTH_ALLOWLIST: list[str] = [
    "who.int", "cdc.gov", "nih.gov", "fda.gov", "thelancet.com",
    "nejm.org", "bmj.com", "cochranelibrary.com", "ema.europa.eu", "nhs.uk",
]

# 第二轮扩容：natcm.gov.cn（国家中医药管理局）6/6、cnsoc.org（中国营养学会）4/4，
# 均达标收录，覆盖养生/中医药类查询（原 3 个域偏西医监管，未覆盖这块）。
# 见 docs/检索路由蓝图.md §9.8。
# 补到 10 个规模：cma.org.cn（中华医学会，官方学会）、chictr.org.cn（中国临床试验
# 注册中心，官方），官方候选已见底，加 dxy.cn（丁香园）、haodf.com（好大夫在线）、
# familydoctor.com.cn（家庭医生在线）三个认可度高的消费级医疗平台补足，均未做
# 命中率验证。
HEALTH_ALLOWLIST_ZH: list[str] = [
    "nhc.gov.cn", "nmpa.gov.cn", "chinacdc.cn", "natcm.gov.cn", "cnsoc.org",
    "cma.org.cn", "chictr.org.cn", "dxy.cn", "haodf.com", "familydoctor.com.cn",
]

# 补到 10 个规模：docs.python.org/learn.microsoft.com/developer.android.com/
# pypi.org/npmjs.com 均为厂商官方文档/包索引站；w3schools.com 是认可度高的教程
# 参考站，官方优先、它排最后。均未做命中率验证。
CODE_ALLOWLIST: list[str] = [
    "github.com", "stackoverflow.com", "developer.mozilla.org", "readthedocs.io",
    "docs.python.org", "learn.microsoft.com", "developer.android.com",
    "pypi.org", "npmjs.com", "w3schools.com",
]

# 中文技术社区，非官方文档——见上方大段注释里对"code 中文侧没测"判断的纠正。
# cnblogs.com（博客园）4/5、csdn.net 1/1、juejin.cn（掘金）4/4；gitee.com/
# segmentfault.com/oschina.net 本轮"教程类"查询未搜到结果，未验证，换更贴近仓库/
# 项目托管场景的查询词可能会有不同结果，留作后续补充项。见 docs/检索路由蓝图.md §9.8。
# 补到 10 个规模：code 域没有对应的政府监管机构，"官方"退而求其次指厂商自建开发者
# 社区——developer.aliyun.com（阿里云开发者社区）优先；其余为认可度高的中文技术
# 社区：gitee.com（码云）、segmentfault.com（思否）、oschina.net（开源中国）、
# v2ex.com（V2EX）、infoq.cn（InfoQ 中文站）、51cto.com。均未做命中率验证。
CODE_ALLOWLIST_ZH: list[str] = [
    "cnblogs.com", "csdn.net", "juejin.cn",
    "developer.aliyun.com", "gitee.com", "segmentfault.com",
    "oschina.net", "v2ex.com", "infoq.cn", "51cto.com",
]

# 补到 10 个规模：enisa.europa.eu（欧盟网络安全局）、ncsc.gov.uk（英国国家网络
# 安全中心）、ic3.gov（FBI 互联网犯罪投诉中心），均为官方机构；sans.org（认可度高
# 的安全研究/培训机构）排最后。均未做命中率验证。
SECURITY_ALLOWLIST: list[str] = [
    "nvd.nist.gov", "cve.org", "cisa.gov", "first.org", "owasp.org", "mitre.org",
    "enisa.europa.eu", "ncsc.gov.uk", "ic3.gov", "sans.org",
]

# 第二轮换候选重测：cert.org.cn（国家互联网应急中心）5/5、anquanke.com（安全客）
# 5/5，均达标，正式接入（首轮 cnvd.org.cn/cnnvd.org.cn 因 Tavily 索引不到被拒绝，
# 详见上方说明）。freebuf.com/kanxue.com/nsfocus.com 本轮未搜到结果，未验证。
# 见 docs/检索路由蓝图.md §9.8。
# 补到 10 个规模：cnvd.org.cn（国家信息安全漏洞共享平台）、cnnvd.org.cn（中国国家
# 信息安全漏洞库）——此前 Tavily 结果里没出现过，这里按用户要求先补进规模，不看
# 命中率；cac.gov.cn（网信办）、mps.gov.cn（公安部）为其他域已用过的官方机构，
# 网络安全同样是其职责范围内，允许跨域复用；再补认可度高的安全厂商/社区：
# nsfocus.com（绿盟科技）、qianxin.com（奇安信）、freebuf.com（FreeBuf）、
# kanxue.com（看雪学院）。
SECURITY_ALLOWLIST_ZH: list[str] = [
    "cert.org.cn", "anquanke.com", "cnvd.org.cn", "cnnvd.org.cn",
    "cac.gov.cn", "mps.gov.cn", "nsfocus.com", "qianxin.com",
    "freebuf.com", "kanxue.com",
]

# environment（环境）：替换掉规模上不去的 ip 域，见上方说明与 docs/检索路由蓝图.md
# §9.9。EN/ZH 双侧首轮实测均 100%：EN 侧 epa.gov/unep.org/unfccc.int/wri.org/
# noaa.gov/iea.org 全部命中（ipcc.ch/irena.org 本轮查询未搜到结果，作为公认权威机构
# 予以保留，参照 ACADEMIC_ALLOWLIST"广撒网"的既有设计取舍）；ZH 侧 mee.gov.cn（生态
# 环境部）12/12、ndrc.gov.cn（发改委，气候口）3/3，craes.org.cn/cnemc.cn/
# ccchina.org.cn 是同类官方/半官方机构，本轮未搜到结果但予以保留，理由同上。
# 补到 10 个规模：eea.europa.eu（欧洲环境署，官方）、worldwildlife.org（WWF，
# 认可度高的环保组织），未做命中率验证。
ENVIRONMENT_ALLOWLIST: list[str] = [
    "epa.gov", "unep.org", "ipcc.ch", "unfccc.int",
    "noaa.gov", "wri.org", "iea.org", "irena.org",
    "eea.europa.eu", "worldwildlife.org",
]

# 补到 10 个规模：forestry.gov.cn（国家林业和草原局）、mnr.gov.cn（自然资源部）、
# mwr.gov.cn（水利部）、acca21.org.cn（中国 21 世纪议程管理中心，气候变化官方口）
# 均为官方机构；greenpeace.org.cn（绿色和平中国，认可度高的环保组织）排最后。
# 均未做命中率验证。
ENVIRONMENT_ALLOWLIST_ZH: list[str] = [
    "mee.gov.cn", "ndrc.gov.cn", "craes.org.cn", "cnemc.cn", "ccchina.org.cn",
    "forestry.gov.cn", "mnr.gov.cn", "mwr.gov.cn", "acca21.org.cn",
    "greenpeace.org.cn",
]

# 域 → (路由名, 白名单) 的统一映射，取代逐域名硬编码的 if/elif。
# science/technology 沿用已实测确认的学术白名单；business/legal/policy/education/
# health/code/environment/security 均已实测确认（见上方各常量注释）。domain 命中
# 此表即触发对应垂直双路，未命中的域（如 general）维持只走 general 路，不做无依据的
# 猜测性扩展。ip 已被 environment 替换，见上方 ENVIRONMENT_ALLOWLIST 注释与
# docs/检索路由蓝图.md §9.9。
VERTICAL_ROUTES: dict[str, tuple[str, list[str]]] = {
    "science": ("academic", ACADEMIC_ALLOWLIST),
    "technology": ("academic", ACADEMIC_ALLOWLIST),
    "business": ("business", BUSINESS_ALLOWLIST),
    "legal": ("legal", LEGAL_ALLOWLIST),
    "policy": ("policy", POLICY_ALLOWLIST),
    "education": ("education", EDUCATION_ALLOWLIST),
    "health": ("health", HEALTH_ALLOWLIST),
    "code": ("code", CODE_ALLOWLIST),
    "environment": ("environment", ENVIRONMENT_ALLOWLIST),
    "security": ("security", SECURITY_ALLOWLIST),
}

# 路由名 → 中文官方源白名单，仅覆盖已实测验证过的路由（见上方 *_ALLOWLIST_ZH 注释）。
# search_vertical_dual() 里，中文查询命中这张表时换用对应中文白名单，而不是沿用
# 面向英文查询设计的主白名单，也不是直接跳过。第二轮扩容后 code/security 也已补齐
# 中文源，全部 9 个垂直路由（除 general）目前都有对应中文白名单，LANGUAGE_GATED_ROUTES
# 暂时清空，留作未来若出现"确认换不出中文源"的新路由时使用。
VERTICAL_ROUTES_ZH: dict[str, list[str]] = {
    "academic": ACADEMIC_ALLOWLIST_ZH,
    "legal": LEGAL_ALLOWLIST_ZH,
    "policy": POLICY_ALLOWLIST_ZH,
    "business": BUSINESS_ALLOWLIST_ZH,
    "education": EDUCATION_ALLOWLIST_ZH,
    "health": HEALTH_ALLOWLIST_ZH,
    "code": CODE_ALLOWLIST_ZH,
    "security": SECURITY_ALLOWLIST_ZH,
    "environment": ENVIRONMENT_ALLOWLIST_ZH,
}

# 差距 #4（结构化子域参数契约，见 docs/检索路由蓝图.md §9.15/§9.16）：route_name →
# 结构化数据源查询函数列表，与上面两张"域名白名单"表不是一回事——命中这张表时，除了
# Tavily 通用路 + 白名单垂直路，还会并发查每一个真正返回结构化字段（DOI/被引数/期刊/
# 年份）的数据源。值是列表而非单个函数：academic 路由下 CrossRef（已正式发表、带 DOI
# 的文献）和 arXiv（预印本，覆盖 CrossRef 索引滞后的前沿研究）是互补关系，不是二选一，
# 命中同一路由时两个都查。其余路由没有对应的免费结构化数据源，不强行凑数。只对非中文
# 查询生效（CrossRef/arXiv 都是英文文献库，中文查询命中率结构性趋近于 0，同
# ACADEMIC_ALLOWLIST 早期未换源前的问题一样）。
STRUCTURED_ROUTE_PROVIDERS: dict[str, list[Callable[[str, int], Awaitable[list[SearchResult]]]]] = {
    "academic": [search_crossref, search_arxiv],
}

# 差距 #2（路由粒度，见 docs/检索路由蓝图.md §9）：最初假设是按子问题的"研究/机制类 vs
# 工程实践类"（曾加过 Planner 输出字段 evidence_lens 验证，命中率 53% vs 53%，完全没有
# 区分度，已回退该字段）。真正验证成立的是查询语言：用 Planner 实际生成的英文/中文查询
# 分别打 ACADEMIC_ALLOWLIST，英文查询命中率 92%（47/51），中文查询只有 9%（4/45）——
# arxiv/IEEE/ACM/Nature 等本身就是纯英文语料，中文查询在这类源上天然吃亏。这一度被
# 当成"academic 路由换不出中文源，只能跳过"的理由，但 ACADEMIC_ALLOWLIST_ZH 证明这个
# 判断是错的：换个思路（不找中文版论文，找中国官方科研机构的研究通报）后命中率有 80%。
# code/security 曾是仅有的两个跳过策略路由，第二轮补测出合格中文源后已从
# LANGUAGE_GATED_ROUTES 移除、改走 VERTICAL_ROUTES_ZH 的换源分支（见上方两常量注释）。
# 该集合当前为空，保留机制供未来出现新路由且中文候选确认不达标时使用。
_CJK_RE = re.compile(r"[一-鿿]")
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")


def _is_cjk_majority(query: str) -> bool:
    return len(_CJK_RE.findall(query)) > len(_ASCII_LETTER_RE.findall(query))


LANGUAGE_GATED_ROUTES: frozenset[str] = frozenset()

# 差距 #3（跨源融合排序，见 docs/检索路由蓝图.md §9.15）：AnySearch 声称按
# authority/diversity/freshness 做 fusion ranking，但融合公式未公开
# （docs/AnySearch技术原理与项目借鉴调研.md §4.3），没法照抄，只能按该调研报告 §6
# 的建议自己定义一版可解释的融合评分，替代原来"通用路优先、按 URL 去重先到先得"的
# 拼接方式。三个维度都用已有数据算，不引入新的评分服务或额外请求：
#   authority：CrossRef 结果用 log1p(被引数) 归一化到 (0,1]；命中垂直白名单的固定给
#     0.75（白名单本身就是人工审核过的权威源），arXiv 结果没有被引数字段，同样落进
#     这一档——预印本未经同行评审，理论上该打折扣，但没有真实标注数据支撑具体折多少，
#     如实按经验默认值处理，不假装算出来的数字有精确依据（见 §9.16）；通用路退回
#     Tavily 自己的 tavily_score，没有分数时给中性 0.5。
#   diversity：同一域名（netloc）在本次候选集里出现次数越多，分数衰减越多——用全局
#     计数做静态惩罚（1/出现次数），不是逐条重排的 MMR，避免大量结果堆在同一个网站。
#   freshness：有发表年份的按距今年数线性衰减，10 年及更早记 0 分下限；没有年份信息
#     的给中性分 0.5，不因为"缺数据"就判定过时。
# 权重 0.5/0.3/0.2（authority 主导）是本项目定的合理默认值，没有可比对的标注排序数据集
# 去拟合最优权重，如实记录为经验取值，不是像白名单命中率那样有真实请求验证过的数字。
_AUTHORITY_WEIGHT = 0.5
_DIVERSITY_WEIGHT = 0.3
_FRESHNESS_WEIGHT = 0.2
_CITATION_LOG_CAP = math.log1p(1000)  # log1p(1000 次被引) 记满分，压缩长尾高被引论文


def _authority_score(r: SearchResult) -> float:
    if r.provider == "crossref" and r.citation_count is not None:
        return min(1.0, math.log1p(max(0, r.citation_count)) / _CITATION_LOG_CAP)
    if r.search_route != "general":
        return 0.75
    return r.tavily_score if r.tavily_score is not None else 0.5


def _freshness_score(r: SearchResult) -> float:
    if not r.published_year:
        return 0.5
    age = max(0, datetime.now().year - r.published_year)
    return max(0.0, 1 - age / 10)


def _diversity_key(r: SearchResult) -> str:
    """diversity 分组键：默认用域名（netloc）。CrossRef 结果的 URL 统一是
    `doi.org/<DOI>`，netloc 全部收敛成同一个 "doi.org"——如果直接用 netloc 分组，
    会把 5 篇分属不同期刊/出版社的论文误判成"同一个来源重复出现"，集体拖累 diversity
    分数（实测发现，见 docs/检索路由蓝图.md §9.15）。改用 DOI 的 registrant 前缀
    （如 "10.3389" 对应 Frontiers）做分组键——同一出版社/期刊反复出现时仍会被
    diversity 惩罚（这是期望行为），不同出版社的论文则各自算作独立来源。
    """
    if r.provider == "crossref" and r.doi and "/" in r.doi:
        return f"doi:{r.doi.split('/', 1)[0]}"
    return urlparse(r.url).netloc


# 跨 provider 去重（见 docs/检索路由蓝图.md §9.16 已知局限 #1）：同一篇论文经 Tavily
# 爬到的出版商/arXiv 页面 URL，跟 CrossRef/arXiv API 返回的规范 URL 形式不同（协议头、
# 版本号后缀、域名跳转都可能不一样），原始 URL 精确匹配抓不住这类重复。业界通用做法
# （OpenAlex/Semantic Scholar/文献管理软件合并同一篇论文的多来源记录）是按 DOI/arXiv ID
# 这类跨来源稳定的规范标识符做实体消歧，不是字符串去重——这里照此思路抽取，两者都没有
# 时才退回原始 URL，与改动前行为一致。
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+")
# (?:abs|pdf|html) 而不是只有 abs/pdf——真实数据测出 Tavily 会爬到 ar5iv.labs.arxiv.org/html/<id>
# 这个官方 arXiv Labs 的 HTML 渲染镜像（域名里仍含 "arxiv.org" 子串），只认 abs/pdf 会漏掉这类命中。
_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf|html)/([a-z-]+/\d{7}|\d{4}\.\d{4,5})", re.IGNORECASE)


def _extract_doi(r: SearchResult) -> str | None:
    """优先用结构化字段（CrossRef 直接填了 doi），字段为空时退而从 URL 里正则抽取
    （覆盖 Tavily 爬到的出版商页面 URL 里带 DOI 路径的情况）。"""
    if r.doi:
        return r.doi.strip().lower()
    m = _DOI_RE.search(r.url)
    return m.group(0).rstrip("/.,)").lower() if m else None


def _extract_arxiv_id(r: SearchResult) -> str | None:
    """从 URL 里抽取 arXiv ID 并去掉版本号后缀（v1/v2…）——同一篇论文的不同版本
    该算同一个来源，不是 diversity 意义上的"新论文"。"""
    m = _ARXIV_ID_RE.search(r.url)
    return m.group(1).lower() if m else None


def _canonical_key(r: SearchResult) -> str:
    doi = _extract_doi(r)
    if doi:
        return f"doi:{doi}"
    arxiv_id = _extract_arxiv_id(r)
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    return r.url


def _record_priority(r: SearchResult) -> int:
    """同一篇论文被去重时保留哪一份的优先级：结构化数据源（有 DOI/被引数/年份等
    真实字段）> 命中垂直白名单的通用搜索（人工审核过的权威域名，但仍是网页摘要）>
    纯通用搜索。优先级相同时保留先出现的那份（沿用改动前"先到先得"的语义）。"""
    if r.provider in ("crossref", "arxiv"):
        return 2
    if r.search_route != "general":
        return 1
    return 0


def _fuse_and_rank(candidates: list[SearchResult]) -> list[SearchResult]:
    """按规范标识符去重（同一篇论文保留结构化信号最丰富的那份），再按可解释融合分排序。"""
    by_key: dict[str, SearchResult] = {}
    for r in candidates:
        key = _canonical_key(r)
        existing = by_key.get(key)
        if existing is None or _record_priority(r) > _record_priority(existing):
            by_key[key] = r
    deduped = list(by_key.values())

    domain_counts: dict[str, int] = {}
    for r in deduped:
        key = _diversity_key(r)
        domain_counts[key] = domain_counts.get(key, 0) + 1

    def _score(r: SearchResult) -> float:
        diversity = 1.0 / domain_counts.get(_diversity_key(r), 1)
        return (
            _AUTHORITY_WEIGHT * _authority_score(r)
            + _DIVERSITY_WEIGHT * diversity
            + _FRESHNESS_WEIGHT * _freshness_score(r)
        )

    return sorted(deduped, key=_score, reverse=True)


class TavilyRateLimitError(Exception):
    """Tavily 限速错误（429）— 可被 tenacity 捕获并按指数退避重试。"""


class TavilyQuotaError(Exception):
    """Tavily key 失效或月配额耗尽（432/403/401）— 需要切换到下一个 key。"""

    def __init__(self, msg: str, key_idx: int = 0) -> None:
        super().__init__(msg)
        self.key_idx = key_idx


def _load_tavily_keys() -> list[str]:
    """加载 Tavily key 轮换池：项目根目录 tavily.txt + .env 里的 TAVILY_API_KEY。

    环境变量那个也要并进池子，否则 probe_key_pool 这类只读池的调用方会认为「一个
    key 都没有」而直接判定耗尽——容器化部署里 tavily.txt 不进镜像（含密钥，被
    .dockerignore 排除），key 只从环境变量来，正是这种情况。
    """
    keys: list[str] = []

    key_file = Path(__file__).parent.parent.parent / "tavily.txt"
    if key_file.exists():
        for line in key_file.read_text(encoding="utf-8").splitlines():
            m = re.search(r"(tvly-\S+)", line.strip())
            if m:
                keys.append(m.group(1))

    env_key = (settings.tavily_api_key or "").strip()
    if env_key and env_key not in keys:
        keys.append(env_key)

    return keys


class TavilyKeyPool:
    """Tavily API key 顺序轮换池 — 额度耗尽时切换到下一个 key。"""

    def __init__(self, keys: list[str]) -> None:
        self._keys = keys
        self._idx = 0
        if keys:
            logger.info("Tavily key pool 已加载 %d 个 key", len(keys))

    @property
    def current_key(self) -> str | None:
        return self._keys[self._idx] if self._idx < len(self._keys) else None

    @property
    def current_idx(self) -> int:
        return self._idx

    def rotate(self, from_idx: int) -> str | None:
        """将 from_idx 处的 key 标记为耗尽并切换到下一个。
        若 _idx 已超过 from_idx（并发场景下其他协程已先轮换），则幂等跳过。
        """
        if self._idx != from_idx:
            return self.current_key
        self._idx += 1
        if self._idx < len(self._keys):
            logger.warning(
                "Tavily key %d 额度耗尽，切换至 key %d/%d",
                from_idx + 1, self._idx + 1, len(self._keys),
            )
            return self._keys[self._idx]
        logger.error("Tavily 所有 %d 个 key 额度已耗尽", len(self._keys))
        return None


_key_pool: TavilyKeyPool | None = None


def _get_key_pool() -> TavilyKeyPool:
    global _key_pool
    if _key_pool is None:
        _key_pool = TavilyKeyPool(_load_tavily_keys())
    return _key_pool


class BaseSearchProvider:
    """搜索提供者基类。"""

    async def search(self, query: str, max_results: int = 5,
                     include_domains: list[str] | None = None) -> list[SearchResult]:
        raise NotImplementedError


class DuckDuckGoProvider(BaseSearchProvider):
    """DuckDuckGo 搜索（免费，适合开发环境）。"""

    async def search(self, query: str, max_results: int = 5,
                     include_domains: list[str] | None = None) -> list[SearchResult]:
        # DuckDuckGo 无域名过滤参数，include_domains 忽略（学术双路仅 Tavily 生效）
        try:
            from ddgs import DDGS

            def _search() -> list[dict[str, Any]]:
                ddgs = DDGS()
                return list(ddgs.text(query=query, max_results=max_results))

            raw_results = await asyncio.to_thread(_search)

            results: list[SearchResult] = []
            for i, item in enumerate(raw_results):
                title = item.get("title", "")
                href = item.get("href", "")
                snippet = item.get("body", "")

                if not href:
                    continue
                if href.startswith("//"):
                    href = "https:" + href

                results.append(SearchResult(
                    title=title or "无标题",
                    url=href,
                    snippet=snippet or "",
                    position=i + 1,
                    provider="duckduckgo",
                ))

            return results

        except ImportError:
            logger.warning("duckduckgo_search 未安装，请执行: pip install duckduckgo_search")
            return []
        except Exception as exc:
            logger.warning("DuckDuckGo 搜索失败: %s", exc)
            return []


_tavily_semaphore = asyncio.Semaphore(3)   # 限制 Tavily 全局并发，避免 dev key 限速


class TavilyProvider(BaseSearchProvider):
    """Tavily Search API 搜索，支持多 key 顺序轮换。"""

    async def search(self, query: str, max_results: int = 5,
                     include_domains: list[str] | None = None) -> list[SearchResult]:
        pool = _get_key_pool()
        # 在进入 semaphore 前记录本次请求使用的 key index，随错误一起传出
        # 这样多个并发任务失败时，rotate(key_idx) 都用同一个 from_idx，
        # 幂等检查生效，只发生一次真正的轮换，避免级联耗尽所有 key。
        key_idx = pool.current_idx
        api_key = pool.current_key or settings.tavily_api_key
        if not api_key:
            logger.warning("Tavily API Key 未配置（tavily.txt 和 .env 均无有效 key）")
            return []

        async with _tavily_semaphore:
            try:
                from tavily import AsyncTavilyClient

                client = AsyncTavilyClient(api_key=api_key)
                search_kwargs: dict[str, Any] = dict(
                    query=query,
                    max_results=max_results,
                    search_depth=settings.tavily_search_depth,
                    include_raw_content=settings.tavily_include_raw_content,
                    include_answer=True,
                )
                if include_domains:
                    search_kwargs["include_domains"] = include_domains
                response = await client.search(**search_kwargs)

                answer: str | None = response.get("answer") or None

                results: list[SearchResult] = []
                for i, item in enumerate(response.get("results", [])):
                    results.append(SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=item.get("content", ""),
                        position=i + 1,
                        raw_content=item.get("raw_content") or None,
                        tavily_score=item.get("score"),
                        query_answer=answer,
                        provider="tavily",
                    ))

                return results

            except Exception as exc:
                # Tavily 库的异常映射：
                #   429 → UsageLimitExceededError（限速，应重试）
                #   432/403 → ForbiddenError（key 失效或月配额耗尽，应换 key）
                #   401 → InvalidAPIKeyError（key 无效，应换 key）
                from tavily.errors import (
                    ForbiddenError as _TForbidden,
                    InvalidAPIKeyError as _TInvalidKey,
                    UsageLimitExceededError as _TUsageLimit,
                )
                if isinstance(exc, _TUsageLimit):
                    raise TavilyRateLimitError(str(exc)) from exc
                if isinstance(exc, (_TForbidden, _TInvalidKey)):
                    raise TavilyQuotaError(str(exc), key_idx) from exc
                logger.error("Tavily 搜索失败: %s", exc)
                return []
            finally:
                # 每次请求后短暂等待，避免连续触发限速
                await asyncio.sleep(0.5)


class SearchService:
    """搜索服务 — 统一入口，USE_TAVILY 开关选择引擎。"""

    def __init__(self) -> None:
        self.providers: dict[str, BaseSearchProvider] = {
            "duckduckgo": DuckDuckGoProvider(),
            "tavily": TavilyProvider(),
        }
        self._current_provider = "tavily" if settings.use_tavily else "duckduckgo"

    @property
    def provider(self) -> BaseSearchProvider:
        return self.providers[self._current_provider]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=8),
        retry=retry_if_exception_type(TavilyRateLimitError),
        reraise=True,
    )
    async def _primary_search(self, query: str, max_results: int,
                              include_domains: list[str] | None = None) -> list[SearchResult]:
        """调用主引擎，限速时由 tenacity 按指数退避自动重试（最多 3 次）。"""
        return await self.provider.search(query, max_results=max_results,
                                          include_domains=include_domains)

    async def search(self, query: str, max_results: int | None = None,
                     include_domains: list[str] | None = None,
                     search_route: str = "general") -> list[SearchResult]:
        """执行单次搜索，限速时重试，额度耗尽时轮换 key，最终失败降级 DuckDuckGo。

        命中缓存时直接返回，完全跳过 Tavily（省配额）。缓存 key 含 provider 与影响返回体
        的 Tavily 参数；只缓存非空结果，避免把瞬时失败/零结果固化。缓存的是未打
        sub_question_id 的原始结果，打标由调用方（retriever_node）在返回后进行。

        search_route 是纯元数据标注（"general"/"academic"），不参与缓存 key——它标注的是
        调用方为什么发起这次检索，不影响实际请求参数或返回体。
        """
        n_results = max_results or settings.max_search_results

        cache = get_cache()
        cache_key = cache.key(
            "search",
            self._current_provider,
            query.strip(),
            n_results,
            settings.tavily_search_depth,
            settings.tavily_include_raw_content,
            ",".join(include_domains) if include_domains else "",  # 学术双路与通用搜索缓存分离
        )
        cached = await cache.get_json(cache_key)
        if cached is not None:
            logger.info(
                "搜索缓存命中 [%s]: query='%s', results=%d",
                self._current_provider, query[:50], len(cached),
            )
            cached_results = [SearchResult(**d) for d in cached]
            for r in cached_results:
                r.search_route = search_route
            return cached_results

        pool = _get_key_pool()

        results: list[SearchResult] = []
        # 最多尝试所有可用 key 次数
        max_key_attempts = len(pool._keys) + 1
        for _ in range(max_key_attempts):
            try:
                results = await self._primary_search(query, n_results, include_domains)
                break
            except TavilyRateLimitError:
                logger.warning("Tavily 限速重试耗尽，降级 DDG: query='%s'", query[:50])
                break
            except TavilyQuotaError as exc:
                # 用 exc.key_idx（请求发出时的 index），而非 pool.current_idx，
                # 确保并发场景下多个任务失败只触发一次真正的轮换。
                new_key = pool.rotate(exc.key_idx)
                if new_key is None:
                    logger.warning("Tavily 所有 key 耗尽，降级 DDG: query='%s'", query[:50])
                    break
                # 继续循环，TavilyProvider.search() 下次调用会取 pool.current_key

        # 零结果降级：Tavily 返回空 → DuckDuckGo 补充
        if not results and self._current_provider == "tavily":
            logger.warning("Tavily 零结果，DDG 降级: query='%s'", query[:50])
            results = await self.providers["duckduckgo"].search(query, max_results=n_results)

        engine = self._current_provider if results else "duckduckgo(fallback)"
        logger.info("搜索完成 [%s]: query='%s', results=%d", engine, query[:50], len(results))

        for r in results:
            r.search_route = search_route

        # 只缓存非空结果：零结果多为瞬时失败/配额耗尽，不应被固化到 TTL 结束
        if results:
            await cache.set_json(
                cache_key,
                [r.model_dump() for r in results],
                settings.search_cache_ttl,
            )
        return results

    async def search_vertical_dual(
        self, query: str, route_name: str, allowlist: list[str],
        max_results: int | None = None,
    ) -> list[SearchResult]:
        """通用路 + 垂直白名单路（+ 命中时的结构化数据源路）并发检索，融合排序后返回。

        泛化自原「学术双路」（方案①），现覆盖 academic/business/legal/policy 等
        VERTICAL_ROUTES 里注册的任意垂直路由，逻辑不变：统一生效、不猜问题类型，
        命中不命中交给下游 reranker/source_evaluator 自过滤。降级安全：非 Tavily
        引擎或某一路失败/为空时，用其余成功的路继续。详见 docs/检索与证据.md、
        docs/检索路由蓝图.md。

        query 中文字符占多数时分两种处理：route_name 在 VERTICAL_ROUTES_ZH 里有对应的
        中文白名单（目前 9 个垂直路由全部覆盖，见该常量注释）就换用那份白名单，而不是
        沿用面向英文查询设计的主白名单——中文查询打对应国别/机构的中文源命中率经实测
        普遍在 80%~100%，远高于打西方白名单的水平。LANGUAGE_GATED_ROUTES 里的路由
        （纯跳过、不换源）留给未来找不到对应中文源、也没必要硬凑白名单的场景。

        route_name 命中 STRUCTURED_ROUTE_PROVIDERS（目前只有 academic，对应 CrossRef +
        arXiv 两个数据源）且查询非中文时，额外并发对应的结构化数据源检索（差距 #4，见
        docs/检索路由蓝图.md §9.15/§9.16）。全部结果最后统一走 `_fuse_and_rank`（差距 #3）
        按 authority/diversity/freshness 融合排序，取代原来"通用优先、按 URL 去重先到
        先得"的拼接方式。
        """
        # 域名过滤是 Tavily 特性；DDG 路直接走通用搜索
        if self._current_provider != "tavily":
            return await self.search(query, max_results=max_results)

        base_route_name = route_name  # 结构化源按原始 route_name 匹配，zh 分支会改写 route_name
        cjk = _is_cjk_majority(query)
        if cjk:
            zh_allowlist = VERTICAL_ROUTES_ZH.get(route_name)
            if zh_allowlist:
                allowlist = zh_allowlist
                route_name = f"{route_name}-zh"
            elif route_name in LANGUAGE_GATED_ROUTES:
                return await self.search(query, max_results=max_results, search_route="general")

        structured_providers = [] if cjk else STRUCTURED_ROUTE_PROVIDERS.get(base_route_name, [])
        tasks = [
            self.search(query, max_results=max_results, search_route="general"),
            self.search(query, max_results=max_results, include_domains=allowlist,
                       search_route=route_name),
        ]
        tasks += [provider(query, max_results or 5) for provider in structured_providers]

        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        general, vertical = outcomes[0], outcomes[1]
        structured_outcomes = outcomes[2:]

        candidates: list[SearchResult] = []
        if isinstance(general, Exception):
            logger.warning("垂直双路[%s]：通用路异常: %s", route_name, general)
        else:
            candidates.extend(general)
        if isinstance(vertical, Exception):
            logger.warning("垂直双路[%s]：垂直路异常: %s", route_name, vertical)
        else:
            candidates.extend(vertical)

        structured: list[SearchResult] = []
        for provider, outcome in zip(structured_providers, structured_outcomes):
            if isinstance(outcome, Exception):
                logger.warning("垂直双路[%s]：结构化路(%s)异常: %s",
                              route_name, provider.__name__, outcome)
                continue
            candidates.extend(outcome)
            structured.extend(outcome)

        merged = _fuse_and_rank(candidates)
        logger.info("垂直双路[%s] [%s]: 候选 %d 条（结构化 %d 条），融合排序后 %d 条",
                   route_name, query[:40], len(candidates), len(structured), len(merged))
        return merged

    async def search_vertical_multi(
        self, query: str, domains: list[str], max_results: int | None = None,
    ) -> list[SearchResult]:
        """通用路 + 多个 domain 各自的垂直白名单路（+ 命中时各自的结构化数据源路），
        融合排序后返回。

        AnySearch 原则②"不确定时是混合，不是二选一"（docs/AnySearch技术原理与项目借鉴调研.md
        §4.2）的落地：当 Planner 判断某个子问题同时明显跨两个 domain（`planner.md` Sub-question
        Rules #6 的 `domain_secondary`）时，两个域各自的垂直源都搜，而不是强迫只选一个——
        命中与否交给下游 reranker/source_evaluator 自过滤，跟 `search_vertical_dual` 单路由
        时的取舍一致。domains 允许传 1 个（等价于 search_vertical_dual）或 2 个；解析到同一个
        route_name（如 science/technology 都指向 academic）时只搜一次，不重复请求。domains
        为空、或全部解析不到 VERTICAL_ROUTES 条目时，退化为纯通用搜索。详见
        docs/检索路由蓝图.md §9.13。

        route_name 命中 STRUCTURED_ROUTE_PROVIDERS 且查询非中文时，该 domain 额外并发
        对应的结构化数据源检索（差距 #4，一个路由可能对应多个数据源，如 academic 的
        CrossRef + arXiv）。全部候选最后走 `_fuse_and_rank`（差距 #3）融合排序，见
        §9.15/§9.16。
        """
        if self._current_provider != "tavily":
            return await self.search(query, max_results=max_results)

        cjk = _is_cjk_majority(query)
        resolved: dict[str, list[str]] = {}
        for domain in domains:
            route = VERTICAL_ROUTES.get(domain)
            if route is None:
                continue
            route_name, allowlist = route
            if cjk:
                zh_allowlist = VERTICAL_ROUTES_ZH.get(route_name)
                if zh_allowlist:
                    route_name, allowlist = f"{route_name}-zh", zh_allowlist
                elif route_name in LANGUAGE_GATED_ROUTES:
                    continue
            resolved.setdefault(route_name, allowlist)

        if not resolved:
            return await self.search(query, max_results=max_results, search_route="general")

        route_names = list(resolved.keys())
        structured_pairs: list[tuple[str, Callable[[str, int], Awaitable[list[SearchResult]]]]] = []
        if not cjk:
            for rn in route_names:
                for provider in STRUCTURED_ROUTE_PROVIDERS.get(rn, []):
                    structured_pairs.append((rn, provider))

        tasks: list[Awaitable[list[SearchResult]]] = [
            self.search(query, max_results=max_results, search_route="general"),
        ]
        tasks += [self.search(query, max_results=max_results, include_domains=resolved[rn],
                              search_route=rn) for rn in route_names]
        tasks += [provider(query, max_results or 5) for _, provider in structured_pairs]

        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        general = outcomes[0]
        vertical_outcomes = outcomes[1:1 + len(route_names)]
        structured_outcomes = outcomes[1 + len(route_names):]

        candidates: list[SearchResult] = []
        if isinstance(general, Exception):
            logger.warning("多域混合检索[%s]：通用路异常: %s", "+".join(route_names), general)
        else:
            candidates.extend(general)

        for route_name, outcome in zip(route_names, vertical_outcomes):
            if isinstance(outcome, Exception):
                logger.warning("多域混合检索[%s]：垂直路异常: %s", route_name, outcome)
                continue
            candidates.extend(outcome)

        n_structured = 0
        for (route_name, provider), outcome in zip(structured_pairs, structured_outcomes):
            if isinstance(outcome, Exception):
                logger.warning("多域混合检索[%s]：结构化路(%s)异常: %s",
                              route_name, provider.__name__, outcome)
                continue
            candidates.extend(outcome)
            n_structured += len(outcome)

        merged = _fuse_and_rank(candidates)
        logger.info("多域混合检索[%s] [%s]: 候选 %d 条（结构化 %d 条），融合排序后 %d 条",
                   "+".join(route_names), query[:40], len(candidates), n_structured, len(merged))
        return merged

    async def probe_key_pool(self) -> None:
        """发 1 次轻量探针确认当前 Tavily key 可用；quota 耗尽则提前 rotate。

        在批量搜索前调用，消除 N 个并发请求同时撞到耗尽 key 时的级联串行重试。
        探针固定用 basic + 无 raw_content，不消耗 advanced credit。
        """
        if self._current_provider != "tavily":
            return
        pool = _get_key_pool()
        for _ in range(len(pool._keys) + 1):
            key_idx = pool.current_idx
            api_key = pool.current_key
            if not api_key:
                logger.warning("Tavily key probe: 所有 key 已耗尽")
                return
            try:
                from tavily import AsyncTavilyClient
                client = AsyncTavilyClient(api_key=api_key)
                await client.search(
                    "probe",
                    max_results=1,
                    search_depth="basic",
                    include_raw_content=False,
                    include_answer=False,
                )
                logger.info("Tavily key probe 通过: key %d/%d 可用", key_idx + 1, len(pool._keys))
                return
            except Exception as exc:
                try:
                    from tavily.errors import ForbiddenError as _F, InvalidAPIKeyError as _I
                    if isinstance(exc, (_F, _I)):
                        new_key = pool.rotate(key_idx)
                        if new_key is None:
                            return
                        continue
                except ImportError:
                    pass
                # 非 quota 错误（网络抖动/限速）—— 保留当前 key，让后续请求正常重试
                logger.warning("Tavily key probe 失败（非 quota 错误，保留当前 key）: %s", exc)
                return

    async def multi_search(
        self,
        queries: list[str],
        max_results_per_query: int = 5,
        concurrency: int | asyncio.Semaphore = 3,
    ) -> list[SearchResult]:
        """执行多关键词搜索并去重。
        concurrency 可传入 int（内部创建 Semaphore）或外部共享的 asyncio.Semaphore，
        后者用于跨多个 multi_search 调用共享全局并发上限。
        """
        semaphore = concurrency if isinstance(concurrency, asyncio.Semaphore) else asyncio.Semaphore(concurrency)

        async def _throttled(q: str) -> list[SearchResult]:
            async with semaphore:
                return await self.search(q, max_results=max_results_per_query)

        tasks = [_throttled(q) for q in queries]
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        seen_urls: set[str] = set()
        all_results: list[SearchResult] = []

        for results in results_lists:
            if isinstance(results, Exception):
                logger.warning("搜索异常: %s", results)
                continue
            for r in results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    all_results.append(r)

        return all_results
