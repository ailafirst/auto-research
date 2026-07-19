# reranker 短板成因验证：粒度 vs 相关性
> 时间: 20260718_105022 | 仅测试
> 若 best_sent(切句) 明显 > blob(整段) 且越过分数线，则短板是「粒度/格式」，应切 chunk 修输入

## [bci] 2026 brain-computer interface latest research directions worth deep investment

- web/academic top-10 分数线(cutoff) = **0.679**；OpenAlex 论文 20 篇

| 指标 | blob(整段摘要) | best_sent(切句取最高) |
|---|---|---|
| 平均分 | 0.037 | 0.049 |
| 中位数 | 0.000 | 0.001 |
| 最高分 | 0.199 | 0.554 |
| **越过分数线的论文数** | **0/20** | **0/20** |

- 提升最大的论文（blob→best_sent）：
  - 0.198→0.554  《<i>Notice of Removal March 3, 2026:</i> Domain Adaptation and Generali》
  - 0.171→0.309  《Review of deep representation learning techniques for brain–computer i》
  - 0.000→0.006  《A Review of Urban Digital Twins Integration, Challenges, and Future Di》

## [battery] solid-state battery latest research breakthroughs and most promising directions 2026

- web/academic top-10 分数线(cutoff) = **0.965**；OpenAlex 论文 19 篇

| 指标 | blob(整段摘要) | best_sent(切句取最高) |
|---|---|---|
| 平均分 | 0.176 | 0.235 |
| 中位数 | 0.113 | 0.129 |
| 最高分 | 0.681 | 0.893 |
| **越过分数线的论文数** | **0/19** | **0/19** |

- 提升最大的论文（blob→best_sent）：
  - 0.222→0.770  《Toward Sustainable All Solid‐State Li–Metal Batteries: Perspectives on》
  - 0.215→0.691  《Li–Solid Electrolyte Interfaces/Interphases in All-Solid-State Li Batt》
  - 0.521→0.893  《Progress of Polymer Electrolytes Worked in Solid‐State Lithium Batteri》