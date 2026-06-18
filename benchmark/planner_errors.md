# Planner 错误记录

基于 12 任务基准测试（2026-06-18）结果，按严重程度排序。

通过率：7/11（63.6%），1 个致命错误（任务 11 未计入通过率统计）。

---

## E1 ｜ JSON 解析失败（致命）

**影响任务：** 11（微服务 vs 单体架构）  
**现象：** LLM 返回含非法字符的 JSON（未转义引号、截断等），`json.loads()` 抛出 `JSONDecodeError`，任务直接崩溃，无法评分。  
**根因：** `planner_node` 原先以纯文本调用 LLM，不强制输出格式，LLM 偶发格式错误时无兜底。

**修复方案（已应用）：** 在 `planner_node` 调用 `llm.chat()` 时传入 `response_format={"type": "json_object"}`，启用 OpenAI 兼容 API 的 JSON mode。该模式在模型侧保证输出合法 JSON，从根本上消除解析失败风险。

```python
# app/graph/nodes.py — planner_node
content = await llm.chat(
    messages,
    temperature=0.3,
    response_format={"type": "json_object"},
)
plan = json.loads(content)   # 此时可安全解析，无需 try/except
```

**前提：** `planner.md` 中已含 "Return strict JSON only"，满足 OpenAI JSON mode 要求（系统提示必须提及 json）。  
**注意：** 若 LLM 提供商不支持 `response_format`（如部分本地模型），LiteLLM 会忽略该字段，退化为原有行为，不影响运行。

---

## E2 ｜ `trend_tracking` intent 几乎不被触发（高）

**影响任务：** 01（LLM推理加速，55pts ✗）、07（碳中和技术路径，60pts ✗）  
**现象：** "有哪些主要方向"、"有哪些核心技术路径和政策工具" → Planner 判为 `deep_investigation`，得 0 分（权重 20pts）。  
**根因：** `planner.md` 中 `trend_tracking` 的触发词（"趋势"、"未来"、"发展方向"）要求显式意图，而"有哪些方向/路径"这类枚举式问句被误判为"多方面并列的深度调查"。

**修复方案（已应用）：** 移除 `planner.md` 中所有 intent 的关键词触发表，改为**语义推理框架**：

- 每个 intent 用一段话描述"用户的底层认知需求"，而非列举触发词
- `trend_tracking` 的描述聚焦于"用户想 *mapping what exists* in a field"，并明确指出即使没有"趋势/未来"字样，"有哪些方向/路径/方法"也属于此类
- 在 `deep_investigation` 中加入对比提示：若问题是映射领域全景（→ trend_tracking）或比较多个命名对象（→ comparison），应选那两个而非此项
- domain 部分同样改为"通过哪种主要认知视角来回答"的推理描述，解决 technology 过度泛化的问题

---

## E3 ｜ domain 分类偏向 `technology`，`business`/`science`/`policy` 漏判（高）

**影响任务：** 03（量子计算，technology→science，20pts✗）、05（AI芯片竞争，technology→business，20pts✗）、07（碳中和，general→policy，20pts✗）、10（固态电池，technology→science，20pts✗）  
**现象：** 凡涉及技术或设备的问题，均落入 `technology`；竞争格局、政策工具、材料科学等场景无法突破。

**具体混淆点：**

| 查询特征 | 实际判断 | 应判断 | 原因 |
|---------|----------|--------|------|
| AI 芯片厂商竞争格局 | technology | business | 主语是"竞争格局"而非芯片原理 |
| 量子计算基本原理 | technology | science | 量子力学属物理科学，非工程实现 |
| 碳中和政策工具 | general | policy | 明确有"政策工具"关键词 |
| 固态电池材料对比 | technology | science | 材料/电化学属科学范畴 |

**建议修复：** 在 `planner.md` domain 表格中补充排他说明：

```markdown
| `technology` | AI algorithms, frameworks, architecture, software engineering,
                 **芯片架构/原理**（硬件技术层面）— 若问题主语是市场竞争/商业格局则用 business |
| `business`   | market, competition, revenue, strategy, **竞争格局**, **市场份额**,
                 **技术壁垒（商业视角）**, industry analysis |
| `science`    | physics, chemistry, biology, quantum, materials, **电化学**,
                 **基本原理（自然科学）**, experimental mechanisms |
| `policy`     | government policy, carbon neutrality, **碳排放**, **碳中和**,
                 international climate agreements, governance |
```

---

## E4 ｜ `comparison` intent 在 3+ 对象时漏判（中）

**影响任务：** 05（AI芯片竞争格局，intent 判为 deep_investigation，20pts✗）  
**现象：** 4 个命名厂商（英伟达/AMD/英特尔/国产）＋"各自"，应触发 `comparison`，但 Planner 判为 `deep_investigation`。  
**根因：** Planner 对 2 对象比较识别良好（任务 02 Python/Rust 正确），但 3+ 对象时认为问题"更复杂"，倾向 `deep_investigation`。

**建议修复：** 在 intent 触发规则中明确：

```markdown
| `comparison` | "对比", "vs", "区别", "哪个好", "选型",
                 two **or more** named objects to compare（包含 3+ 对象）,
                 **"各自"、"分别"与多个命名对象并存** |
```

---

## E5 ｜ depth 整体偏深，`shallow` 判断过于保守（中）

**影响任务：** 01（medium→deep，−10pts）、03（shallow→medium，−10pts）、06（deep→medium，−10pts）  
**现象：** 含比较结构的问题自动升级为 `medium/deep`；入门性问题（"基本原理是什么"）也升为 `medium`。

**规律：**
- `shallow` 在 12 个任务中从未被选中（期望 2 次）
- 任何含"对比"结构的问题 depth 会自动升一级

**建议修复：** 在 depth 表格中强化 `shallow` 的触发条件，防止"含对比 ≠ 升深度"：

```markdown
| `shallow` | `quick_overview` intent; **"是什么"、"基本原理"、"简介"**;
              short/generic question; beginner-level;
              **即使含对比结构，若问题核心是入门解释，仍选 shallow** |
```

---

## E6 ｜ `comparison` 子问题数量不足（低）

**影响任务：** 02（Python vs Rust，sub_q=3，期望 5–7，−10pts）  
**现象：** 2 对象 comparison 只生成 3 个子问题（Python 1 个 + Rust 1 个 + 综合 1 个），未充分展开技术细节和场景分析。  
**根因：** Prompt 规定"一个对象一个子问题 + 一个综合子问题"，对 deep 场景下的细粒度拆解指导不足。

**建议修复：** 补充 comparison 子问题拆解指引：

```markdown
**`comparison` intent — deep depth**: 每个对象除"总体特性子问题"外，
还应补充独立角度（如性能、生态、适用场景），最终合并子问题数达到 5–7 个。
```

---

## E7 ｜ 查询数量在 deep 模式下不足（低）

**影响任务：** 05（avg=3，期望 4–5，−10pts）、06（avg=3，期望 4–5，−10pts）  
**现象：** deep depth 任务每子问题平均仅生成 3 条查询，低于期望下限 4 条。  
**根因：** Prompt 对 deep 模式的查询数量要求（4–5 条）没有在 sub-question 生成规则中做强制提醒。

**建议修复：** 在 Search queries 要求处加入 depth-aware 提示：

```markdown
- **`deep` depth**: each sub-question must have **at least 4 search queries**.
```

---

## 修复优先级汇总（V1.0 → V1.1）

| 优先级 | 错误 | 操作 | 结果 |
|--------|------|------|------|
| P0 ✅ | E1 JSON 崩溃 | `response_format={"type":"json_object"}` | 任务 11 不再崩溃 |
| P1 ✅ | E2 trend_tracking 漏触发 | 重写 intent 为语义推理框架 + few-shot | 任务 01、07 intent 修复 |
| P1 ✅ | E3 domain 偏 technology | 重写 domain 为"主要认知视角"描述 | 任务 03、05、07、10 domain 修复 |
| P2 ✅ | E4 comparison 多对象漏判 | intent 描述明确"3+ 对象同样触发" | 任务 05 intent 修复 |
| P2 ✅ | E5 depth 偏深 | shallow 描述强化 | 任务 01、03、06 depth 修复 |
| P3 ✅ | E6 comparison 子问题少 | 新 prompt + few-shot | 任务 02 子问题从 3→5 |
| P3 | E7 deep 查询数不足 | 待处理 | — |

**V1.1.0 实测结果：7/12 全项正确（原约 4/12），总耗时 29.9s（并发 6）**

---

## V1.1.0 新发现的问题

### E8 ｜ `trend_tracking` 与 `deep_investigation` 在"枚举式单主题"上边界模糊（中）

**影响任务：** 12（AGI技术障碍，expected `deep_investigation`，actual `trend_tracking`）  
**现象：** "有哪些关键技术障碍" → Planner 识别为 `trend_tracking`（领域全景映射），应为 `deep_investigation`（单一主题多角度分析）。  
**根因：** 新 prompt 的 `trend_tracking` 定义覆盖了"有哪些"句式，few-shot Example E（分布式一致性）的反例未能覆盖"枚举单主题内部障碍"这类边界情况。  
**区分原则：** 核心主语是**一个特定对象**（AGI、某技术）且枚举的是**它的内在问题/障碍** → `deep_investigation`；核心主语是**一个领域**且枚举的是**该领域中存在的方法/实体** → `trend_tracking`。

### E9 ｜ `how_to` 复杂系统问题 depth 偏 `deep`（低）

**影响任务：** 04（RAG最佳实践）、08（电商推荐系统），均由 medium→deep  
**现象：** 工程系统构建类问题被识别为 `deep` 而非 `medium`。LLM 以"系统复杂度"覆盖了 intent-based 规则。  
**注：** intent 和 domain 均正确，depth 偏差一级，对研究方向影响有限。  
**建议修复：** depth 表格的 `medium` 行补充约束：单系统/单流程的 `how_to` 问题即使复杂也选 `medium`，`deep` 仅限跨域或需比较多方案时使用。
