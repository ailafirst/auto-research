# Planner 改进方案

> 整理自开发讨论，涵盖已落地改动与待实现优化方向。

---

## 一、已实施改动

### 1.1 Prompt 重写（`app/prompts/planner.md`）

| 维度 | 改前 | 改后 |
|------|------|------|
| 搜索词数量 | 未明确（实际 1-2 条） | 明确要求 **3-5 条** |
| 搜索词语言 | 仅中文 | **中英文混合**（专业术语保留英文） |
| 子问题字段 | `id, question, search_queries` | 新增 `priority`（优先级）、`angle`（研究角度） |
| 角度覆盖 | 无 | 列出 9 个角度供选择，`comparison` 类型强制含对比角度 |
| 问题类型 | 4 种 | 增加 `deep_analysis` |

新增 JSON 字段：
```json
{
  "id": "q1",
  "priority": 1,
  "angle": "核心机制与工作原理",
  "question": "...",
  "search_queries": ["中文词1", "中文词2", "English term"]
}
```

### 1.2 规则降级方案增强（`_rule_based_plan`）

- 每个子问题从 2 条搜索词扩展到 **4 条**（含英文变体）
- 新增 `priority` 和 `angle` 字段，与 LLM 路径输出对齐
- `comparison` 类型总查询数从 12 条提升到 **24 条**

### 1.3 Analyst 节点并发化（关联改动）

子问题分析从串行 `for` 循环改为 `asyncio.gather` 并发，Planner 生成的子问题越多，加速效果越显著：

```
旧（串行）：N 个子问题 × 3s/次 LLM = N×3s
新（并发）：max(各子问题耗时) ≈ 3-5s（受最慢一次决定）
```

---

## 二、待实现优化方向

### 方向 A：自适应规划深度 ⭐ 推荐优先

**问题**：`search_depth`（basic/advanced）和 `report_type`（summary/deep/comparison）已作为参数传入，但 planner 完全忽略了它们。

**方案**：

| 参数组合 | 子问题数 | 搜索词/题 | 说明 |
|---------|---------|---------|------|
| `summary` | 2-3 | 2 条 | 快速概览，减少 LLM 调用 |
| `deep`（默认） | 4-5 | 4 条 | 当前行为 |
| `comparison` | N个对比对象+2 | 4-5 条 | 每个对比对象独立成题 |
| `search_depth=basic` | -1 题 | -1 词 | 轻量模式 |

**改动位置**：`planner_node`（传入 `state` 中已有 `report_type` 字段），`app/prompts/planner.md`（加入动态指令段）。

**收益**：减少不必要的 LLM 调用和搜索次数，summary 场景速度提升约 40%。

---

### 方向 B：搜索词语义去重 ⭐ 推荐优先

**问题**：Planner 生成 20-30 条搜索词后，其中存在大量语义重复：
```
"LangGraph 概述" ≈ "LangGraph overview" ≈ "LangGraph 介绍"
```
三条词触发三次搜索，但结果高度重叠。

**方案**：在 `planner_node` 输出后、进入 `retriever_node` 前，对 `search_queries` 做向量相似度过滤：

```python
# 伪代码
embeddings = embed(search_queries)          # fastembed 本地模型
filtered = cosine_dedup(embeddings, threshold=0.85)
state["search_queries"] = filtered
```

**改动位置**：`planner_node` 尾部 或 新增 `query_dedup_node`（插在 planner 和 retriever 之间）。

**收益**：预计减少搜索次数 30-50%，降低 DuckDuckGo/Tavily 调用频率，同时减少后续爬取和向量化压力。

---

### 方向 C：多视角并行规划

**问题**：当前 planner 只向 LLM 发一次请求，视角单一，可能遗漏某些维度。

**方案**：并发发出 2-3 个不同角色的规划请求，然后合并去重：

```python
perspectives = [
    "你是一名学术研究者，关注理论基础和文献证据",
    "你是一名行业从业者，关注落地实践和工程挑战",
    "你是一名终端用户，关注使用体验和实际效果",
]

plans = await asyncio.gather(*[
    llm.chat(build_messages(perspective, query))
    for perspective in perspectives
])

# 合并子问题，按覆盖度去重
merged = merge_and_dedup(plans)
```

**收益**：研究覆盖面更广，报告质量提升；由于并发执行，**时延与单次规划相同**，代价是 3 倍 token 消耗。

**改动位置**：`planner_node`，需新增 `merge_plans()` 合并逻辑。

---

### 方向 D：多轮感知规划

**问题**：当前第 2 轮研究直接用 `follow_up_queries` 跳过 planner，重走 retriever，planner 不知道第 1 轮已经发现了什么，导致：
- 可能重复搜索已有答案的问题
- 无法针对性地补充特定缺口

**方案**：在 round ≥ 2 时，让 planner 读取上一轮的 `fact_check_result.issues` 和 `sub_answers`，生成**针对性补充计划**：

```python
if state["current_round"] > 1:
    context = {
        "previous_answers": state["sub_answers"],
        "issues": state["fact_check_result"]["issues"],
        "follow_up": state["follow_up_queries"],
    }
    prompt = build_followup_plan_prompt(context)
else:
    prompt = build_initial_plan_prompt(query)
```

**收益**：多轮研究质量显著提升，补充搜索更有针对性，减少无效重复。

**改动位置**：`planner_node`（判断 round）、`app/prompts/` 新增 `planner_followup.md`。

---

### 方向 E：领域自动识别与专化角度

**问题**：通用角度模板（背景/机制/现状/风险/趋势）对不同领域效果不同，医疗类需要"临床证据"，金融类需要"监管政策"，技术类需要"性能基准"。

**方案**：在 planner 中增加一步轻量领域分类，然后注入领域专属角度：

```python
DOMAIN_ANGLES = {
    "technology": ["开源生态", "性能基准", "社区活跃度", "API 设计"],
    "finance":    ["监管政策", "风险敞口", "市场规模", "合规要求"],
    "medical":    ["临床证据级别", "副作用", "适应症", "指南推荐"],
    "legal":      ["法律依据", "司法实践", "争议焦点", "地区差异"],
}
```

**收益**：领域专业度提升，减少通用模板的"废话"子问题。

**改动位置**：`planner_node` 增加领域检测逻辑，`app/prompts/planner.md` 增加条件角度块。

---

### 方向 F：规划反思（Reflection Loop）

**问题**：planner 生成计划后直接执行，无验证环节。若 retriever 返回结果与预期严重偏差，整轮研究可能方向错误。

**方案**：retriever 完成后，插入轻量 LLM 调用评估"搜索结果与研究计划的覆盖匹配度"：

```
planner → retriever → [reflection] → （匹配度低 → re-plan） → content_extractor → ...
```

**收益**：减少因规划不准导致的整轮无效研究。

**代价**：增加 1 次 LLM 调用延迟（约 2-3s），且需要修改 LangGraph 图结构加入条件边。

---

## 三、优先级建议

| 优先级 | 方向 | 理由 |
|--------|------|------|
| ★★★ 立即做 | A 自适应深度 | 改动小，参数已有，收益直接 |
| ★★★ 立即做 | B 搜索词去重 | 减少下游压力，无需新 LLM 调用 |
| ★★☆ 下一迭代 | C 多视角并行规划 | 质量提升最显著，token 成本增加 |
| ★★☆ 下一迭代 | D 多轮感知规划 | 改善多轮研究的核心痛点 |
| ★☆☆ 长期 | E 领域专化 | 需要维护领域知识库，成本较高 |
| ★☆☆ 长期 | F 反思循环 | 需修改 LangGraph 图结构，复杂度高 |
