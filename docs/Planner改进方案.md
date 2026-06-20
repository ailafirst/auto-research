# Planner 改进方案

> 记录 `planner_node`（`app/graph/nodes.py`）及 `app/prompts/planner.md` 的完整改进历程，按版本号组织。

---

## v0.1.0 — 初始版本

### 基本实现

- `planner_node` 调用 LLM 生成研究计划，输出 `sub_questions`（子问题列表）和每个子问题的 `search_queries`
- 存在 `_rule_based_plan()` 函数作为 LLM 不可用时的降级路径，使用固定模板生成计划

### 已知缺陷

- 规则降级产生的计划质量极低，且 LLM 不可用时 analyst/fact_checker/report_writer 同样无法运行，降级毫无意义
- Prompt 为简单指令，无结构化分析阶段，depth/intent 无法根据问题自适应
- `analyst_node` 对子问题串行分析：`N 个子问题 × T = N×T` 总耗时

---

## v1.1.0 — Planner 自适应深度 & Benchmark

### 1. 规则降级移除

`_rule_based_plan()` **完全移除**。当前 `planner_node` 强制走 LLM 路径，LLM 不可用时直接抛出异常而非静默降级。

### 2. Prompt 重写（V2：两阶段分析生成）

`app/prompts/planner.md` 从简单指令升级为**两阶段结构化 Prompt**：

#### Phase 1 — 问题分析（Question Analysis）

LLM 在生成子问题前，必须先完成四维分类：

**① Intent（意图）— 5 类**

| Intent | 含义 |
|--------|------|
| `quick_overview` | 入门定向，用户需要基础心智模型 |
| `deep_investigation` | 单一主题，多角度并行深入分析 |
| `comparison` | 多个具名对象，评估差异与权衡 |
| `how_to` | 行动导向，需要可操作的步骤 |
| `trend_tracking` | 领域全景，映射已有方向与格局 |

**② Domain（领域）— 7 类**

`technology` / `business` / `science` / `legal` / `education` / `policy` / `general`

分类依据为"回答该问题**主要需要哪类知识**"，不仅看话题表面。

**③ Research Depth（研究深度）— 自适应**

| Depth | 触发条件 | 子问题数 | 搜索词/题 |
|-------|----------|---------|---------|
| `shallow` | `quick_overview` intent；入门级问题 | 2–3 | 2–3 |
| `medium` | 标准问题；单域 `how_to` / `trend_tracking` | 4–5 | 3–4 |
| `deep` | `deep_investigation` / `comparison`；跨域；多维并行问题 | 5–7 | 4–5 |

**关键设计**：深度由 LLM 根据 intent 自主推断，不再依赖用户传入的 `search_depth` 参数硬控制。`search_depth` 参数仍通过 hint_block 传入，但仅作为**软偏好**参考。

**④ Research Dimensions（研究维度）— 7 种**

`conceptual` / `technical` / `comparative` / `practical` / `trend` / `critical` / `contextual`

选 2–4 种指导子问题角度选取，部分 intent 有强制要求（如 `comparison` 必含 `comparative`）。

#### Phase 1 输出（存入 `research_strategy`）

```json
{
  "intent": "trend_tracking",
  "domain": "technology",
  "depth": "medium",
  "dimensions": ["technical", "trend"],
  "reasoning": "One sentence explaining the classification decision."
}
```

#### Phase 2 — 研究计划生成

基于 Phase 1 分析，按以下规则生成子问题：

**领域专属角度库**（替代通用模板）

| 领域 | 推荐角度 |
|------|---------|
| technology | 核心机制 · 主流实现/框架 · 性能与工程挑战 · 生态与工具链 · 演进趋势 |
| business | 市场规模/格局 · 商业模式 · 竞争优劣势 · 用户分析 · 监管风险 |
| science | 基础原理 · 实验证据 · 当前科学共识 · 开放问题 · 应用前景 |
| legal | 法律依据 · 执法实践 · 合规要求 · 司法管辖差异 · 标志性判例 |
| policy | 政策目标 · 关键措施 · 执行现状 · 国际比较 · 争议与挑战 |

**子问题规则（关键约束）**

- `comparison` intent：每个对比对象一个子问题 + 一个综合权衡子问题
- `how_to` intent：子问题按"前提 → 核心步骤 → 验证"顺序排列
- 搜索词：中英混合，技术术语保留英文，同一子问题内搜索词保持**语义多样性**（不只是换词）

#### Phase 2 输出（存入 `research_plan` / `sub_questions`）

```json
{
  "question_analysis": { ... },
  "research_goal": "一句话研究目标（用户语言）",
  "question_type": "兼容字段",
  "sub_questions": [
    {
      "id": "q1",
      "priority": 1,
      "angle": "角度标签（用户语言）",
      "question": "完整子问题句",
      "search_queries": ["中文词A", "中文词B", "English term C"]
    }
  ]
}
```

### 3. 用户偏好软注入（hint_block）

`planner_node` 在 LLM 调用前，将用户参数作为**软偏好**拼入 user_message：

```python
hints = state.get("user_hints") or {}
hint_lines = []
if hints.get("report_type"):
    hint_lines.append(f"用户偏好报告类型: {hints['report_type']}")
if hints.get("search_depth"):
    hint_lines.append(f"用户偏好搜索深度: {hints['search_depth']}")
hint_block = "用户可选偏好（仅供参考，可根据问题实际情况调整）：\n" + "\n".join(hint_lines)
```

### 4. Analyst 节点并发化（关联改动）

`analyst_node` 将子问题分析从串行 `for` 循环改为 `asyncio.gather` + `asyncio.Semaphore(5)` 并发：

```
旧（串行）：N 个子问题 × LLM耗时/次 = N × T
新（并发）：max(各子问题 LLM 耗时) ≈ T（受最慢一次决定）
```

### 5. Benchmark 数据（Prompt V2 实测）

| 任务 | Intent | Domain | Depth | 子问题数 | 搜索词数 |
|------|--------|--------|-------|---------|---------|
| LLM推理加速（景观问题） | trend_tracking | technology | medium | 4 | 12 |
| Python vs Rust（对比问题） | comparison | technology | deep | 6 | 24 |
| 量子计算基础（入门问题） | quick_overview | science | shallow | 3 | 8 |

自适应深度结果与预期完全一致：景观 → medium、对比 → deep、入门 → shallow。

---

## v0.1.2 — 多轮补充研究修复

### 问题（修复前）

第 2 轮将 `sub_questions` 重置为空，`planner_node` 对同一个 `query` 从头规划，生成几乎相同的计划；`follow_up_queries` 作为初始 `search_queries` 传入后立刻被 planner 覆盖，实际从未被搜索。第 2 轮等于原样重做第 1 轮。

### 修复方案

**`routes_research.py`**：第 2 轮 `new_state` 携带第 1 轮的 `sub_questions`、`research_plan`、`research_strategy`、`follow_up_queries`，不再重置为空。

**`planner_node`**：检测到 `sub_questions` 非空且 `current_round > 1` 时，直接透传已有计划，跳过 LLM 调用：

```python
existing_sqs = state.get("sub_questions", [])
if existing_sqs and state.get("current_round", 1) > 1:
    return {
        "sub_questions":     existing_sqs,
        "research_plan":     state.get("research_plan", {}),
        "research_strategy": state.get("research_strategy", {}),
        "search_queries":    state.get("follow_up_queries", []),
        ...
    }
```

**`retriever_node`**：第 2 轮检测到 `follow_up_queries` 非空时，仅搜索 follow_up 查询，不重复第 1 轮搜索词：

```python
follow_ups = state.get("follow_up_queries", [])
if follow_ups and state.get("current_round", 1) > 1:
    tagged = [(q, "") for q in follow_ups[:5]]
else:
    tagged = [(q, sq_id) for sq in sub_questions for q in sq["search_queries"]]
```

### 效果

| | 修复前 | 修复后 |
|--|--------|--------|
| 第 2 轮 planner | 重新 LLM 规划（~5–10s，结果近似第 1 轮） | 跳过，直接复用第 1 轮计划 |
| 第 2 轮 retriever | 重复搜索第 1 轮相同关键词 | 搜索 fact_checker 识别的补充方向 |
| follow_up_queries | 传入但被 planner 覆盖，实际未使用 | 由 retriever 执行，获取增量证据 |

---

## Backlog — 待实现优化方向

### B：搜索词语义去重 ⭐ 推荐优先

**问题**：Planner 生成 20–30 条搜索词后，其中存在大量语义重复：

```
"LangGraph 概述" ≈ "LangGraph overview" ≈ "LangGraph 介绍"
```

**方案**：在 `planner_node` 输出后、进入 `retriever_node` 前，对 `search_queries` 做向量相似度过滤：

```python
embeddings = embed(search_queries)   # fastembed 本地模型（已有，无额外依赖）
filtered = cosine_dedup(embeddings, threshold=0.85)
```

**收益**：预计减少搜索次数 30–50%，降低 Tavily/DDG 调用频率。

---

### C：多视角并行规划

并发发出 2–3 个不同角色的规划请求，然后合并去重：

```python
perspectives = ["学术研究者视角", "行业从业者视角", "终端用户视角"]
plans = await asyncio.gather(*[llm.chat(build_messages(p, query)) for p in perspectives])
merged = merge_and_dedup(plans)
```

**收益**：研究覆盖面更广，时延与单次规划相同，代价是 3 倍 token 消耗。

---

### D：多轮感知规划（完整版）

**当前（v0.1.2 简化版）**：第 2 轮跳过 planner，直接复用第 1 轮子问题结构 + follow_up 搜索。

**完整版目标**：round ≥ 2 时让 planner 读取上一轮的 `fact_check_result.issues` 和 `sub_answers`，用 LLM 生成**专门针对证据缺口的新子问题**：

```python
if state["current_round"] > 1:
    context = {
        "previous_answers": state["sub_answers"],
        "issues": state["fact_check_result"]["issues"],
        "follow_up": state["follow_up_queries"],
    }
    prompt = build_followup_plan_prompt(context)   # 新增 planner_followup.md
```

**改动位置**：`planner_node`（判断 round）、`app/prompts/` 新增 `planner_followup.md`。

---

### E：领域专化细化（长期）

Phase 1 的 domain 分类已提供领域识别基础，Phase 2 已有各领域角度库。可注入更细粒度的子领域专属角度（医疗、金融、半导体等），当前方案对通用领域已够用。

---

### F：规划反思（Reflection Loop）

retriever 完成后，插入轻量 LLM 评估搜索结果与研究计划覆盖匹配度：

```
planner → retriever → [reflection] → （匹配度低 → re-plan） → content_extractor → ...
```

**代价**：增加 1 次 LLM 调用延迟（约 2–3s），需修改 LangGraph 图结构加入条件边。

---

## 版本汇总

| 版本 | 改动 | 状态 |
|------|------|------|
| v0.1.0 | 基本 planner + _rule_based_plan（已废弃） | 已发布 |
| v1.1.0 | Prompt V2 两阶段、自适应深度、hint_block、analyst 并发 | 已发布 |
| v0.1.2 | 多轮修复：planner 跳过重规划，retriever 用 follow_up_queries | 已发布 |
| 下一版本 | B 搜索词去重（推荐优先） | Backlog |
| 下一版本 | C 多视角并行规划 | Backlog |
| 未来版本 | D 多轮感知规划完整版、E 领域专化、F 反思循环 | Backlog |
