# 多 Agent 协调架构方案

> 承接《[深度研究架构调研与体验优化.md](深度研究架构调研与体验优化.md)》里被列为「P2，可选，需先跑 benchmark 验证」的 orchestrator-worker 选项，以及《[数据结构重构方案.md](数据结构重构方案.md)》里阶段 3 的 `ResearchNode` 树。本篇把这两处悬而未决的东西定下来：选哪个多 agent 协调方案、为什么选它、怎么落到现有 8 节点图上。

## 0. 结论先行

**选 LangGraph 官方的 supervisor / orchestrator-worker 模式**，不引入新框架。

理由是这是唯一同时满足两个硬约束的选项：

1. **范式已经在生产系统里验证过** —— Anthropic 自家的 multi-agent researcher（lead agent + 并行 subagent，各自独立 context window）是这个模式的原型，工程博客公开了生产实测数据。
2. **工程实现和现有技术栈零缝隙** —— 项目已经在用 LangGraph（`requirements.txt` 锁定 `langgraph>=0.4.0`），supervisor 模式是同一个框架官方维护的另一种图拓扑，不是要接入的新依赖。

这不是本项目的自研方向，是把已验证过的模式套进已经在用的框架里。风险边界因此比"引入一个新框架赌它成不成熟"小得多。

代价也要一起说清楚：Anthropic 公开的实测数字是 multi-agent 比单轮对话多耗约 **15 倍 token**，比单个精调好的 agent 也要多约 **4 倍**。这是同一个被验证过的系统给出的成本，不是理论风险，第 4 节给出对应的放量策略。

## 1. 候选方案对比

| 方案 | 成熟度证据 | 结构 | 是否适配本项目 |
|---|---|---|---|
| **Anthropic multi-agent researcher** | 官方工程博客公开生产架构与实测：lead agent + 3-5 个并行 subagent，各自独立 context window，比单体质量提升约 90.2% | 并行 fan-out，supervisor 决定开几个 worker、何时收敛 | 是范式来源，不是可安装的库，协调逻辑要自己写 |
| **LangGraph supervisor 模式**（`langgraph-supervisor`） | 官方维护；结构上是上一行范式的框架化实现，用 `Command(goto=..., update=...)` 做节点间路由，worker 天然跑在隔离的 subgraph state 里 | 同上，且原生支持 subgraph 隔离状态 | **选中**——与现有 `StateGraph`（`app/graph/builder.py`）同源，改的是图拓扑，不是技术栈 |
| OpenAI Swarm / Agents SDK handoff | 有生产验证（ChatGPT 的路由类场景），但控制权是**单线程接力**，一次只有一个 agent 持有 | 顺序 handoff，不做并行 fan-out | 排除——问题形状不匹配。本项目要解决的是"多个子问题并行深挖"，不是"客服式路由到某个专精 agent" |

## 2. 现状：决策点只有两个，且都是一次性的

`app/graph/builder.py` 里的边是纯线性、零条件边：

```python
workflow.add_edge("planner", "retriever")
workflow.add_edge("retriever", "content_extractor")
workflow.add_edge("content_extractor", "source_evaluator")
workflow.add_edge("source_evaluator", "evidence_builder")
workflow.add_edge("evidence_builder", "analyst")
workflow.add_edge("analyst", "fact_checker")
workflow.add_edge("fact_checker", "report_writer")
```

八个节点里真正做决策的只有两处：

- **planner**：一次 LLM 调用产出 `sub_questions` 后即冻结（`app/graph/nodes.py::planner_node`，第 2+ 轮直接跳过重新规划）。
- **fact_checker**：唯一判断"证据够不够"的地方，但在**整条链路跑完之后**才发生。不够时触发的是 `research_runner.py::_supplement_round`，对 `failed_sub_questions` 重跑 retriever→content_extractor→source_evaluator→evidence_builder→analyst→fact_checker **六个节点的整段路径**，而不是针对性地补一次检索。

中间的 retriever / content_extractor / source_evaluator / evidence_builder / analyst 五个节点，没有一个能在执行中改变自己要做什么——analyst 读到的是执行前就已经装好的固定证据池，发现证据不足也无法自主触发新检索，只能等 fact_checker 在终点判定后整段重放。

## 3. 目标架构：supervisor + worker subgraph

### 3.1 拓扑

```
supervisor（新增；取代现在"跑一次就废弃"的 planner 心智模型）
  │  职责：产出子问题 + 按需追加 worker + 判断何时收敛
  │  依据：question_analysis 的复杂度信号（planner_node 已经在产出，未被下游使用）
  │
  ├─ worker subgraph × N（一个子问题一个，并行）
  │    retriever → content_extractor → source_evaluator → evidence_builder → analyst
  │    独立 state 切片，互不共享中间证据，只把 SubAnswer 交回 supervisor
  │
  └─ supervisor 收到 worker 结果后：
       跑 fact_checker 式判断 → 不够就只对相关 worker 重新触发
       （而不是像现在 _supplement_round 那样重放六个节点）→ 够了转 report_writer
```

`supervisor` 的子问题结构直接复用《数据结构重构方案.md》里提出的 `ResearchNode` 树（`parent_id` / `children` / `status`），而不是现在 `state.py` 里的扁平 `sub_questions: list[dict[str, Any]]`——worker 的动态增补（一个子问题挖出新的子子问题）需要树形结构才能表达，扁平列表表达不了父子关系。这是两篇文档在这一点上的直接耦合，落地时应该一起做，不要拆开。

### 3.2 一个必须提前解决的并发风险：Citation Registry

现在 `analyst_node`（`app/graph/nodes.py:914-931`）构建全局 Citation Registry 的方式，是在**单线程同步代码**里先把所有 `accepted_docs` 登记完、`url_to_cid` 字典建好，之后才发起各子问题的并发 LLM 分析（`_analyze_single_question` 的 `asyncio.gather`）。`_ensure_cid` 闭包（第 987 行）虽然后续也会被调用，但调用点仍在并发 fan-out 之前的证据文本构建阶段，不存在竞态。

换成 worker subgraph 并行执行后，这个前提不再成立：如果每个 worker 各自独立检索、各自往 registry 里登记来源，`url_to_cid` 的写入就会发生在真正并发的多个 subgraph 里，直接照搬现在的闭包写法会产生竞态（两个 worker 同时给同一个新 URL 分配不同 CID，或分配号跳号）。

落地时必须选其中一种，不能绕过：

- **收敛登记**：worker 只上报"我要引用哪些 URL"，CID 分配统一放到 supervisor 收到全部结果后的合并步骤里做——继续沿用现在"先登记、后使用"的顺序模型，只是把"先登记"的执行点从 analyst 内部挪到 supervisor。
- **预分配区间**：给每个 worker 分配互不重叠的 CID 号段，规避锁，但需要在最终 registry 里做一次紧凑化（否则 CID 会有大段跳号，影响可读性）。

推荐前者——改动小，且和现有"追加而非重建"的跨轮语义（`_supplement_round` 里显式传入 `citation_registry`）保持一致，不需要额外发明新规则。

## 4. 成本与放量策略

Anthropic 的 15 倍 token 成本数据来自同一个被验证过的生产系统，必须原样纳入落地策略，不能只学范式不学约束。做法是让 supervisor 自己做这个判断，而不是全量切换：

- `planner_node` 已经在产出的 `question_analysis`（问题复杂度信号）直接喂给 supervisor，作为要不要多开 worker 的依据。
- 简单问题：1 个 worker，等价于现在的线性流程，token 成本不变。
- 复杂问题：按子问题数量开多个 worker，成本上升但仅限确实需要深挖的查询。

这是加法，不是替换——旧的单 worker 路径原样保留，多 agent 化只在判定复杂的查询上生效。

## 5. 迁移路径

| 阶段 | 内容 | 风险 |
|---|---|---|
| 1 | 引入 supervisor 节点，但先固定只开 1 个 worker——即现在的线性链路原样跑一遍，只是换了个壳。验证图拓扑改动本身不引入回归 | 低，无行为变化 |
| 2 | Citation Registry 改为收敛登记模型（3.2 节），配合 `ResearchNode` 树的 Phase 1（《数据结构重构方案.md》Phase 1，Source 聚合与命名统一）一起做 | 中，动了核心引用逻辑，需要跑一遍 `benchmark/GoldenDataset` 回归 |
| 3 | supervisor 按 `question_analysis` 复杂度动态开多个并行 worker | 中，需要先有第 2 阶段的并发安全 registry 打底 |
| 4 | worker 内部支持针对性追加检索（对应第一篇文档的 P2-1），替换掉 `_supplement_round` 的整段重放 | 高，改变了核查失败后的补救路径，需要专项回归 |

每阶段独立可发布，出问题可以停在任意一步，不需要一次性切换。

## 6. 不做什么

- **不放弃 LangGraph**——supervisor 模式是同一框架内的另一种图拓扑，不是引入新的 agent 框架。
- **不做无上限的 ReAct 式自由循环**——worker 内部的追加检索需要硬上限（沿用第一篇文档 P2-1 的建议：每子问题至多 1-2 次），否则失去现在"零条件边、行为可预测"的可测试性优势。
- **不在并发安全的 Citation Registry 落地之前开并行 worker**——3.2 节的竞态是真实存在的代码事实，不是假设风险，必须先解决再谈并行。
- **不全量切换**——按复杂度门控，简单查询保留现在的单 worker 直线路径。

## 参考

- Anthropic, "How we built our multi-agent research system"（lead agent + 并行 subagent 生产架构与实测数据）
- LangGraph 官方文档：multi-agent supervisor 模式、`Command` 路由、subgraph 状态隔离
- 本仓库：`app/graph/builder.py`（当前线性图）、`app/graph/nodes.py`（Citation Registry 构建逻辑，第 914-1019 行）、`app/services/research_runner.py`（`_supplement_round` 现有的整段重放实现）
- [深度研究架构调研与体验优化.md](深度研究架构调研与体验优化.md)（P2 orchestrator-worker 提案出处）
- [数据结构重构方案.md](数据结构重构方案.md)（`ResearchNode` 树，Phase 3）
