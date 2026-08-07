# LLM 节点改进：Planner / Analyst / Fact-checker / Report Writer

> 四个 LLM 节点的提示词、输出约束与并发演进，以及支撑这些改动的八轮 benchmark。
> 合并自原《Planner 改进方案》《Analyst 改进基准测试报告》。
>
> **贯穿全篇的结论：约束输出结构，比调教措辞有效得多。** few-shot 把字段填写正确率从 43% 拉到 91%，却对 issue 数量毫无影响（77→77）；换成 JSON Schema strict 后 issue 直接 -94%（77→5）。

---

## 一、总览

| # | 节点 | 改动 | 实测效果 |
|---|---|---|---|
| — | planner | 移除 `_rule_based_plan()` 降级 | LLM 不可用时直接抛异常，不再产出低质计划 |
| — | planner | Prompt V2：两阶段（分析 → 生成）+ 自适应深度 | 景观题→medium、对比题→deep、入门题→shallow，与预期完全一致 |
| #1 | analyst→writer | Citation Registry（全局 CID） | 引用链路打通，`citation_mismatch` 有了可判定的基准 |
| #2 | analyst | 深度/意图感知提示词 | deep 答案均长 1165→881（-24%） |
| #3 | fact_checker | 按子问题并发独立核查 | 核查看到完整子答案 + 原始来源片段，不再截断拼接 |
| #4 | report_writer | 注入 `research_strategy` / `report_type` | 三种报告类型结构差异化，格式符合率长期 12/12 |
| #5 | 全部 | SHORT 输出阈值 + 诊断式重试 | 第四轮触发 5 次，救回 4 次（含 report_writer deep ×3） |
| #6 | fact_checker | issue 类型路由 + 引用修正闭环 | `citation_mismatch` 不再误触发补充轮 |
| #8 | analyst/fc | **JSON Schema strict 结构化输出** | **总 issues 77→5，citation_mismatch 33→0** |
| #9 | analyst/fc | few-shot 示例 | `needed_evidence` 正确率 43%→91% |

---

## 二、Planner：两阶段结构化提示词

`app/prompts/planner.md` 从简单指令升级为两阶段 Prompt。**Phase 1 强制 LLM 先分类再规划**，分类结果存入 `research_strategy` 并向下游传递。

**① Intent（5 类）** `quick_overview` / `deep_investigation` / `comparison` / `how_to` / `trend_tracking`
**② Domain（7 类）** technology / business / science / legal / education / policy / general —— 依据是"回答该问题**主要需要哪类知识**"，不看话题表面
**③ Depth（自适应）**

| Depth | 触发 | 子问题数 | 搜索词/题 |
|---|---|---|---|
| shallow | `quick_overview`；入门级 | 2–3 | 2–3 |
| medium | 标准问题；单域 `how_to` / `trend_tracking` | 4–5 | 3–4 |
| deep | `deep_investigation` / `comparison`；跨域 | 5–7 | 4–5 |

**关键设计**：深度由 LLM 从 intent 推断，用户传的 `search_depth` **降级为软偏好**——它和 `report_type` 一起走 `hint_block` 拼进 user_message（"仅供参考，可根据问题实际情况调整"），不再硬控制。

**④ Dimensions（7 种）** conceptual / technical / comparative / practical / trend / critical / contextual，选 2–4 种；部分 intent 有强制项（`comparison` 必含 `comparative`）。

Phase 2 据此生成计划，用**领域专属角度库**替代通用模板（如 science → 基础原理·实验证据·科学共识·开放问题·应用前景），并加两条硬约束：`comparison` 每个对比对象一个子问题 + 一个综合权衡题；`how_to` 按"前提 → 核心步骤 → 验证"排序。搜索词要求中英混合且**语义多样**（不只是换词）。

> 完整字段结构见 `app/prompts/planner.md` 与 `app/graph/state.py`，此处不重复贴 JSON。

同期把 `analyst_node` 的子问题分析从串行 `for` 改为 `asyncio.gather` + `Semaphore(5)`：耗时从 `N × T` 变成 `max(各子问题耗时)`。

---

## 三、Analyst / Fact-checker：引用体系与输出约束

### 引用体系断裂 → Citation Registry（#1）

改前是三套编号各说各话：analyst 对**每个子问题**独立从 R1/S1 编起，report_writer 的来源列表**根本没有 ID**，两套编号还在同一条 user_message 里同时出现。

改后在并发分析**之前**对全部通过评估的文档分配全局稳定 ID（`C01`/`C02`…）建成 `citation_registry`，贯穿 analyst 上下文标签、fact_checker 来源标注（来源数 3→5 条）、report_writer 参考列表 `[C01] [标题](url)`。验证：第四轮思维链里出现 `"分析结果引用了C01、C02、C03"`，全链路 CID 一致。

### 静默截断 → SHORT 阈值 + 诊断式重试（#5）

模型上下文溢出或 thinking 耗尽 token 时会**静默返回极短内容**（实测 report_writer 只生成 60 字符、1.6s 完成），节点无感知照常放行。

`_llm_with_short_retry()` 检测 `len(output) < threshold` 后，注入含**错误类型 / 实际长度 / 上次输出 / 可能原因 / 内容要求**的诊断消息重试一次；仍失败则 `logger.error` 放行（不中断流程）。阈值按节点分别设定（planner 720、analyst 276、fact_checker 50、report_writer 三种类型各一）。

> 遗留 Bug：`report_writer_summary` 阈值 956 取自实测最短值 ×1.2，但 summary 格式本身要求 600 字以内 —— 897 字符的正常输出会被误触发重试。**应调低至 300。**

### issue 一视同仁 → 类型路由（#6 #7）

改前四种 issue 类型全部触发同一路径：标 `fact_check_passed=False` 并进入下一轮补充搜索。但 `citation_mismatch` 不需要补搜，它只需要告诉 analyst 改引用。

```
_SERIOUS_ISSUE_TYPES = {contradiction, overclaim, insufficient_evidence}
  serious          → any_failed=True，触发补充轮；needed_evidence 必填具体补证描述
  citation_mismatch → 写入 state["citation_mismatches"]，不影响 fact_check_passed；needed_evidence 强制留空
```

`_revise_citations()`（`routes_research.py`，每轮 `_run_round()` 后）：有 mismatch 且无 serious issue 时，以 revision 模式**只重跑有 mismatch 的子问题** → **再跑一次 fact_checker 验证** → 重出报告。早期设计是直接把 `fact_check_passed` 置 True 不做验证，无法确认修正是否有效。

### 自由文本 → JSON Schema strict（#8 #9）

analyst 和 fact_checker 原本都以自由文本回答，LLM 自行决定引用格式和字段，解析脆弱、引用随意（33 条 `citation_mismatch`/轮）。

- `_ANALYST_RESPONSE_FORMAT`：`sub_question_id / answer / citations[] / confidence / evidence_gap`
- `_FACT_CHECKER_RESPONSE_FORMAT`：`passed / issues[]{type,claim,reason,needed_evidence} / follow_up_queries`，`type` 为四值枚举
- `response_format` 经 `_llm_with_short_retry(**llm_kwargs)` 透传到 `LLMService`

few-shot（#9）则重写了两个提示词：`fact_checker.md` 加 `needed_evidence` 规则表 + 4 条示例（pass / insufficient_evidence / citation_mismatch 空 NE / overclaim）；`analyst.md` 加 3 条，第 3 条专门示范"❌ 引用不相关来源 vs ✓ 只引用真正支持该声明的来源"。

---

## 四、Benchmark：八轮实测

固定 12 个任务（`benchmark/results/analyst_*.json` 存全部原始数据），每轮只改一个杠杆。

| 轮次 | 本轮新增 | issues 均值/题 | 格式符合 | 挂钟 |
|---|---|---|---|---|
| 一 | 基线 —— RAG 因 qdrant-client API 断层完全失效 | 14.3 | 12/12 | 3252s（并发 1）|
| 二 | RAG 修复（并发 6）| 10.9† | 11/11 | 825s |
| 三 | 并发调优：任务 4 / analyst LLM 3 | 11.9 | 10/12 | 1288s |
| 四 | #1 Citation Registry、#5 SHORT 重试 | 7.9 | 12/12 | 1234s |
| 六 | #6 issue 路由 + `needed_evidence` | 6.4（共 77 条）| 12/12 | — |
| 七 | #9 few-shot | 6.4（共 77 条）| 12/12 | — |
| **八** | **#8 JSON Schema strict** | **0.4（共 5 条）** | 12/12 | 1104s |

† 第二轮有 3 个任务 embedding OOM 产出 0 chunks，人为压低了均值。第五轮是 v0.1.4 参照运行，fact_checker 用旧格式（无 `issues` 数组），不作质量基准。

三条关键结论：

1. **few-shot 改行为，schema 改结构。** 第七轮 issue 数量与第六轮一模一样（77 vs 77），但 `needed_evidence` 正确率 43%→91%；第八轮换 JSON Schema strict，`citation_mismatch` 33→**0**、`overclaim` 7→**0**、总 issues **-94%**。约束格式才是决定性的。
2. **剩下的 5 条是真实知识空白**，不是引用错误——涉及 RAG 重排序细节和幻觉检测方法，引用到的来源只有标题/摘要片段，属检索层信息不足。
3. **`_revise_citations` 路径存在但未被触发**：引用质量上来后 fact_checker 不再产 `citation_mismatch`，该路径作为安全网保留。

**瓶颈不在 LLM。** 第八轮串行耗时构成：

| 节点 | 类型 | 均值 | 串行占比 |
|---|---|---|---|
| evidence_builder | Embed | 211.5s | **58.9%** ← 主瓶颈 |
| retriever | Search | 67.6s | 18.9% |
| report_writer / analyst / planner / fact_checker | LLM | 合计 79.7s | 22% |

这条数据直接导向了后来的模型服务化与 FP16，见 [模型服务化与显存内存优化](模型服务化与显存内存优化.md)。

**benchmark 暴露的生产 bug**：

| Bug | 现象 | 处置 |
|---|---|---|
| A (P0) | qdrant-client 1.18.0 删除 `search()`，RAG 完全失效 | 改 `query_points(query=...).points` |
| C (P1) | 并发 6 时 3 个任务 `ONNXRuntimeError` 分配 3.2GB 失败，0 chunks | 并发降到 4 自然消除；**4 是当前硬件的 ONNX 安全上限** |
| D (P2) | 并发 6 触发 Tavily / LLM 限速，任务 03 完全失败 | 任务并发 4 + analyst LLM 并发 3 |
| E (P2) | report_writer 静默产出 60 字符 | #5 SHORT 阈值 + 重试 |
| F | `report_writer_summary` 阈值 956 误触发 | **未修**，应调至 300 |
| G | `passed = not any_failed`，任意 issue 即整体 false（0/12） | #6 只有 serious issue 影响 passed |

---

## 五、多轮补充研究修复（v0.1.2）

改前：第 2 轮把 `sub_questions` 重置为空，planner 对同一 query 从头规划出几乎相同的计划；`follow_up_queries` 作为初始 `search_queries` 传入后**立刻被 planner 覆盖，从未被搜索**。第 2 轮等于原样重做第 1 轮。

三处联动修复：

- `routes_research.py`：第 2 轮 `new_state` 携带第 1 轮的 `sub_questions` / `research_plan` / `research_strategy` / `follow_up_queries`
- `planner_node`：`sub_questions` 非空且 `current_round > 1` 时直接透传，**跳过 LLM 调用**（省 5–10s）
- `retriever_node`：同条件下只搜 `follow_up_queries[:5]`，不重复第 1 轮关键词

效果：第 2 轮从"重复第 1 轮"变成"按 fact_checker 识别的缺口补证"。后续的完整方案（多轮增量补证：通过的子问题复用、失败的才补）见 [v0.2.1](version/v0.2.1.md)。

---

## 六、Backlog

| 方向 | 说明 |
|---|---|
| **搜索词语义去重** ⭐ | planner 产 20–30 条搜索词，`"LangGraph 概述" ≈ "LangGraph overview"` 大量重复。出 planner 后做向量余弦过滤（阈值 0.85，复用已有本地 embedding，无新依赖），预计减少搜索 30–50% |
| `report_writer_summary` 阈值 | 956 → 300（Bug F） |
| 多视角并行规划 | 并发发 2–3 个不同角色（学术研究者 / 从业者 / 终端用户）的规划请求再合并去重。时延不变，代价 3× token |
| 多轮感知规划完整版 | round ≥ 2 时让 planner 读上一轮 `issues` + `sub_answers`，用 LLM 生成**针对证据缺口的新子问题**（需新增 `planner_followup.md`）。当前是简化版：直接复用第 1 轮结构 |
| 规划反思 | retriever 后插一次轻量 LLM 评估"搜索结果 vs 研究计划"覆盖度，不足则 re-plan。代价 +2–3s 且要给 LangGraph 加条件边 |
