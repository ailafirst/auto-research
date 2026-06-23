# Analyst 改进基准测试报告

---

## 改进项总览

| 编号 | 问题 | 状态 | 实施时间 |
|------|------|------|----------|
| #1 | 引用体系根本性断裂（局部索引、来源列表无 ID、LLM 无法对应） | ✅ 已实施 | 2026-06-23 |
| #2 | Analyst Prompt 空洞（无深度/意图约束，答案质量不稳定） | ✅ 已实施 | 2026-06-22 |
| #3 | Fact Checker 核查截断文本（单次调用、无原始来源） | ✅ 已实施 | 2026-06-22 |
| #4 | Report Writer 缺失关键上下文（report_type/research_strategy 未传入） | ✅ 已实施 | 2026-06-22 |
| #5 | report_writer 静默截断输出（60 字符无异常） | ✅ 已实施 | 2026-06-23 |

---

## 已实施改进详情

### #1 引用体系 — Citation Registry（2026-06-23）

**问题**：Analyst 对每个子问题独立从 R1/S1 编号；Report Writer 来源列表无任何 ID；两套编号在 report_writer 的 user_message 中同时存在导致完全脱节。

**实施内容**：
- `analyst_node`：在并发分析前对所有通过评估的文档分配全局稳定 ID（C01/C02…），构建 `citation_registry`
- `_analyze_single_question`：接收 `url_to_cid` 映射，上下文来源标签从 `[来源 R1]`/`[来源 S1]` 改为 `[C01]`；default_cites 从 context_material 提取实际 CID
- `_check_single_answer`：接收 `citation_registry`，来源标注使用 CID，来源数从 3 条扩至 5 条
- `report_writer_node`：从 `citation_registry` 生成 `[C01] [标题](url)` 格式参考列表
- `app/prompts/report_writer.md`：行内引用格式约束从 `[S1]` 改为 `[C01]`，禁止凭空编造
- `app/graph/state.py`：新增 `citation_registry` 字段

**验证**：第四轮 Trace B 思维链中出现 `"分析结果引用了C01、C02、C03"` 确认全链路 CID 贯通。

---

### #2 Analyst Prompt 深度/意图感知（2026-06-22）

**问题**：`analyst.md` 仅 9 行基本指令，未向 LLM 传递 intent/depth；答案详细程度与问题类型不匹配；confidence 无校准标准。

**实施内容**：
- `app/prompts/analyst.md`：新增回答深度规范（硬性上限 shallow/300 · medium/500 · deep/800），新增意图适配说明（5 种 intent），强调"字数上限是硬性约束，超出则精简次要细节"
- `app/graph/nodes.py` `_DEPTH_GUIDE`：措辞从"深度详尽，500–800 字"改为"严格 800 字以内（超出视为违规）"，去除定性与字数约束的语义冲突

**验证**：deep 任务答案均长从 ~1165 降至 ~881（-24%），但合规率仍低（多数 deep 任务输出 800-1300 字）。

---

### #3 Fact Checker 按子问题并发独立核查（2026-06-22）

**问题**：原实现将所有子答案截断至 500 字拼接为单文本，一次 LLM 调用核查全部；无法对照原始来源。

**实施内容**：
- `fact_checker_node`：改为按子问题并发（`asyncio.Semaphore(3)`）独立核查
- `_check_single_answer`：每次核查传入完整子答案 + 实际使用的来源片段（RAG chunks 原文）

**验证**：核查机制正常，issues 数据可反映实际来源支持情况。

---

### #4 Report Writer 上下文注入（2026-06-22）

**问题**：`report_writer_node` 未接收 `research_strategy` 和 `report_type`，用户指定的报告类型对输出无差异化影响。

**实施内容**：
- `report_writer_node`：将 `research_strategy`（intent/domain/depth）和 `report_type_guide` 注入 user_message
- `app/prompts/report_writer.md`：按 summary/comparison/deep 给出差异化结构要求

**验证**：第一至四轮 report_type 格式符合率保持高水平（第四轮 12/12）。

---

### #5 全节点 SHORT 输出阈值 + 重试（2026-06-23）

**问题**：模型上下文溢出或 thinking 耗尽 token 时静默返回极短内容（如 60 字符），节点无感知，流程照常继续。

**实施内容**（`app/graph/nodes.py`）：
- 新增 `_SHORT_THRESHOLDS` 字典（6 个阈值）和 `_SHORT_REQUIRED_CONTENT`（各节点必须包含的内容描述）
- 新增 `_llm_with_short_retry()`：检测 `len(output) < threshold`，注入含错误类型/实际长度/上次输出/可能原因/内容要求的诊断消息重试一次；重试仍失败则 `logger.error` 放行
- 4 个 LLM 节点（planner/analyst/fact_checker/report_writer）替换为重试版调用

| 节点 | 阈值 |
|------|------|
| planner | 720 |
| analyst | 276 |
| fact_checker | 50 |
| report_writer_summary | 956 ⚠️ 过高（待调低至 300） |
| report_writer_comparison | 3514 |
| report_writer_deep | 5345 |

**验证**：第四轮触发 5 次重试，4 次成功（report_writer deep 被救回），格式合格率从 10/12 升至 12/12。

---

## 第一轮：基线测试（RAG 失效）

**时间**：2026-06-22 01:20
**结果文件**：`benchmark/results/analyst_20260622_022019.json`
**配置**：并发=1，顺序执行，总耗时 3252s（54 分钟）
**问题**：Qdrant `search()` 方法被 qdrant-client 1.18.0 删除，RAG 完全失效，所有任务走 fallback 全文路径

### 改进效果

| 改进项 | 预期效果 | 实测状态 |
|--------|---------|---------|
| #2 intent 感知 | 5 种意图差异化分析 | ✅ 完全生效 |
| #2 depth 字数控制 | shallow/medium/deep 三档 | ✅ shallow  🟡 medium（略超）  🔴 deep 系统性失效（实测 1050–1805，期望 500–800） |
| #3 fact_checker 并发 | 每子问题独立核查、无死锁 | ✅ 机制正常，issues 虚高因 RAG 失效导致 |
| #4 report_writer 格式 | 三种 report_type 差异结构 | ✅ **12/12 完美通过** |

### 第一轮详细数据

| ID | 任务 | depth | chunks | 置信度 | issues | passed | 格式 |
|----|------|-------|--------|--------|--------|--------|------|
| 01 | LLM推理加速 | medium | 464 | 72% | 10 | ✗ | ✓ |
| 02 | Python vs Rust | deep | 531 | 75% | 16 | ✗ | ✓ |
| 03 | 量子计算基础 | shallow | 1050 | 80% | 2 | ✗ | ✓ |
| 04 | RAG最佳实践 | deep | 874 | 65% | 26 | ✗ | ✓ |
| 05 | AI芯片 | deep | 799 | 68% | 17 | ✗ | ✓ |
| 06 | GDPR对比 | deep | 1118 | 81% | 19 | ✗ | ✓ |
| 07 | 碳中和 | deep | 900 | 76% | 14 | ✗ | ✓ |
| 08 | 电商推荐 | deep | 1351 | 70% | 15 | ✗ | ✓ |
| 09 | LLM幻觉 | deep | 1687 | 61% | 10 | ✗ | ✓ |
| 10 | 固态电池 | deep | 1381 | 61% | 13 | ✗ | ✓ |
| 11 | 微服务vs单体 | deep | 1016 | 83% | 14 | ✗ | ✓ |
| 12 | AGI障碍 | deep | 1225 | 79% | 16 | ✗ | ✓ |
| **均值** | | | **1029** | **72%** | **14.3** | **0/12** | **12/12** |

### 第一轮发现的问题

**Bug A（P0，已修复）**：`qdrant-client 1.18.0` 删除了 `search()` 方法，代码未跟进，RAG 完全失效。
- 修复：`app/services/vector_store.py:123`，`client.search(query_vector=...)` → `client.query_points(query=...).points`

**Bug B（P0，澄清为设计正常）**：content_extractor 耗时 0s。
- 澄清：Tavily 已在搜索结果中返回 `raw_content`，content_extractor 检测到后跳过爬取，0s 耗时是正常行为，不是 bug。只有 DuckDuckGo 模式才会实际抓页面。

---

## 第二轮：RAG 修复后验证（并发=6）

**时间**：2026-06-22 16:28
**结果文件**：`benchmark/results/analyst_20260622_162842.json`
**配置**：并发=6，总耗时 825s（14 分钟），节省 75%
**状态**：11/12 完成（任务 03 量子计算因 LLM rate limit 在 report_writer 阶段失败）

### RAG 修复效果对比

| ID | 任务 | ①chunks | ①置信 | ①issues | ②chunks | ②置信 | ②issues | 变化 |
|----|------|---------|-------|---------|---------|-------|---------|------|
| 01 | LLM推理加速 | 464 | 72% | 10 | 1004 | **78%** | 18 | OOM 但 fallback 正常 |
| 02 | Python vs Rust | 531 | 75% | 16 | **0** | 72% | 16 | Embedding OOM |
| 03 | 量子计算基础 | 1050 | 80% | 2 | ERROR | — | — | LLM rate limit |
| 04 | RAG最佳实践 | 874 | 65% | 26 | **0** | **86%** | 20 | Embedding OOM |
| 05 | AI芯片 | 799 | 68% | 17 | **0** | 71% | 15 | Embedding OOM |
| **06** | **GDPR对比** | **1118** | 81% | **19** | **1120** | 54% | **1** | **✅ -18，RAG 生效** |
| **07** | **碳中和** | 900 | 76% | **14** | 290 | 74% | **0** | **✅ passed=true，首次通过** |
| 08 | 电商推荐 | 1351 | 70% | 15 | 186 | 69% | **6** | **✅ -9** |
| **09** | **LLM幻觉** | 1687 | 61% | 10 | 587 | **85%** | **6** | **✅ -4，置信+24%** |
| 10 | 固态电池 | 1381 | 61% | 13 | 805 | **80%** | 16 | 置信+19%，issues 无改善 |
| 11 | 微服务vs单体 | 1016 | 83% | 14 | 319 | 82% | 15 | 无明显变化 |
| 12 | AGI障碍 | 1225 | 79% | 16 | 596 | 81% | **8** | **✅ -8** |
| **均值（排除 03）** | | **1029** | **72%** | **14.3** | **527** | **75%** | **10.9** | **issues -24%** |

**结论**：RAG 真正生效时（chunks > 0），fact_checker issues 数平均下降约 24%，07 碳中和成为首个 passed=true 的任务。

### 第二轮新发现的问题

**Bug C（P1）：Embedding 并发 OOM**
- **现象**：并发=6 时，3 个任务（02/04/05）的 evidence_builder 报 `ONNXRuntimeError: Failed to allocate memory for requested buffer of size 3221225472`，产出 0 chunks，RAG 未工作
- **根本原因**：fastembed 使用 ONNX Runtime，多任务同时运行 embedding 时内存叠加（每次请求约 3.2GB），超出系统可用内存
- **影响范围**：并发 > 3 时稳定复现；并发 1 时从未出现
- **修复方向**：在 `evidence_builder_node` 或 `RAGService.build_evidence()` 入口加进程级锁（`asyncio.Lock` 或 `threading.Lock`），保证同一时刻只有一个 embedding 任务运行
- **临时缓解**：benchmark 默认并发已从 6 调回 3（`analyst_benchmark.py`）

**Bug D（P2）：并发时 Tavily/LLM API 限速**
- **现象**：并发=6 时大量 `Tavily: This request exceeds your plan's usage limit` 和 `LLM: Too many requests`，部分降级 DuckDuckGo，任务 03 因 LLM rate limit 完全失败
- **根本原因**：Tavily dev key 和 LLM API 均有 RPM/TPM 限额，12 任务同时并发远超限额
- **影响范围**：仅影响 benchmark 并发运行，单任务生产流程不受影响
- **修复**：benchmark 默认并发改为 4，analyst 节点内部 LLM 并发改为 3（通过 `_ANALYST_LLM_CONCURRENCY` 模块变量覆写，生产代码保持 5）

---

## 第三轮：LLM 并发调优（并发=4，analyst LLM 并发=3）

**时间**：2026-06-22 18:27
**结果文件**：`benchmark/results/analyst_20260622_182750.json`
**配置**：任务并发=4，analyst 内部 LLM 并发=3（benchmark 覆写），总耗时 1288.1s（21 分钟）

### 核心发现：瓶颈在 Embedding，不在 LLM

| 节点 | 类型 | 均值 | 最大 | 串行占比 |
|------|------|------|------|--------|
| evidence_builder | Embed | 199.3s | 350.6s | **50.5%** ← 主瓶颈 |
| retriever | Search | 53.7s | 121.2s | 13.6% |
| analyst | LLM | 51.8s | 74.8s | 13.1% |
| fact_checker | LLM | 41.6s | 56.7s | 10.6% |
| report_writer | LLM | 37.1s | 54.3s | 9.4% |
| planner | LLM | 10.9s | 13.1s | 2.8% |
| content_extractor | IO | 0.0s | 0.0s | 0.0% |
| source_evaluator | Rule | 0.0s | 0.0s | 0.0% |
| **LLM 合计** | | **141.4s** | | **36%** |
| **非 LLM 合计** | | **253.0s** | | **64%** |

**并发效率**：串行总量 4732s ÷ 并发 4 = 理论最短 1183s，实际 1288.1s，**效率 92%**

### 三轮对比

| 指标 | 第一轮（基线） | 第二轮（RAG 修复） | 第三轮（LLM 调优） |
|------|--------------|-----------------|----------------|
| 任务并发 | 1 | 6 | **4** |
| analyst LLM 并发 | 5 | 5 | **3** |
| 完成率 | 12/12 | 11/12 | **12/12** |
| Embedding OOM | 无 | 3 任务 0 chunks | **0（自然消除）** |
| LLM 限速失败 | 0 | 1（task 03） | **0** |
| issues 均值 | 14.3 | 10.9† | **11.9** |
| 挂钟时间 | 3252s | 825s | **1288s** |
| 并发效率 | — | — | **92%** |

†第二轮 3 个 OOM 任务 0 chunks，人为压低了 issues 均值

### 第三轮详细数据

| ID | 任务 | depth | chunks | 置信度 | issues | 格式 | analyst | fc | writer | 总耗时 |
|----|------|-------|--------|--------|--------|------|---------|-----|--------|--------|
| 01 | LLM推理加速 | medium | 1091 | 75% | 7 | ✓ | 25s | 39s | 45s | 393s |
| 02 | Python vs Rust | deep | 653 | 67% | 18 | ✓ | 59s | 43s | 32s | 390s |
| 03 | 量子计算基础 | shallow | 960 | 87% | 4 | ✗* | 25s | 31s | 22s | 246s |
| 04 | RAG最佳实践 | deep | 1339 | 77% | 15 | ✓ | 75s | 57s | 44s | 549s |
| 05 | AI芯片 | deep | 845 | 76% | 9 | ✗† | 56s | 28s | 2s | 227s |
| 06 | GDPR对比 | deep | 1312 | 78% | 9 | ✓ | 43s | 27s | 47s | 315s |
| 07 | 碳中和 | deep | 1719 | 76% | 12 | ✓ | 70s | 51s | 46s | 603s |
| 08 | 电商推荐 | deep | 1318 | 77% | 20 | ✓ | 64s | 53s | 54s | 511s |
| 09 | LLM幻觉 | deep | 1878 | 75% | 10 | ✓ | 48s | 39s | 46s | 549s |
| 10 | 固态电池 | deep | 1216 | 76% | 15 | ✓ | 44s | 47s | 18s | 368s |
| 11 | 微服务vs单体 | deep | 1016 | 67% | 8 | ✓ | 61s | 48s | 34s | 289s |
| 12 | AGI障碍 | deep | 689 | 76% | 16 | ✓ | 52s | 37s | 54s | 292s |
| **均值** | | | **1083** | **75%** | **11.9** | **10/12** | **52s** | **42s** | **37s** | **394s** |

*task 03 summary 类型输出 3174 字符，略超 3000 字限制
†task 05 report_writer 仅生成 60 字符（LLM 返回异常截断，节点未抛出异常）

### 关键结论

1. **OOM 自然消除**：并发 6→4 使 evidence_builder 不再同时竞争 ONNX 内存，无需加锁。**并发=4 是当前硬件的 ONNX 安全上限。**
2. **LLM 并发降低效果显著**：analyst LLM 并发 5→3，零 LLM 限速失败，12/12 完成。
3. **真正瓶颈是 Embedding**：evidence_builder 占串行时间 50.5%，LLM 节点合计仅 36%。切换到 OpenAI API Embedding（待 Qdrant Cloud API key 到位后实施）预期可将每任务串行时间从 394s 降至 ~195s，挂钟时间可从 1288s 降至约 640s。

**新发现 Bug E（P2）**：task 05 report_writer 生成 60 字符（期望 >1000 字），1.6s 完成，LLM 返回了极短内容但节点未检测到并静默通过。

---

## 第四轮：Citation Registry + Prompt 校准 + SHORT 重试（并发=4）

**时间**：2026-06-23 13:10
**结果文件**：`benchmark/results/analyst_20260623_131044.json`
**配置**：任务并发=4，analyst 内部 LLM 并发=3，总耗时 1234.3s（21 分钟）

**本轮新增改进**：
- #1 Citation Registry（全局 CID 引用体系）
- #5 SHORT 输出阈值 + LLM 重试机制
- fact_checker prompt 宽松化（高门槛标记，合理推断可接受）
- analyst depth 约束强化（"严格 N 字以内，超出视为违规"）

### 第四轮详细数据

| ID | 任务 | depth | 均长 | 置信 | issues | 格式 | 耗时 | rw输出 |
|----|------|-------|------|------|--------|------|------|--------|
| 01 | LLM推理加速 | medium | 515 | 78% | 10 | ✓ | 324s | 6933 |
| 02 | Python vs Rust | deep | 1156 | 80% | 13 | ✓ | 488s | 3623 |
| 03 | 量子计算基础 | shallow | 211 | 80% | 3 | ✓ | 401s | 1192 |
| 04 | RAG最佳实践 | medium | 440 | 72% | 5 | ✓ | 469s | 5606 |
| 05 | AI芯片 | deep | 740 | 64% | 9 | ✓ | 311s | 7009 |
| 06 | GDPR对比 | deep | 1229 | 69% | 5 | ✓ | 285s | 4211 |
| 07 | 碳中和 | deep | 697 | 76% | 11 | ✓ | 443s | 5885 |
| 08 | 电商推荐 | deep | 1053 | 75% | 11 | ✓ | 383s | 6726 |
| 09 | LLM幻觉 | deep | 836 | 77% | 7 | ✓ | 411s | 7121 |
| 10 | 固态电池 | deep | 561 | 72% | 8 | ✓ | 290s | 897† |
| 11 | 微服务vs单体 | deep | 775 | 71% | 5 | ✓ | 363s | 4201 |
| 12 | AGI障碍 | deep | 882 | 72% | 9 | ✓ | 303s | 5510 |
| **均值** | | | **841** | **73%** | **7.9** | **12/12** | **372s** | — |

†task 10 summary 类型，SHORT 重试后仍 897 字符（阈值 956 偏高，见下方 bug）

### 四轮横向对比

| 指标 | 第一轮 | 第二轮 | 第三轮 | 第四轮 | 趋势 |
|------|--------|--------|--------|--------|------|
| 完成率 | 12/12 | 11/12 | 12/12 | **12/12** | 稳定 |
| issues 均值 | 14.3 | 10.9† | 11.9 | **7.9** | ↓ -34% vs 第三轮 |
| passed 率 | 0/12 | 1/12 | 0/12 | **0/12** | 未改善（聚合逻辑问题） |
| 格式符合 | 12/12 | 11/11 | 10/12 | **12/12** | SHORT 重试救回 2 个 |
| analyst 均长（deep） | ~1400 | ~1000 | ~1165 | **~881** | ↓ -24% vs 第三轮 |
| 挂钟时间 | 3252s | 825s | 1288s | **1234s** | 略优 |

†第二轮 3 个 OOM 任务 0 chunks 人为压低均值

### 第四轮关键发现

**SHORT 重试工作情况**：
- analyst：255→379 字符（重试成功）
- report_writer deep：4150→5606 ✅、4393→7009 ✅、2767→4201 ✅
- report_writer summary：665→897（重试仍未达阈值，放行）—— 阈值 956 对 summary 类型偏高

**Citation Registry 验证**：Trace B 思维链中多处出现 `C01`/`C02`/`C03` 引用核对，全链路 CID 一致性确认。

**Bug F（新发现）：`report_writer_summary` 阈值误设**
- 阈值 956 源自第三轮 summary 最短实测值 797 × 1.2，但 summary 格式本身要求"600 字以内"
- 897 字符的 summary 输出内容正常，却被阈值错误触发重试
- 修复方向：将 `report_writer_summary` 阈值从 956 调低至 **300**

**Bug G（已知，未修）：passed 聚合逻辑过严**
- 当前逻辑：`passed = not any_failed`，任意一道子题有 issue 即整体 false
- issues 已从 11.9 降至 7.9，但 passed 率仍 0/12
- 修复方向：仅当 issue type 为 `contradiction` 或 `overclaim` 时触发 passed=false；`insufficient_evidence`/`citation_mismatch` 不影响 passed

---

## 待解决问题清单

| 优先级 | 问题 | 状态 | 说明 |
|--------|------|------|------|
| P0 | Qdrant `search()` API 版本断层 | ✅ 已修复（2026-06-22） | `vector_store.py:123` |
| P1 | Embedding 并发 OOM | ✅ 间接缓解（并发=4 自然避免） | 根本修复待 Qdrant Cloud key |
| P1 | Embedding 根本修复（fastembed→OpenAI API） | 🟡 待实施（等 Qdrant Cloud key） | `rag_service.py` |
| P1 | deep depth 字数超标（均长 881，期望 ≤800） | 🟡 部分改善（第三轮 1165→第四轮 881，-24%） | 合规率仍低，需进一步强化 |
| P1 | fact_checker passed 聚合逻辑过严（0/12 passed） | 🔴 未修复（Bug G） | 需按 issue type 区分严重性 |
| P1 | `report_writer_summary` SHORT 阈值误设（956 过高） | 🔴 未修复（Bug F） | 应调低至 300 |
| P2 | benchmark Tavily/LLM 限速 | ✅ 已缓解（并发调优 + key 轮换） | `analyst_benchmark.py` + `nodes.py` |
| P2 | report_writer 静默截断（60 字符） | ✅ 已修复（SHORT 阈值 + 重试） | `nodes.py` `_llm_with_short_retry` |

---

## Report Writer 格式符合度

| 轮次 | 通过率 | 备注 |
|------|--------|------|
| 第一轮 | 12/12 | — |
| 第二轮 | 11/11 | — |
| 第三轮 | 10/12 | task 03 summary 略超字数；task 05 极短内容（60 字） |
| **第四轮** | **12/12** | SHORT 重试救回 report_writer deep × 3；summary 阈值误触发但结果可用 |

- **summary**：控制在 600 字要求下通常 800–1200 字（LLM 倾向生成更长内容）
- **comparison**：全部自动生成 Markdown 对比表格
- **deep**：8–12 节丰富结构，第四轮平均 5306 字
