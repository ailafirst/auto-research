# RAG 改进方案

> v1.2 · 2026-06-24 | 评测工具：`benchmark/test_rag.py` / `benchmark/benchmark_rag.py`

---

## 一、当前实现

```
TextChunker（段落切分, chunk_size=800, overlap=200字符≈40词）
    → BAAI/bge-m3（1024维，多语言，模块级单例）
    → Qdrant 内存模式（task_id 隔离）
    → cosine 检索（threshold=0.45筛选, top_k=6）
    → 共享证据池 → LLM 分析
```

---

## 二、已实施改进

| # | 内容 | 版本 |
|---|------|------|
| A1 | RAGService 模块级单例，消除冷启动 | v0.1.3 |
| A2 | chunk_overlap 120 → 200（有效40词） | v0.1.3 |
| A3 | 筛选阈值 0.3 → 0.45 | v0.1.3 |
| A4 | Chunk 拼接文档标题增强上下文 | v0.1.3 |
| B1 | 升级 bge-m3（1024维，支持中文） | v0.1.3 |
| P1 | Analyst prompt 强化：声明必须内联引用 + 禁止训练知识 | v0.1.4 |
| B2 | CrossEncoder Reranker（BAAI/bge-reranker-v2-m3），`RERANKER_ENABLED=true` 开启 | v0.1.4 |

---

## 三、当前不足

| 问题 | 表现 | 待解决方案 |
|------|------|-----------|
| Q1 无查询改写 | 问句与文档答案语义鸿沟（asymmetric retrieval），如"如何评估X风险？"检索不到"X风险包括……" | HyDE（C1）/ Multi-Query（C3）|
| Q2 无重排序 | 向量近似排序，相似度高≠真正相关 | bge-reranker-v2-m3（B2）|
| Q3 纯向量检索 | 专有名词、版本号等精确词无法命中 | Hybrid Search Dense+Sparse（C2）|
| Q4 评估循环验证 | bge-m3 同时用于索引和评估，`chunk_score_mean ≈ context_precision`（差值<0.02）| LLM Judge（C4）|
| Q5 阈值偏低 | Task 04 测试显示有效分数下界≈0.65，0.45无实际过滤作用 | 待全量数据后调整 |
| Q6 评估盲区 | Context Recall / Answer Correctness 需 ground truth；Chunk Utilization 未提取 | C4 / C5 |

---

## 四、RAG 评估指标体系

### 4.1 Chunking 质量（间接信号）

| 指标 | 计算方式 | 参考值 |
|------|----------|-------|
| 语义凝聚度 | chunk 内各句 embedding 平均相互余弦相似度 | 好 > 0.75，差 < 0.5 |
| 边界渗漏 | 相邻 chunk 首尾 embedding 相似度 | 好 < 0.6，差 > 0.8（切断连贯句）|
| 长度分布 | 字符数 P10/P50/P90 | P50 ≈ 600–800，P10<100 说明碎片化 |
| 引用命中率 | analyst 引用 URL 能回溯到 chunk 的比例 | 好 > 80%，差 < 50% |

**当前问题**：chunk_size=800字符≈400 token，远低于 bge-m3 上限 8192；HTML转文本残留标签污染；overlap 40词对长复合句偏短。

### 4.2 Query 改写/生成质量

当前是 **query generation**（planner 生成 search_queries），非 query rewriting。若引入改写，评估维度：

| 指标 | 说明 | 合理范围 |
|------|------|---------|
| 意图保真度 | `cosine_sim(原始问题, 改写问题)` | 不应低于 0.65（跑偏）|
| 检索提升率 | 改写后 top-1 分数 - 改写前 | 正值有效 |
| 关键词扩展率 | 改写后 unique 关键词 / 原始关键词 | > 1.5 说明有效扩展 |

### 4.3 检索精准度

**有 ground truth**：Precision@K、NDCG@K、MRR（当前无法计算）

**无 ground truth（当前代理指标）**：

| 指标 | 含义 | 已实现 | 局限 |
|------|------|--------|------|
| chunk_score_mean | Qdrant cosine 均值 | ✅ | 与 context_precision 循环验证 |
| threshold_pass_rate | 分数 > 0.45 的比例 | ✅ | 阈值合理性待验证 |
| context_precision (embed) | `mean(cosine_sim(q_emb, c_emb))` | ✅ | 同一 encoder，circular validation |
| context_precision (LLM) | LLM 判断每个 chunk 是否相关 | ❌ | 每 chunk 一次 LLM 调用 |
| chunk_utilization_rate | analyst 引用 URL / 检索 URL | ❌ | 需从 citation_registry 提取 |

**Circular Validation 诊断**：`|chunk_score_mean - context_precision| < 0.02` → 指标不独立，需 LLM judge 或异构 encoder 验证。

### 4.4 语义相关性（RAGas 风格）

**当前实现（RAGAS LLM-judge，v0.1.4+）**：

| 指标 | RAGAS 类 | 实现方式 | 需要 ground truth |
|------|---------|---------|-----------------|
| `faithfulness` | `Faithfulness` | 答案分解为 atomic claims，LLM 逐一验证是否能从 context 推导 | 否 |
| `context_precision` | `ContextPrecisionWithoutReference` | LLM 判断每个 chunk 对回答是否有用，加权精度 | 否 |

> ~~embedding-based 自定义指标（`faithfulness_sem`、`context_precision`、`answer_relevancy`）已移除~~  
> 原因：同一 encoder 同时用于索引和评估，存在 circular validation，指标不独立。

**RAGAS 待实现（需 ground truth）**：

| 指标 | 实现方式 | 需要 ground truth |
|------|---------|-----------------|
| Context Recall | LLM 判断 ground truth 每句话能否从 context 找到支撑 | **是** |
| Answer Correctness | F1(生成答案, ground truth) | **是** |

---

## 五、基准测试数据

### Task 04 历史参考基线（2026-06-24，embed-based 指标，已废弃）

> ⚠️ 以下指标为旧版 embedding-based 自定义实现，存在 circular validation，已被 RAGAS 替换，仅作历史参考。

| 子问题 | chunk_score | pass_rate | ctx_prec(embed) | faith_sem(embed) | 证据 |
|--------|------------|-----------|----------------|-----------------|------|
| q1 架构核心组件 | 0.713 | 6/6 | 0.686 | 0.795 | 充足 |
| q5 生产部署工程 | 0.656 | 6/6 | 0.652 | 0.713 | 充足 |
| **均值** | **0.692** | **1.000** | **0.679** | **0.759** | **0/5 gap** |

关键结论（仍有效）：① 0.45 阈值无实际过滤（下界≈0.65），② circular validation 差值 0.013 已确认，③ q5 生产话题检索质量最低。

### RAGAS 基线 v0（改进前，Task 04，2026-06-24）

> 旧版 analyst prompt，无 Reranker

| 子问题 | faithfulness | context_precision | chunk_score |
|--------|-------------|-----------------|------------|
| q1 核心架构组件与数据流 | 0.500 | 1.000 | 0.725 |
| q2 检索工程实践（嵌入/分块/混合检索） | 0.692 | 0.383 | 0.658 |
| q3 上下文整合与忠实回答 | 0.242 | 0.500 | 0.668 |
| q4 性能评估与迭代改进 | 0.389 | 0.917 | 0.747 |
| **均值** | **0.456** | **0.700** | **0.699** |

### RAGAS 基线 v1（analyst prompt 加强，Task 04，2026-06-24）

> 新增"每条声明必须内联引用"+"严禁使用训练知识"约束，无 Reranker

| 子问题 | faithfulness | context_precision | chunk_score |
|--------|-------------|-----------------|------------|
| q1 核心架构组件与数据流 | **1.000** | **1.000** | 0.763 |
| q2 检索优化（分块/向量化/混合检索） | **1.000** | **1.000** | 0.783 |
| q3 提示工程与上下文整合 | 0.615 | 0.250 | 0.676 |
| q4 评估指标与生产工程 | 0.895 | 0.887 | 0.697 |
| **均值** | **0.878 (+92%)** | **0.784 (+12%)** | **0.727** |

系统指标：`threshold_pass_rate=1.000`，`evidence_gap_rate=0.000`，`avg_confidence=0.825`

**解读**：prompt 约束对幻觉抑制效果极显著（+92%）。q3 的 context_precision 仅 0.250，说明"提示工程"类方法论问题检索精准度不足，是 B2 Reranker 和 C1 HyDE 的核心攻克目标。

---

## 六、后续改进方案

### 方案 B（引入新依赖）

**B2 Cross-Encoder Reranker**：检索 top-20，`BAAI/bge-reranker-v2-m3` 精排后取 top-6。预期 Recall@6 提升 15–25%，延迟 +100–300ms/子问题。

**B3 API Embedding**：本地 fastembed 替换为 LiteLLM aembedding（text-embedding-3-small，1536维）。零冷启动，约 $0.02/百万 token，适合无 GPU 环境。

### 方案 C（架构扩展）

**C1 HyDE**：子问题先由 LLM 生成假设性简短答案，用答案文本做 embedding 检索，绕过问句—文档语义鸿沟。预期提升 10–20%，每子问题增加 1 次轻量 LLM 调用。

**C2 Hybrid Search**：Qdrant RRF 融合 dense + sparse 向量（bge-m3 原生输出 sparse），解决专有名词/版本号精确匹配问题。前提：Qdrant 切换持久化模式。

**C3 Multi-Query + RRF**：每子问题 LLM 生成 3 个查询变体，分别检索后 RRF 融合，扩大召回面。每子问题增加 1 次 LLM 调用 + 3× 检索。

**C4 LLM Context Precision**：对每个 chunk 发 LLM 判断（"这段对回答问题有用吗？是/否"），替代 embedding-based 版本，消除 circular validation。

**C5 Chunk Utilization Rate**：从 sub_answers.citations 提取引用 URL，与检索 URL 取交集，比值 = 利用率。零额外成本，低 < 0.3 说明检索噪声多，高 > 0.7 表明检索质量好。

---

## 七、优先级

| 方案 | 质量提升 | 延迟 | 成本 | 状态 |
|------|---------|------|------|------|
| ~~A1–A4, B1~~ | — | — | — | ✅ 已完成 |
| C5 Chunk Utilization | 评估覆盖 | 无 | 极低 | ⭐⭐⭐⭐⭐ |
| B2 Reranker | 高 | +200~500ms | 中 | ⭐⭐⭐⭐⭐ |
| C1 HyDE | 中~高 | +0.5~1s | 中 | ⭐⭐⭐⭐ |
| C4 LLM Context Precision | 评估独立性 | +N×LLM | 中 | ⭐⭐⭐⭐ |
| C2 Hybrid Search | 高（专有名词）| +100ms | 高 | ⭐⭐⭐ |
| C3 Multi-Query+RRF | 中 | +0.5~1s | 中 | ⭐⭐⭐ |
| B3 API Embedding | 稳定性 | 稳定 | 低 | ⭐⭐⭐ 备选 |

---

## 附录：评测命令

```bash
python benchmark/test_rag.py 04 --no-report   # 单任务快速诊断
python benchmark/test_rag.py 04 --verbose      # 显示 chunk 文本摘要
python benchmark/benchmark_rag.py              # 全量 12 任务
python benchmark/benchmark_rag.py 01 04 07 --save  # 指定任务并保存 JSON
```
