# RAG 改进方案

> v1.3 · 2026-07-06 | 评测工具：`benchmark/test_rag.py` / `benchmark/benchmark_rag.py` + `rag_experiments/`

---

## 一、当前流水线

```
chunk (size=800, overlap=200) → bge-m3 embedding (1024d, GPU) → Qdrant (内存, task_id 隔离)
  → [Query Rewriting] 改写优先: 子问题 → 英文假设段落(HyDE) / 关键词组 (← 多模式)
  → [xling 双路] 对每个含中文的 query 附英文译文 (改写后逐 query 翻译)
  → 多 query 合并 → CrossEncoder rerank (bge-reranker-v2-m3)
  → 共享证据池 → LLM 分析
```

---

## 二、已实施改进

| # | 内容 | 版本 |
|---|------|------|
| A1–A4 | RAGService 单例、overlap 120→200、阈值 0.3→0.45、拼接标题 | v0.1.3 |
| B1 | bge-m3 升级（1024d, 中文支持） | v0.1.3 |
| P1 | Analyst prompt 加强（内联引用 + 禁止训练知识） | v0.1.4 |
| B2 | CrossEncoder Reranker（bge-reranker-v2-m3），`RERANKER_ENABLED=true` | v0.1.4 |
| — | Embedding GPU 加速：fastembed ONNX CPU → sentence-transformers CUDA | v0.1.5 |
| — | 跨语言检索默认开启（XLING_ENABLED=true），Recall@6 +3.4pp | v0.1.5 |
| C1 | Query Rewriter — HyDE（假设文档嵌入检索），Recall@6 +1.7pp，R@6=0.892→0.908 | v0.1.6 |
| — | 附加实验：Keyword Expansion（R@6 +0.8pp）和 HyDE+Keyword 融合（R@6 +0.8pp，融合噪声大于互补收益） | 实验存档 |

---

## 三、当前不足

| 问题 | 表现 | 方案 |
|------|------|------|
| Q1 问句-文档措辞鸿沟 | 中文问句→英文文档时嵌入空间不匹配，如"如何评估X"检不到"X包括…" | C1 ✅ HyDE +1.7pp，R@6=0.892→0.908 |
| Q2 无重排序 | 向量近似排序，高相似≠真相关 | B2 ✅ 已完成 |
| Q3 纯向量检索 | 专有名词/版本号无法精确命中 | Hybrid Search (dense+sparse) |
| Q4 评估循环验证 | bge-m3 同时用于索引和评估，差值 <0.02 | LLM Judge |
| Q5 阈值 0.45 过低 | 有效下界 ≈0.65，无实际过滤 | 待调整 |
| Q6 评估盲区 | Context Recall / Chunk Utilization 未提取 | C4 / C5 |
| Q7 碎片化 | top-6 chunks 原子独立，无关系视图 | — ⚠️ 验证无效（见§七）|

---

## 四、评估体系

### 4.1 Chunking 质量

| 指标 | 算法 | 参考值 |
|------|------|--------|
| 语义凝聚度 | chunk 内句 embedding 平均余弦 | 好>0.75 差<0.5 |
| 边界渗漏 | 相邻 chunk 首尾 embedding 余弦 | 好<0.6 差>0.8 |
| 长度分布 | 字符 P10/P50/P90 | P50≈600–800, P10<100 碎片 |
| 引用命中率 | analyst 引用 URL 回溯到 chunk 的比例 | 好>80% 差<50% |

> 当前问题：chunk_size=800≈400token，远低于 bge-m3 上限 8192；HTML 残留标签污染；overlap 40词对长复合句偏短。

### 4.2 检索质量

**代理指标（无 ground truth）**：

| 指标 | 含义 | 实现 | 局限 |
|------|------|------|------|
| chunk_score_mean | Qdrant cosine 均值 | ✅ | 与 ctx_prec 循环验证 |
| threshold_pass_rate | 分数>0.45 的比例 | ✅ | 阈值合理性待验 |
| context_precision (embed) | mean(cos(q_emb, c_emb)) | ✅ | 同一 encoder, circular |
| context_precision (LLM) | LLM 判断 chunk 是否相关 | ❌ | 逐 chunk LLM 调用 |
| chunk_utilization_rate | 引用 URL / 检索 URL | ❌ | 需解析 citation_registry |

**RAGAS LLM-judge（v0.1.4+, 无 ground truth）**：

| 指标 | 实现 | 需 ground truth |
|------|------|-----------------|
| faithfulness | 答案拆 atomic claims → LLM 验证可推导性 | 否 |
| context_precision | LLM 判断每个 chunk 是否有用 → 加权精度 | 否 |

**待实现（需 ground truth）**：Context Recall / Answer Correctness

> circular validation 诊断：`|chunk_score_mean - context_precision| < 0.02` → 指标不独立，需 LLM judge 或异构 encoder 验证。[embed-based 指标已移除]

---

## 五、基准数据（Task 04, 2026-06-24）

### RAGAS 基线 v0 — 旧 prompt, 无 Reranker

| 子问题 | faithfulness | context_precision |
|--------|-------------|-----------------|
| q1 核心架构组件与数据流 | 0.500 | 1.000 |
| q2 检索工程实践 | 0.692 | 0.383 |
| q3 上下文整合与忠实回答 | 0.242 | 0.500 |
| q4 性能评估与迭代改进 | 0.389 | 0.917 |
| **均值** | **0.456** | **0.700** |

### RAGAS 基线 v1 — prompt 加强, 无 Reranker

> 新增"必须内联引用"+"禁止训练知识"约束

| 子问题 | faithfulness | context_precision |
|--------|-------------|-----------------|
| q1 核心架构组件与数据流 | **1.000** | **1.000** |
| q2 检索优化 | **1.000** | **1.000** |
| q3 提示工程与上下文整合 | 0.615 | 0.250 |
| q4 评估指标与生产工程 | 0.895 | 0.887 |
| **均值** | **0.878 (+92%)** | **0.784 (+12%)** |

**解读**：prompt 约束对幻觉抑制极显著。q3 ctx_prec=0.250 说明方法论类问题检索精准度不足，是 Reranker/HyDE 的核心目标。

### 历史参考（embed-based, 已废弃）

| 指标 | Task 04 均值 | 有效结论 |
|------|-------------|---------|
| chunk_score_mean | 0.692 | 0.45 阈值无实际过滤（下界≈0.65） |
| context_precision(embed) | 0.679 | circular validation 差值 0.013 已确认 |
| faithfulness_sem | 0.759 | — |

---

## 六、后续改进方案

| 方案 | 类型 | 描述 | 预期增益 | 额外延迟 | 状态 |
|------|------|------|---------|---------|------|
| B2 Reranker | 引入依赖 | top-20 → bge-reranker-v2-m3 → top-6 | Recall@6 +15~25% | +200~500ms/q | ✅ 已完成 |
| B3 API Embedding | 引入依赖 | fastembed → LiteLLM aembedding (ada-002) | 稳定性、零冷启动 | — | ⭐⭐⭐ 备选 |
| **C1 Query Rewriter** | **架构扩展** | **改写优先→xling 在后。HyDE 模式（假设文档嵌入检索）** | **Recall@6 +1.7pp（黄金集 120 问, R@6 0.892→0.908）** | **+0.5s/q** | **✅ HyDE 完成** |
| C2 Hybrid Search | 架构扩展 | Qdrant RRF 融合 dense + sparse | 专有名词命中 | +100ms | ⭐⭐⭐ |
| ~~C3 Multi-Query+RRF~~ | 架构扩展 | 归入 C1 Query Rewriter | — | — | ❌ 已合并 |
| C4 LLM Context Precision | 评估 | 逐 chunk LLM 判断相关性 | 消除循环验证 | +N×LLM | ⭐⭐⭐⭐ |
| C5 Chunk Utilization Rate | 评估 | 引用 URL / 检索 URL | 检索噪声诊断 | 无 | ⭐⭐⭐⭐⭐ |
| D1 GraphRAG-V | **架构扩展** | chunk 相似度图 + Louvain 社区 | ❌ 验证无效（见§七） | < 1s | ❌ 已关闭 |

### 优先级总览

| 方案 | 质量提升 | 延迟 | 成本 | 状态 |
|------|---------|------|------|------|
| ~~A1–A4, B1~~ | — | — | — | ✅ 完成 |
| C5 Chunk Utilization | 评估覆盖 | 无 | 极低 | ⭐⭐⭐⭐⭐ |
| B2 Reranker | 高 | +200~500ms | 中 | ✅ 完成 |
| **C1 Query Rewriter** | **中（+1.7pp）** | **+0.5s** | **中** | **✅ HyDE 完成** |
| C4 LLM Context Precision | 评估独立 | +N×LLM | 中 | ⭐⭐⭐⭐ |
| ~~D1 GraphRAG-V~~ | — | — | — | ❌ 验证无效 |
| C2 Hybrid Search | 专有名词 | +100ms | 高 | ⭐⭐⭐ |
| B3 API Embedding | 稳定性 | 稳定 | 低 | ⭐⭐⭐ 备选 |

---

## 七、已验证无效方案：GraphRAG-V chunk 相似度图

> 2026-07-03 | 结论：在检索质量和 LLM 输出质量上均无正面效果，已关闭。

### 7.1 动机

当前 top-6 chunks 为独立原子，LLM 无法感知哪些属于同一话题。希望用 chunk embedding 相似度图构建主题社区，以**零额外 LLM 调用 + < 1s 延迟**实现社区感知检索。

### 7.2 实验验证

**实验目录**：`rag_experiments/`，完整源代码和结果已存档。

**方案**：
- **GraphBuilder**：从 Qdrant 读全量 vectors → 余弦相似度矩阵 → NetworkX 图 → Louvain 社区
- **Phase 1**（检索召回）：top-6 查所属社区 → 将同社区其他 chunk（含远邻，即 top-40 之外的）加入候选池 → rerank 重新排序取 top-6，测 Recall@6
- **Phase 2**（LLM 质量）：shared_system_content 中证据按社区分组标注 → 测 RAGAS faithfulness

**基线**：生产配置（bge-m3 GPU + xling 双路检索 + rerank），黄金集 120 条问题

### 7.3 结果

**Phase 1 — Recall@k（12 任务, 120 条黄金问题）**

| 指标 | Baseline | GraphRAG-V | Δ |
|------|----------|-----------|----|
| Recall@40 | 0.917 | 0.917 | 0 |
| Recall@6 | **0.892** | **0.892** | **0** |
| MRR | 0.738 | 0.738 | 0 |
| NDCG@6 | 0.778 | 0.778 | 0 |

社区扩张未改善任何检索指标。原因：bge-m3 余弦 >0.70 的两个 chunk 几乎是"同一篇文章的不同段落说同一件事"，而非"语义相关但不同的文档"。reranker 已对所有候选正确排序，社区没有提供额外信息。

**Phase 2 — Faithfulness（2 任务验证）**

| 任务 | Baseline | GraphRAG-V | Δ |
|------|----------|-----------|----|
| 04 RAG 系统最佳实践 | 0.893 | 0.783 | **-0.110** |
| 07 碳中和技术路径 | 0.711 | 0.645 | **-0.066** |

社区分组后 faithfulness 不升反降。分析：社区标注词消耗上下文窗口，且同一 community 的 chunk 原本就是高相似度的重复内容，分组展示无助于减少幻觉。

### 7.4 根因分析

```
假设：bge-m3 embedding 的余弦相似度能反映"语义相关程度"
实测：余弦 > 0.70 实际上是"近乎相同的内容"
     → 同一社区的 chunks 几乎重复 → LLM 看不到新信息
     → 不同社区的 chunks 余弦 < 0.70 但有查询相关性
     → reranker 捕捉到这些跨社区相关性，社区结构反而成了噪声

结论：Chunk 级相似度图不适合深度调研场景。
     真正的信息增益需要实体级关系（A outperforms B、A depends_on C），
     但这需要 LLM 抽取（MS GraphRAG = 600 次 LLM 调用），
     对单次任务而言成本不可接受。
```

### 7.5 关闭原因总结

| 阶段 | 假设 | 实测 | 结论 |
|------|------|------|------|
| Phase 1 社区扩张检索 | 捞回遗漏相关 chunk → Recall↑ | Recall@6 无变化 | chunk 余弦相似度 ≠ 查询相关性 |
| Phase 2 社区分组组织 | LLM 看到主题结构 → 幻觉↓ | Faithfulness **-0.07 ~ -0.11** | 分组对 LLM 质量无益反损 |
| 综合 | GraphRAG-V 在检索+生成均有价值 | 两阶段均无正面效果 | 此方向关闭，资源投入其他方案 |

---

## 八、Query Rewriting — 实验验证（v0.1.6）

### 8.1 黄金集评测结果

**实验框架**：`rag_experiments/query_rewriter.py` + `rag_experiments/experiment_query_rewrite.py`，独立于生产代码运行。基线 = 生产配置（bge-m3 GPU + xling 双路 + reranker，R@6=0.892）。

| 方案 | Recall@6 | Δ vs Baseline | MRR | NDCG@6 | 恢复 miss 数 |
|------|----------|:---:|:---:|:---:|:---:|
| Baseline (v0.1.5) | 0.892 | — | 0.738 | 0.778 | — |
| **HyDE** ✅ **选定** | **0.908** | **+1.7pp** | **0.746** | **0.786** | **2** |
| Keyword | 0.900 | +0.8pp | 0.741 | 0.780 | 1 |
| HyDE+Keyword 融合 | 0.900 | +0.8pp | 0.746 | 0.786 | 1 |

**结论**：HyDE 单独最优（+1.7pp），Keyword 和融合方案均不及 HyDE。融合时向量 Recall@40 从 0.917 暴跌至 0.817（-10pp），5 条 extra queries 引入大量噪声将 gold chunk 挤出 top-40，reranker 仅能部分修复。HyDE 的单一假设段落聚焦效果更好——语义空间缩窄而非拓宽。

### 8.2 恢复案例分析

HyDE 恢复的 2 条 miss：
1. **Task 01（LLM 推理加速）**：*"有哪些减少大模型 KV Cache 大小的常见算法？"* → HyDE 的假设段落（MQA / GQA / sliding window / sparse attention）使向量命中相关文档。
2. **Task 02（Python vs Rust 对比）**：*"Rust 在构建 WebAssembly 时如何与 JavaScript 进行数据交换？"* → HyDE 段落的 wasm-bindgen / serde 等具体技术术语锚定检索。

### 8.3 流水线顺序

所有 query 统一走「改写 → 翻译 → 检索」管道：

```
子问题（中文）
  ↓ QueryRewriter（HyDE 模式）
  ├─ 原始 query（中文）
  └─ HyDE 假设段落（英文，2-3 句技术风格）
  ↓ xling 双路（逐 query 判断——含中文才翻译）
  ├─ (中文) → Chinese vec + English vec
  └─ (英文) → English vec only（xling 直通）
  ↓ 批量 Embed + 并行 Search
  ↓ 按 chunk 合并取最高分 → Reranker → top-k
```

### 8.4 配置

```bash
QUERY_REWRITER_ENABLED=true
QUERY_REWRITER_MODE=hyde       # hyde 为选定模式
QUERY_REWRITER_MAX_GROUPS=1    # HyDE 单段落
```

### 8.5 冷启动 & 降级

- LLM 调用失败 → 静默空列表，降级为原始 query
- 改写产出的 HyDE 段落为空 → 降级为原始 query

---

## 九、已关闭方案

### GraphRAG-V（§七）

### HyDE+Keyword 融合

> 2026-07-06 | 结论：HyDE 和 Keyword 虽互补（恢复不同 miss），但融合引入噪声大于互补收益，不应合并使用。

| 评估维度 | 结论 |
|---------|------|
| 向量 Recall@40 | Baseline 0.917 → 融合 0.817（-10pp，多路检索噪声挤占） |
| Reranker 修复后 | Baseline 0.892 → 融合 0.900（+0.8pp，不及 HyDE 单独 +1.7pp） |
| 根因 | 4 组关键词 + 1 段 HyDE = 5 路 query 并行检索，不相关的关键词命中噪声 chunk 将 gold 挤出 top-40 |

### Keyword Expansion 单独

> 2026-07-03 | 结论：+0.8pp 增益偏低，且与 HyDE 恢复的 miss 不重叠，无组合价值。

## 附录：评测命令

```bash
python benchmark/test_rag.py 04 --no-report   # 单任务快速诊断
python benchmark/test_rag.py 04 --verbose      # 显示 chunk 文本摘要
python benchmark/benchmark_rag.py              # 全量 12 任务
python benchmark/benchmark_rag.py 01 04 07 --save  # 指定任务并保存 JSON
```