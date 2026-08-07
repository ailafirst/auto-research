# RAG 改进方案

> 网页正文入库之后这一段：切片 → 向量化 → 检索 → 重排序。
> 上游的「搜什么、收哪些源」见 [检索与证据](检索与证据.md)。
> 评测工具：`benchmark/test_rag.py` / `benchmark/benchmark_rag.py` + `rag_experiments/`

---

## 一、当前流水线

```
chunk (size=800, overlap=200) → bge-m3 embedding (1024d, GPU) → Qdrant (task_id 隔离)
  → [Query Rewriting] 改写优先：子问题 → 英文假设段落 (HyDE)
  → [xling 双路] 含中文的 query 附英文译文（改写后逐 query 翻译）
  → 多 query 合并 → CrossEncoder rerank (bge-reranker-v2-m3)
  → 共享证据池 → LLM 分析
```

## 二、已实施改进

| # | 内容 | 版本 |
|---|---|---|
| A1–A4 | RAGService 单例、overlap 120→200、阈值 0.3→0.45、拼接标题 | v0.1.3 |
| B1 | bge-m3 升级（1024d，中文支持） | v0.1.3 |
| P1 | Analyst prompt 加强（内联引用 + 禁止训练知识） | v0.1.4 |
| B2 | CrossEncoder Reranker（bge-reranker-v2-m3），`RERANKER_ENABLED=true` | v0.1.4 |
| — | Embedding GPU 加速：fastembed ONNX CPU → sentence-transformers CUDA | v0.1.5 |
| — | 跨语言双路默认开启（`XLING_ENABLED=true`），Recall@6 +3.4pp | v0.1.5 |
| C1 | Query Rewriter — HyDE，Recall@6 0.892→0.908（+1.7pp） | v0.1.6 |

## 三、仍未解决

| # | 问题 | 方向 |
|---|---|---|
| Q3 | 纯向量检索，专有名词/版本号无法精确命中 | Hybrid Search（Qdrant RRF 融合 dense + sparse），+100ms |
| Q4 | 评估循环验证：bge-m3 同时用于索引和评估，差值 <0.02 | LLM Judge 逐 chunk 判相关性 |
| Q5 | 阈值 0.45 形同虚设，实测有效下界 ≈0.65 | 待调整 |
| Q6 | Chunk Utilization Rate（引用 URL / 检索 URL）未提取 | 成本极低、诊断价值高，**优先做** |
| — | chunk_size=800 ≈ 400 token，远低于 bge-m3 上限 8192；HTML 残留标签污染 | 待验证是否值得调 |

---

## 四、评估体系

**Chunking 质量**（`benchmark/test_rag.py`）：语义凝聚度（chunk 内句 embedding 平均余弦，好 >0.75）、边界渗漏（相邻 chunk 首尾余弦，好 <0.6）、长度分布（P50 应在 600–800）、引用命中率（好 >80%）。

**检索质量**分两类：

| 类别 | 指标 | 局限 |
|---|---|---|
| 代理指标（无 ground truth） | `chunk_score_mean`、`threshold_pass_rate` | 与 context_precision 循环验证 |
| RAGAS LLM-judge（v0.1.4+） | `faithfulness`（答案拆 atomic claim → LLM 验可推导性）、`context_precision`（LLM 逐 chunk 判有用性） | 需 LLM 调用 |
| 需 ground truth，未实现 | Context Recall、Answer Correctness | — |

> **循环验证诊断**：`|chunk_score_mean − context_precision(embed)| = 0.013 < 0.02` → 两个指标不独立，embed-based 那套已废弃移除，改用 LLM judge。

**基准数据**（Task 04，2026-06-24）：同一批检索结果，仅给 analyst prompt 加"必须内联引用 + 禁止训练知识"两条约束——

| | faithfulness | context_precision |
|---|---|---|
| v0 旧 prompt | 0.456 | 0.700 |
| v1 加约束 | **0.878（+92%）** | **0.784（+12%）** |

**解读**：prompt 约束对幻觉抑制极显著。剩下的 q3（方法论类问题）ctx_prec 仅 0.250，说明**检索精准度不足**，这正是 Reranker 与 HyDE 的靶子。

---

## 五、Query Rewriting — HyDE（v0.1.6）

基线 = 生产配置（bge-m3 GPU + xling 双路 + reranker，R@6=0.892），黄金集 120 问：

| 方案 | Recall@6 | Δ | MRR | 恢复 miss |
|---|---|---|---|---|
| Baseline | 0.892 | — | 0.738 | — |
| **HyDE** ✅ 选定 | **0.908** | **+1.7pp** | 0.746 | 2 |
| Keyword | 0.900 | +0.8pp | 0.741 | 1 |
| HyDE + Keyword 融合 | 0.900 | +0.8pp | 0.746 | 1 |

**为什么融合反而不行**：4 组关键词 + 1 段 HyDE = 5 路并行检索，向量 Recall@40 从 0.917 **暴跌到 0.817**（-10pp）——不相关关键词命中的噪声 chunk 把 gold 挤出 top-40，reranker 只能部分修复。**HyDE 的单一假设段落是把语义空间缩窄而非拓宽**，这才是它更优的原因。

恢复的 2 条 miss 都印证这点：*"减少 KV Cache 大小的常见算法"* 靠 HyDE 段落里的 MQA/GQA/sliding window 命中；*"Rust 构建 WebAssembly 如何与 JS 交换数据"* 靠 wasm-bindgen/serde 锚定。

配置与降级：

```bash
QUERY_REWRITER_ENABLED=true
QUERY_REWRITER_MODE=hyde
QUERY_REWRITER_MAX_GROUPS=1    # HyDE 单段落
```

LLM 调用失败或产出为空 → 静默降级为原始 query。

---

## 六、已关闭方案

### GraphRAG-V：chunk 相似度图 + Louvain 社区（2026-07-03）

动机是 top-6 chunks 互为孤立原子，希望用 embedding 相似度图构建主题社区，**零额外 LLM 调用 + <1s 延迟**实现社区感知检索。两阶段实验全部无正面效果：

| 阶段 | 假设 | 实测 |
|---|---|---|
| Phase 1 社区扩张检索 | 捞回遗漏相关 chunk → Recall↑ | Recall@6 / MRR / NDCG@6 **一位小数都没动**（0.892 / 0.738 / 0.778） |
| Phase 2 社区分组组织 | LLM 看到主题结构 → 幻觉↓ | faithfulness **-0.066 ~ -0.110** |

> **根因：假设"余弦相似度反映语义相关程度"是错的。** bge-m3 余弦 >0.70 的两个 chunk 实际上是"同一篇文章的不同段落说同一件事"——同社区 chunk 近乎重复，LLM 看不到新信息；而真正跨社区的相关性 reranker 早就捕捉到了，社区结构反倒成了噪声，标注词还白占上下文窗口。
>
> 真正的信息增益需要**实体级**关系（A outperforms B、A depends_on C），但那需要 LLM 抽取（MS GraphRAG ≈ 600 次调用/任务），单次研究任务承担不起。**chunk 级相似度图不适合深度调研场景。**

### HyDE + Keyword 融合、Keyword 单独（2026-07-06）

见 §五。Keyword 单独 +0.8pp 增益偏低，且与 HyDE 恢复的 miss **不重叠**——看似互补，融合却因噪声净亏。

---

## 附录：评测命令

```bash
python benchmark/test_rag.py 04 --no-report        # 单任务快速诊断
python benchmark/test_rag.py 04 --verbose          # 显示 chunk 文本摘要
python benchmark/benchmark_rag.py                  # 全量 12 任务
python benchmark/benchmark_rag.py 01 04 07 --save  # 指定任务并保存 JSON
```
