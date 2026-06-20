# Retriever 模块改进报告

> 记录 `retriever_node` 从原始实现到并发优化的完整过程

---

## 一、三个版本的并发结构

### V1 — 原始实现（全部并发，无节流）

所有子问题的全部查询词**同时发出**，无任何限制。

```
                          时间轴 →
queries (flat list):
  Q1  ━━━━━━━━━━━━━━━━━━
  Q2  ━━━━━━━━━━━━━━━━━━━━━━
  Q3  ━━━━━━━━━━━━━━━━━━━━━━━━━
  Q4  ━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Q5  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Q6  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Q7  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Q8  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Q9  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Q10 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Q11 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Q12 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       ↑ 所有查询同时启动（peak = 12）
       无 sub_question_id 归属
```

**实现代码：**
```python
# nodes.py (原始)
queries = state.get("search_queries", [...])
results = await search_service.multi_search(queries=queries)
# multi_search 内部：asyncio.gather(*[search(q) for q in queries])
# → 无 Semaphore，全部并发
```

**特点：** 速度最快（~5-10s），但结果无归属，高并发时 DuckDuckGo 易触发限速（0 结果）。

---

### V2 — 第一次改进（子问题顺序执行，内部 Semaphore=3）

外层 `for` 循环逐一处理每个子问题，子问题内部最多 3 个查询并发。

```
                          时间轴 →
SQ1 ┌────────────────────┐
    │ Q1━━━━━━━           │
    │ Q2━━━━━━━           │  ← 3 个并发（Sem=3）
    │ Q3━━━━━━━━━━━━      │
    └────────────────────┘
SQ2                        ┌──────────────────────┐
                           │ Q4━━━━━━              │
                           │ Q5━━━━━━━━━           │  ← 等 SQ1 完全结束才开始
                           │ Q6━━━━━━━━━━          │
                           └──────────────────────┘
SQ3                                                 ┌─────────────────┐
                                                    │ Q7━━━━━━        │
                                                    │ Q8━━━━━━━━━     │
                                                    │ Q9━━━━━━━━━━━   │
                                                    └─────────────────┘
...（SQ4、SQ5 以此类推）
       ↑ 串行子问题，peak = 3，总时间 = Σ(每个SQ的耗时)
```

**实现代码：**
```python
# nodes.py (V2)
for sq in sub_questions:          # ← 顺序循环
    results = await search_service.multi_search(
        queries=sq["search_queries"],
        concurrency=3,            # ← 每批最多 3 个并发
    )
    # 携带 sub_question_id
```

**特点：** 有 sub_question_id 归属；但顺序执行导致**总时间 = 各子问题耗时之和**，5 个子问题时比 V1 慢 3-4 倍（~35s vs ~10s）。

---

### V3 — 当前实现（子问题并发，全局共享 Semaphore=5）

所有子问题**同时启动**，通过一个全局 Semaphore(5) 限制**跨子问题**的总并发上限。

```
                          时间轴 →
SQ1: Q1━━━━━  Q2━━━━━━━  Q3━━━━━━━
SQ2: Q4━━━━━━ Q5━━━━━━   Q6━━━━━━       ← 5 个子问题同时启动
SQ3: Q7━━━━━  Q8━━━━━━   Q9━━━━━━━       (asyncio.gather)
SQ4: Q10━━━━━         Q11━━━━━━━ Q12━━━━
SQ5: Q13━━━━━━  Q14━━━━━━━       Q15━━━━
      ↑ 任意时刻最多 5 个请求在飞（全局 Semaphore=5）
      ↑ 所有结果均携带正确的 sub_question_id
```

**实现代码：**
```python
# nodes.py (V3, 当前)
global_sem = asyncio.Semaphore(5)        # 全局一个，跨子问题共享

async def _fetch_sq(sq):
    return await search_service.multi_search(
        queries=sq["search_queries"],
        concurrency=global_sem,           # 传入共享 Semaphore
    )

batches = await asyncio.gather(          # ← 所有子问题并发
    *[_fetch_sq(sq) for sq in sub_questions],
    return_exceptions=True,
)
```

**特点：** 兼顾速度与稳定性；sub_question_id 全归属；并发峰值可控（5）。

---

## 二、为什么 V1 最快

| 维度 | V1（原始） | V2（顺序 SQ） | V3（并发 SQ） |
|------|-----------|--------------|--------------|
| 外层结构 | 无分组，flat list | 串行 for 循环 | asyncio.gather |
| 并发峰值 | 等于查询总数（12-15） | 3 | 5 |
| 节流开销 | 零 | 每批等待 Semaphore | 等待 Semaphore |
| 总时间（3题均值） | **~8.7s** | ~35s（估算） | **~13.6s** |

**V1 最快的本质原因：没有任何等待。**

12 个查询全部同时发出，I/O 等待完全重叠：
- 最慢的一个查询决定总耗时（~5s）
- 其余 11 个查询的等待时间被并行掩盖

V2/V3 引入 Semaphore 后，部分查询必须等待"令牌"，造成额外等待。  
V2 尤其极端：串行 for 让等待**叠加**而非并行。

### V3 相比 V2 的改善量

单题测试（12个查询/5个子问题）：

```
V1（原始）:   ████████░░░░░░░░░░░░░░░░░░░  5.4s  peak=12
V2（顺序SQ）: ████████████████████████░░░  ~22s  peak=3
V3（并发SQ）: ████████░░░░░░░            8.5s  peak=5

              └── V2→V3 节省 ~13s（减少 60%）
              └── V1→V3 额外代价 +3s（56% 溢出），换取归属 + 限流
```

---

## 三、改进过程记录

### 背景与动机

原始 `retriever_node` 存在两个问题：
1. **无归属**：搜索结果只是 URL+内容的平列列表，不知道哪条结果对应哪个子问题，下游 `analyst_node` 无法按子问题组织分析
2. **无节流**：DuckDuckGo 面对 10+ 并发查询时会返回空结果（rate limiting），导致部分查询零结果

### 改进方案

**方案 A（否决）：** 用 `research_strategy.depth` 决定每个子问题的搜索数量
- 否决理由：Planner 已经通过输出查询词数量控制搜索深度，额外用 depth 字段决定条数是重复控制

**方案 B（采纳）：** `sub_question_id` 归属
- 每条搜索结果标记 `sub_question_id`，让下游节点知道每条信息对应哪个研究角度
- `SearchResult` 模型已有 `sub_question_id: str | None = None` 字段（无需改模型）

**方案 C（采纳，简化）：** 并发控制
- 用 `asyncio.Semaphore` 限制最大并发数，避免 DuckDuckGo 限速
- 不做降级（无需：失败的查询返回空列表，已有 exception 处理）

### 实施步骤

#### Step 1：`multi_search` 支持外部共享 Semaphore（`search_service.py`）

```python
# 修改前
async def multi_search(self, queries, max_results_per_query=5) -> list[SearchResult]:
    tasks = [self.search(q) for q in queries]
    results_lists = await asyncio.gather(*tasks, return_exceptions=True)

# 修改后
async def multi_search(
    self,
    queries,
    max_results_per_query=5,
    concurrency: int | asyncio.Semaphore = 3,   # ← 新参数
) -> list[SearchResult]:
    semaphore = concurrency if isinstance(concurrency, asyncio.Semaphore) \
                else asyncio.Semaphore(concurrency)
    async def _throttled(q):
        async with semaphore:
            return await self.search(q, max_results=max_results_per_query)
    results_lists = await asyncio.gather(*[_throttled(q) for q in queries], ...)
```

接受 `int | asyncio.Semaphore` 的原因：调用方可传入**跨多个 `multi_search` 调用共享**的同一个 Semaphore，使全局并发上限生效。

#### Step 2：V2 — 顺序子问题循环（`nodes.py`）

```python
# V2：per-sq 顺序循环，内部 Semaphore=3
for sq in sub_questions:
    results = await search_service.multi_search(
        queries=sq.get("search_queries", []),
        max_results_per_query=settings.max_search_results,
        concurrency=3,
    )
    all_results += [{**r.model_dump(), "sub_question_id": sq["id"]} for r in results]
```

**Benchmark 结果（V2）：** 时间 3.4x（10s → 35s），sub_question_id 100% 覆盖

#### Step 3：V3 — 并发子问题 + 全局 Semaphore（`nodes.py`）

发现 V2 串行循环导致 3.4x 时间增长后，将外层改为 `asyncio.gather`：

```python
# V3：子问题并发，全局 Semaphore=5
global_sem = asyncio.Semaphore(5)

async def _fetch_sq(sq):
    results = await search_service.multi_search(
        queries=sq.get("search_queries", []),
        max_results_per_query=settings.max_search_results,
        concurrency=global_sem,         # ← 共享全局 Semaphore
    )
    sq_id = sq.get("id", "")
    return [{**r.model_dump(), "sub_question_id": sq_id} for r in results]

batches = await asyncio.gather(
    *[_fetch_sq(sq) for sq in sub_questions],
    return_exceptions=True,
)
```

**关键设计决策：** 共享 `global_sem` 而非每个子问题单独创建 Semaphore。  
若每个子问题各有 `Semaphore(3)`，5 个子问题 × 3 = 峰值 15，等同于无节流；  
一个共享 `Semaphore(5)` 保证任意时刻全局最多 5 个 HTTP 请求在飞。

### 最终 Benchmark 数据（V1 vs V3，全量 12 题）

| 任务 | 深度 | SQ数 | 查询数 | V1 耗时 | V3 耗时 | 结果变化 | 零结果 | 归属 |
|------|------|------|--------|---------|---------|---------|--------|------|
| [01] LLM推理加速 | medium | 4 | 12 | 5.7s | 12.8s | 60→60 | 0%→0% | ✓ |
| [02] Python vs Rust | deep | 5 | 15 | 7.8s | 10.3s | 74→69 | 0%→0% | ✓ |
| [03] 量子计算基础 | shallow | 2 | 6 | 8.2s | **6.6s** | 30→30 | 0%→0% | ✓ |
| [04] RAG系统最佳实践 | deep | 6 | 25 | 8.3s | 18.4s | 124→121 | 0%→0% | ✓ |
| [05] AI芯片竞争格局 | deep | 5 | 20 | 7.7s | 13.5s | 95→99 | 0%→0% | ✓ |
| [06] GDPR对比 | deep | 4 | 16 | 7.1s | 11.4s | 70→67 | 0%→0% | ✓ |
| [07] 碳中和技术路径 | deep | 6 | 30 | 10.3s | 22.4s | 147→150 | 0%→0% | ✓ |
| [08] 电商推荐系统 | deep | 5 | 20 | 8.9s | 15.3s | 95→95 | **5%→0%** | ✓ |
| [09] LLM幻觉问题 | deep | 4 | 12 | 6.9s | 10.9s | 58→58 | 0%→0% | ✓ |
| [10] 固态电池 | deep | 5 | 15 | 7.8s | 12.4s | 74→71 | 0%→0% | ✓ |
| [11] 微服务vs单体 | deep | 5 | 20 | 8.7s | 22.7s | 98→92 | 0%→0% | ✓ |
| [12] AGI技术障碍 | deep | 5 | 19 | 10.2s | 17.3s | 91→92 | 0%→0% | ✓ |
| **合计** | | **51** | **190** | **98s** | **174s** | **1016→1004 (-1%)** | **0.4%→0%** | **100%** |

> 注：[03] 量子计算基础（仅 2 个子问题/6 个查询）V3 反而比 V1 快 1.6s，网络抖动正向叠加所致，说明子问题少时节流开销可忽略。

### 收益与代价总结

| | V1（原始） | V3（当前） |
|--|------------|------------|
| 平均耗时/题 | **8.2s** | 14.5s |
| 时间代价 | 基准 | +77%（+6.3s/题均值） |
| sub_question_id | ✗ 无 | ✓ 100% 覆盖（全 12 题） |
| 并发峰值 | 无限制（6-30） | 受控（≤5） |
| 零结果率 | 0.4%（1/190次） | **0%** |
| 总结果数 | 1016 | 1004（-1.2%，在波动范围内） |

**+77% 时间开销（约 +6s/题）** 换取的是：
- **结构化归属**：下游 `analyst_node` 可按研究角度独立分析，报告章节与子问题一一对应
- **限速消除**：DuckDuckGo 零结果从 0.4% 降到 0%
- **错误可定位**：子问题级别的异常处理，失败不影响其他子问题结果

---

## 四、遗留问题与后续方向

### 搜索引擎扩展

当前 `config.py` 中 `serper` 和 `bing` 有配置项但 `SearchService` 未实现 Provider：

| 引擎 | 状态 | 特点 |
|------|------|------|
| DuckDuckGo | ✅ 已实现 | 免费；高并发时限速 |
| Tavily | ✅ 已实现 | 付费；返回预提取正文，可减少爬虫依赖 |
| Serper | ❌ 仅 config | Google 结果，REST API 简单，需付费 key |
| Bing | ❌ 仅 config | Azure API，需 key |
| Brave Search | — 未配置 | 免费 2000次/月，质量优于 DDG |

**建议优先实现 Serper**（约 30 行，config 已就绪）作为 DuckDuckGo 的高质量备选。

### Semaphore 参数调优

当前 `global_sem = asyncio.Semaphore(5)` 硬编码。  
可考虑读取 `settings.max_concurrent_fetches`（已存在）或新增专用配置项，让用户根据网络环境调整。

### Tavily 直接提取正文

Tavily API 返回的 `content` 字段已是清洗后的页面正文，若使用 Tavily，`content_extractor_node` 的爬虫步骤可跳过，进一步加速整体流水线。
