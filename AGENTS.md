# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 项目概述
基于 LangGraph、RAG 与 FastAPI 的自动化深度调研系统。用户输入研究问题，系统自动完成搜索、抓取、分析、验证，生成 Markdown 结构化报告。

## 环境要求
所有命令必须使用 Conda 环境 `D:\conda\envs\deepresearch`（Python 3.11）：

```bash
D:\conda\envs\deepresearch\python.exe -m uvicorn app.main:app --reload --port 8000
D:\conda\envs\deepresearch\python.exe -m pytest -v
```

## 常用命令

```bash
# 开发服务器
uvicorn app.main:app --reload --port 8000

# CLI 研究（直接运行）
python -m app.main "研究问题"
# 或使用 pyproject.toml 中注册的脚本入口
deepresearch "研究问题"

# Streamlit Web UI
streamlit run ui/streamlit_app.py

# 测试
pytest -v                          # 全部测试
pytest tests/test_graph.py -v      # 仅 Graph 测试
pytest tests/test_api.py -v        # 仅 API 测试
pytest tests/test_graph.py::test_rule_based_plan -v  # 单个测试

# 代码检查
ruff check .
ruff format .
mypy app/                              # 可选，strict=false

# 基准测试（需先启动 API 服务）
python benchmark/run_benchmark.py --list          # 查看全部 12 个任务
python benchmark/run_benchmark.py                 # 运行全部任务
python benchmark/run_benchmark.py 01 03           # 运行指定编号任务
python benchmark/run_benchmark.py --api http://localhost:8000

# Qdrant（可选，默认走内存模式）
docker compose up -d qdrant
```

`asyncio_mode = "auto"` 已在 `pyproject.toml` 中配置，无需在测试函数上手动加 `@pytest.mark.asyncio`。

## 架构要点

### LangGraph 工作流
图在 `app/graph/builder.py` 中构建，节点实现在 `app/graph/nodes.py`，状态类型为 `ResearchState`（`app/graph/state.py`）。

**节点流水线**（线性，无条件边）：
```
planner → retriever → content_extractor → source_evaluator
        → evidence_builder → analyst → fact_checker → report_writer → END
```

`planner_node` 同时填充 `research_plan`（子问题列表、搜索词）和 `research_strategy`（LLM 自动分析出的研究策略，如深度/广度权衡）两个字段。

> 注意：`nodes.py` 中的 `should_continue_research()` 函数**未被 builder.py 使用**（死代码）。多轮补充研究的循环逻辑在 `app/api/routes_research.py` 的 `_run_research()` 中通过外部 while 循环控制，并非 LangGraph 条件边。

> **CLI 已知 bug**：`app/main.py:_run_cli_research()` 中 `research_graph.ainvoke(initial_state)` 被调用了两次（行 199 和 202），导致 CLI 模式下每次研究实际执行两轮完整图遍历，消耗双倍 API 调用。

### LLM 调用
所有节点（`planner`、`analyst`、`fact_checker`、`report_writer`）均**强制调用 LLM**，无规则降级模式。`LLM_API_KEY` 未配置时系统将在 LLM 调用时抛出异常而非静默降级。

`analyst_node` 使用 `asyncio.Semaphore(5)` 限制最大并发 LLM 调用数，以避免 API 限速。

LLM 调用统一走 `LLMService`（`app/services/llm_service.py`），底层使用 LiteLLM，内置 3 次指数退避重试。自定义 `LLM_BASE_URL` 时模型名自动加 `openai/` 前缀（如 `openai/gpt-4o-mini`）。

### 向量存储（Qdrant）
`VectorStoreService`（`app/services/vector_store.py`）**硬编码为内存模式**（`AsyncQdrantClient(location=":memory:")`），即使配置了 `QDRANT_URL` 也不会自动切换到远程。运行时向量在每次进程/请求后重置。切换为持久化模式需修改构造函数中的 `AsyncQdrantClient` 初始化。

### 信源评估
`source_evaluator_node` 评分完全基于规则（相关性 = 内容长度 / 3000，可信度固定 0.7，时效性固定 0.8），不调用 LLM。`accepted` 阈值：`final_score > 0.3 且 content_length > 50`。

### 任务持久化
`TaskService` 使用内存字典 + JSON 文件双写（`output/tasks/<task_id>.json`）。`list_tasks()` **仅返回当前内存中的任务**，不扫描磁盘文件——进程重启后老任务只有在 `get_task(task_id)` 被直接调用时才会从磁盘恢复。

### RAG 流程
`RAGService`（`app/services/rag_service.py`）：
1. `TextChunker`：按段落切片（`chunk_size=800`, `chunk_overlap=120`），`chunk_overlap` 实际转换为词数（`overlap // 5`）。
2. Embedding：`fastembed` 本地加载 `BAAI/bge-small-en-v1.5`（384 维），首次运行自动下载。`embed()` 是同步生成器，通过 `asyncio.to_thread` 在线程池运行。
3. 检索时按 `task_id` 过滤，相关度阈值 > 0.3 才纳入分析上下文；若无 RAG 命中则回退到原始文档全文。

### 网页抓取
`CrawlerService`（`app/services/crawler_service.py`）：
- 黑名单：`facebook.com`, `twitter.com/x.com`, `instagram.com`, `youtube.com`, `tiktok.com`
- 限制：单页最大 5MB，并发数由 `MAX_CONCURRENT_FETCHES`（默认 5）控制
- 内容提取优先 `<article>`/`<main>` 标签，再 class/id 模式匹配

### 搜索服务
`SearchService`（`app/services/search_service.py`）通过 `SEARCH_PROVIDER` 配置选择引擎（`duckduckgo` / `tavily` / `serper` / `bing`），`multi_search` 并发执行多关键词搜索并按 URL 去重。DuckDuckGo 走 `asyncio.to_thread` 包装同步接口；其余引擎需对应 API key。

### 提示词
提示词以 Markdown 文件存放在 `app/prompts/`（`planner.md`、`analyst.md`、`fact_checker.md`、`report_writer.md`）。节点在文件不存在时自动回退到 `nodes.py` 内置的默认提示词。

## 配置
所有配置通过 `.env` 文件加载（参考 `.env.example`），由 pydantic-settings 管理（`app/core/config.py`）。
- `LLM_API_KEY` 必须配置，否则所有 LLM 节点将在执行时抛出异常
- `SEARCH_PROVIDER=duckduckgo` 无需 API key（适合本地开发）
- Qdrant 默认内存模式，无需 Docker

## API 端点
- `POST /api/research` — 创建任务（异步后台执行）
- `GET /api/research/{task_id}/status` — 查询进度
- `GET /api/research/{task_id}` — 获取任务详情
- `GET /api/research/{task_id}/report` — 获取报告
- `GET /api/research` — 列出内存中的任务
- `GET /health` — 健康检查
- `GET /docs` — Swagger UI

### `ResearchTaskCreate` 参数
| 参数 | 默认值 | 约束 |
|------|--------|------|
| `query` | — | 2–500 字符 |
| `max_rounds` | `2` | 1–5 |
| `language` | `zh-CN` | `zh-CN` \| `en` |
| `report_type` | `deep` | `summary` \| `deep` \| `comparison` |
| `search_depth` | `advanced` | `basic` \| `advanced` |
| `top_k` | `6` | 1–20（RAG 检索片段数）|
| `enable_fact_check` | `true` | — |

`report_type` 和 `search_depth` 仅在设为非默认值时才作为 `user_hints` 传入 planner（见 `routes_research.py:_build_state`）。

### 任务状态生命周期
`pending` → `planning` → `retrieving` → `indexing` → `analyzing` → `fact_checking` → `writing` → `completed` | `failed`

实际写入 `TaskService` 的 status 值为节点名称字符串（如 `planner`、`retriever`），上表为语义化命名。
