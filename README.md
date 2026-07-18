# 🔬 Deep Research Agent

基于 **LangGraph + RAG + FastAPI** 的自动化深度调研系统。

输入一个研究问题，系统自动完成：问题分解 → 联网搜索 → 网页抓取 → RAG 证据构建 → 多轮分析论证 → 事实核查 → 生成带引用来源的结构化 Markdown 报告。核查未通过时自动发起补充研究轮次，直至证据充分或达到轮次上限。

## 特性

- **多轮补充研究** —— 事实核查发现矛盾/夸大/证据不足时，自动生成 follow-up 查询再跑一轮（控制流在 `research_runner.py` 的外层循环，非图内分支）。
- **可追溯引用** —— Citation Registry 为每个采纳信源分配稳定 `C01/C02…` id，贯穿 analyst → fact_checker → report_writer。
- **过程透明** —— `/progress` 端点暴露逐节点中间产物（策略、子问题、信源、证据、核查、轮次历史），前端 React 界面实时可视化调查进度。
- **横向扩展** —— arq + Redis 任务队列，多 worker 进程池分摊任务；全局 LLM 速率由 Redis 令牌桶跨进程限住。
- **显存/内存优化** —— 模型服务化（独立进程持有 embed/rerank/translate 三模型），worker 不 import torch；FP16 量化。详见 [`docs/模型服务化与显存内存优化.md`](docs/模型服务化与显存内存优化.md)。
- **降级安全** —— Redis / 任务队列 / 模型服务任一不可用时，均自动回退到进程内执行。单机开发无需任何外部依赖。

## 技术栈

| 模块 | 技术 |
|------|------|
| Agent 编排 | LangGraph（线性链 + 外层多轮控制流）|
| LLM 调用 | LiteLLM（OpenAI 兼容 / Claude / MiMo 等任意 API）|
| API 服务 | FastAPI + Uvicorn |
| 任务队列 | arq + Redis（多 worker 进程池，可降级为进程内）|
| 持久化 | SQLite（冷层 SoR）+ Redis（热层进度/缓存/限流）|
| 向量数据库 | Qdrant（内存态）|
| Embedding | 可插拔：`st`（sentence-transformers，默认 BAAI/bge-m3 1024 维）/ `fastembed`（CPU ONNX）/ `api` |
| Reranker | CrossEncoder（bge-reranker-v2-m3）|
| 跨语言检索 | opus-mt 本地 MT（中问→英源双路召回）|
| 模型服务 | 独立 FastAPI 进程（embed / rerank / translate）|
| 网页抓取 | httpx + BeautifulSoup |
| 搜索引擎 | Tavily（key 轮换池）/ DuckDuckGo |
| Web 前端 | React 19 + Vite + TypeScript（`web/`）；Streamlit（`ui/`，legacy）|
| 配置管理 | Pydantic Settings + `.env` |

## 快速开始

### 1. 环境准备

```bash
conda activate deepresearch                 # Python 3.11
pip install -r requirements.txt
cp .env.example .env                         # 编辑 .env，至少填 LLM_API_KEY
```

> 每个节点都调用 LLM，**没有规则兜底**——`LLM_API_KEY` 缺失会在执行时报错。

### 2. 单机最简运行（无需 Redis / worker / 模型服务）

`.env` 中令 `QUEUE_ENABLED=false`（或不启 Redis），研究流程在 API 进程内直接执行：

```bash
# API 服务，Swagger 见 /docs
uvicorn app.main:app --reload --port 8000

# 或 CLI 模式，输出写入 output/<task_id>.md
python -m app.main "你的研究问题"
deepresearch "你的研究问题"                   # pyproject.toml 入口点
```

### 3. 完整部署（多 worker + 模型服务 + React 前端）

```bash
# ① Redis（队列 / 限流 / 缓存 / 热层进度），.env 设 REDIS_URL=redis://localhost:6379/0
docker compose up -d redis

# ② 模型服务（独立进程持有全部模型），.env 设 MODEL_SERVICE_URL=http://localhost:8100
uvicorn app.model_server:app --host 0.0.0.0 --port 8100

# ③ API 服务
uvicorn app.main:app --port 8000

# ④ worker 进程池（QUEUE_ENABLED=true），可在多个终端各起一个横向扩展
arq app.worker.WorkerSettings

# ⑤ React 前端（dev 由 Vite 代理 /api、/health 到 :8000）
cd web && npm install && npm run dev          # http://localhost:5173
```

Streamlit 旧界面仍可用：`streamlit run ui/streamlit_app.py`（http://localhost:8501）。

## 项目结构

```
deepresearch/
├── app/
│   ├── main.py                 # FastAPI 应用入口（lifespan 预热模型）
│   ├── model_server.py         # 模型服务（embed/rerank/translate 独立进程）
│   ├── worker.py               # arq worker（研究任务执行进程）
│   ├── api/
│   │   ├── routes_research.py  # 研究任务 API 路由（含 /progress）
│   │   └── schemas.py          # 请求/响应模型
│   ├── core/                   # config / logging / exceptions
│   ├── graph/
│   │   ├── state.py            # ResearchState 状态定义
│   │   ├── nodes.py            # 8 个工作流节点实现
│   │   └── builder.py          # 线性图构建
│   ├── services/
│   │   ├── llm_service.py      # LLM 调用（并发闸 + 退避重试）
│   │   ├── search_service.py   # Tavily / DuckDuckGo 搜索
│   │   ├── crawler_service.py  # 网页抓取
│   │   ├── vector_store.py     # Qdrant 向量库
│   │   ├── rag_service.py      # 分块 / 嵌入 / 检索 / 重排 / 跨语言
│   │   ├── translation_service.py  # opus-mt 本地 MT
│   │   ├── research_runner.py  # 多轮控制流（API 与 worker 共用）
│   │   ├── queue.py            # arq 队列封装（降级安全）
│   │   ├── rate_limiter.py     # Redis 令牌桶（跨进程限流）
│   │   ├── cache_service.py    # Redis 缓存
│   │   ├── db.py               # SQLite 引擎
│   │   └── task_service.py     # 任务持久化（SQLite + Redis 热层）
│   └── prompts/                # planner / analyst / fact_checker / report_writer（可编辑 Markdown）
├── web/                        # React + Vite + TS 前端（详见 web/README.md）
├── ui/streamlit_app.py         # Streamlit 界面（legacy）
├── tests/                      # pytest（asyncio_mode=auto）
├── benchmark/                  # 端到端与逐节点基准
├── rag_experiments/            # RAG 检索质量实验
├── docs/                       # PRD / 架构 / 各改进方案 / 版本说明
├── docker-compose.yml          # Qdrant + Redis
├── .env.example
├── pyproject.toml
└── requirements.txt
```

## API 文档

启动后访问 `http://localhost:8000/docs` 查看 Swagger。

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/research` | POST | 创建研究任务 |
| `/api/research/{id}/status` | GET | 查询任务状态（轻量）|
| `/api/research/{id}/progress` | GET | 过程透明快照（策略/子问题/信源/证据/核查/轮次）|
| `/api/research/{id}` | GET | 查询任务详情 |
| `/api/research/{id}/report` | GET | 获取研究报告 |
| `/api/research` | GET | 列出所有任务 |
| `/health` | GET | 健康检查 |

## LangGraph 工作流

图本身是**纯线性链**（无条件边）；多轮补充与引用修正等控制流在图外的 `research_runner.py`：

```
Planner → Retriever → Content Extractor → Source Evaluator
    → Evidence Builder → Analyst → Fact Checker → Report Writer → 报告

外层循环：Fact Checker 未通过（矛盾/夸大/证据不足）
    → 生成 follow-up 查询 → 再跑一轮（≤ max_rounds）
```

## 测试

```bash
pytest -v                                    # asyncio_mode=auto，无需 @pytest.mark.asyncio
pytest tests/test_graph.py -v
ruff check . && ruff format .
mypy app/                                     # strict=false，可选
```

## 许可

MIT
