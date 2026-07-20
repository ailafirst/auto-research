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

## 快速开始（本地 Web UI）

目标是在本机启动 Web UI、提交研究问题并查看报告。开发模式只需要 Python 和 Node.js；Windows 完整本地服务模式额外使用 Docker、Redis、模型服务和 Nginx。

### 0. 前置条件

| 必需 | 说明 |
|------|------|
| **Git** | 拉取代码 |
| **Miniconda / Anaconda**（或任意 Python 3.11）| 后端运行环境 |
| **Node.js ≥ 20** | 前端（Vite 8）需要，装完自带 `npm` |
| **一个 LLM API Key** | OpenAI / MiMo / 任意 OpenAI 兼容后端。每个节点都要调 LLM，没有 key 会在执行时报错 |

可选：Docker Desktop（完整本地服务模式的 Redis）、NVIDIA GPU（本地嵌入/重排加速；无 GPU 可改用轻量 CPU 路径，见第 3 步）。

### 1. 下载代码

```bash
git clone https://github.com/ailafirst/auto-research.git
cd auto-research
```

### 2. 后端：建环境 + 装依赖

```bash
conda create -n deepresearch python=3.11 -y
conda activate deepresearch
$env:PIP_CACHE_DIR = (Join-Path $PWD "deployment\runtime\package-cache\pip")
pip install -r requirements.txt
# 可选：想用 `deepresearch "问题"` 命令行入口，再执行一次 pip install -e .
```

> 没有 conda 也行：自备 Python 3.11，`python -m venv .venv` 激活后再 `pip install -r requirements.txt`。

### 3. 配置 .env（最少改一行）

```bash
cp .env.example .env      # PowerShell 下 cp 同样可用；或 copy .env.example .env
```

用编辑器打开 `.env`，**至少**填下面几项，其余保持默认：

```ini
LLM_PROVIDER=openai            # OpenAI 兼容后端都填 openai
LLM_MODEL=gpt-4o-mini          # 换成你的模型，如 mimo-v2.5
LLM_API_KEY=sk-...             # ← 必填：你的 key
LLM_BASE_URL=                  # 仅自建/第三方兼容端点需要，如 https://api.xxx.com/v1

USE_TAVILY=false               # 先用免费 DuckDuckGo（无需搜索 key）；有 Tavily key 再改回 true
```

> **想零外部依赖直接跑通**：保持 `REDIS_URL=` 为空即可——队列 / 缓存 / 限流会自动禁用并回退进程内执行，无需 Redis、无需另起 worker。
> **没有 GPU**：把 `EMBEDDING_PROVIDER=st` 改为 `fastembed`，并设 `RERANKER_ENABLED=false`、`XLING_ENABLED=false`，走纯 CPU 轻量路径。

### 4. 前端：装依赖

```bash
cd web
npm ci --cache ..\deployment\runtime\package-cache\npm
cd ..
```

### 5. 开发模式启动 Web UI

开发模式需要两个终端，适合代码调试。

**终端 ①｜后端 API**
```bash
conda activate deepresearch
uvicorn app.main:app --port 8000
```
> 首次启动会预热本地嵌入/重排模型，可能等十几秒到一两分钟；出现 `Application startup complete` 即就绪（Swagger 在 http://localhost:8000/docs ）。

**终端 ②｜前端 Web UI**
```bash
cd web
npm run dev
```

### 6. 打开浏览器

访问 **http://localhost:5173** —— 输入研究问题并提交，即可实时看到
「规划 → 搜索 → 抓取 → 评估 → 证据 → 分析 → 核查 → 报告」的全过程，以及最终带引用来源的报告。

> 前端通过 Vite 把 `/api`、`/health` 代理到 :8000，所以①②两个服务要同时开着。

### 7. Windows 完整本地服务启动

需要 Redis、多 worker、独立模型服务和 Nginx 本地入口时，先准备本机运行目录和私有配置：

```powershell
.\deployment\start-local.ps1 -PrepareOnly
```

在根目录 `.env` 或 `deployment/runtime/secrets.env` 填写私有配置；将 Nginx Windows 包放入 `deployment/runtime/nginx/`，确保其中有 `nginx.exe` 与 `conf/mime.types`。然后启动：

```powershell
.\deployment\start-local.ps1 -PythonExe "D:\conda\envs\deepresearch\python.exe"
```

浏览器访问 `http://127.0.0.1/`。脚本会把模型、包和临时缓存写入被 Git 忽略的 `deployment/runtime/`，避免持续占用 C 盘。

关闭本地服务：

```powershell
.\deployment\stop-local.ps1
```

短暂停止 Web UI、但希望下次快速恢复时可保留 Redis：

```powershell
.\deployment\stop-local.ps1 -KeepRedis
```

保留 Redis 会继续占用内存和 `6379` 端口，但会保留队列、进度、缓存和限流的热状态。SQLite 才是任务权威存储；当前 Redis 未配置持久卷，不应将其当作长期数据保存。

---

### 不想用前端？命令行也能跑

```bash
python -m app.main "你的研究问题"      # 报告写到 output/<task_id>.md（开箱即用）
deepresearch "你的研究问题"            # 等价入口，需先 pip install -e .
```

Streamlit 旧界面（legacy）：`streamlit run ui/streamlit_app.py` → http://localhost:8501 。

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
├── deployment/                 # Windows 本地服务启动与停止脚本
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
