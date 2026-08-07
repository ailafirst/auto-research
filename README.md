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
| 持久化 | MySQL 8.4 或 SQLite（冷层 SoR，按 `DATABASE_URL` 切换）+ Redis（热层进度/缓存/限流）|
| 向量数据库 | Qdrant（部署态为独立服务，向量持久化 + 7 天 TTL；开发态回退进程内内存实例）|
| Embedding | 可插拔：`st`（sentence-transformers，默认 BAAI/bge-m3 1024 维）/ `fastembed`（CPU ONNX）/ `api` |
| Reranker | CrossEncoder（bge-reranker-v2-m3）|
| 跨语言检索 | opus-mt 本地 MT（中问→英源双路召回）|
| 模型服务 | 独立 FastAPI 进程（embed / rerank / translate）|
| 网页抓取 | httpx + BeautifulSoup |
| 搜索引擎 | Tavily（key 轮换池）/ DuckDuckGo |
| Web 前端 | React 19 + Vite + TypeScript（`web/`）；Streamlit（`ui/`，legacy）|
| 配置管理 | Pydantic Settings + `.env` |
| 部署 | Docker Compose（nginx / api / worker×N / mysql / redis / qdrant）+ 宿主机模型服务 |

## 快速开始

两条路，按目的选：

| | **A. Docker 部署** | **B. 开发模式** |
|---|---|---|
| 适合 | 部署、日常使用、对外提供服务 | 改代码，要热重载 |
| 组件 | nginx / api / worker×N / mysql / redis / qdrant 全在容器里 | 单进程 API + Vite dev server |
| 需要 | Docker Desktop + Python 3.11 + **NVIDIA GPU** | Python 3.11 + Node.js ≥ 20 |
| 启动 | 一条命令 | 两个终端 |

两条路都需要**一个 LLM API Key**（OpenAI / MiMo / 任意 OpenAI 兼容后端）：流程里每个节点都要调 LLM，缺它任务必然失败。

---

## A. Docker 部署

只有模型服务留在宿主机 —— 它要吃 GPU 和约 7.1GB 权重，进容器没有收益：

```
宿主机   model-server :8100  ←──┐  容器经 host.docker.internal 访问
                               │
容器     nginx :80 ──→ api ────┤
                   ↘  worker × N
                        ↘ mysql（任务 SoR） / redis（队列·缓存·限流） / qdrant（向量）
```

> 下面是**照着做就能跑起来**的步骤。想知道每个决策为什么这么定、以及部署过程中踩过哪些坑（compose 的 `$` 与 `$$`、`mysqladmin ping` 的假阳性、qdrant 版本不兼容导致证据被静默跳过、PowerShell 5.1 的 BOM 陷阱……），见 [`docs/容器化部署.md`](docs/容器化部署.md)。

### A1. 前置条件

- **Docker Desktop**（已启动）
- **Python 3.11 环境**，只用来跑模型服务
- **NVIDIA GPU**，显存 ≥ 6GB。没有 GPU 请走[开发模式](#b-开发模式)，那条路可以用纯 CPU 的 fastembed

前端不需要 Node.js —— 镜像构建时在容器内完成 `npm ci && npm run build`。

> **系统**：一键脚本 `start-all.ps1` 是 PowerShell 写的，A2–A8 按 **Windows** 描述。容器栈本身与系统无关（compose 里已配 `host.docker.internal:host-gateway`，Linux 上照样连得到宿主机模型服务），Linux / macOS 走 [A9 的手动路径](#a9-不用脚本启动linux--macos)。

### A2. 拉代码，填 `.env`

```powershell
git clone https://github.com/ailafirst/auto-research.git
cd auto-research
copy .env.example .env
```

打开 `.env`，**至少**填这几项，其余保持默认：

```ini
LLM_PROVIDER=openai            # OpenAI 兼容后端都填 openai
LLM_MODEL=gpt-4o-mini          # 换成你的模型，如 mimo-v2.5
LLM_API_KEY=sk-...             # ← 必填
LLM_BASE_URL=                  # 仅自建/第三方兼容端点需要

USE_TAVILY=false               # 先用免费 DuckDuckGo；有 Tavily key 再改回 true
```

`REDIS_URL` / `DATABASE_URL` / `QDRANT_*` 不用管：容器里的取值由 `deployment/docker/compose.yaml` 覆盖，`.env` 里那几行只影响宿主机上直接跑 Python 的场景。

### A3. 装模型服务依赖

```powershell
conda create -n deepresearch python=3.11 -y
conda activate deepresearch
pip install -r requirements-model.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

`requirements-model.txt` = `requirements.txt` + torch 系。容器镜像只装前者，所以 api / worker 进程内不会 import torch（每个 worker 常驻内存因此从 6.9GB 降到 0.75GB）。

### A4. 启动

在**上一步激活过环境的那个终端**里执行（脚本按 `-PythonExe` → `.venv\Scripts\python.exe` → `$env:CONDA_PREFIX` 的顺序找解释器；都没有就会明确报错，这时用 `-PythonExe` 指定即可）：

```powershell
.\deployment\start-all.ps1
```

一条命令做完这些事：

1. 校验根 `.env`（缺 `LLM_API_KEY` 立即报错，不会等到跑任务时才失败）
2. 首次运行生成 `deployment/docker/.env`，为 MySQL / Redis / Qdrant 各写入一串随机口令
3. 拉起宿主机模型服务
4. 构建镜像并启动容器栈（趁模型装载的时间并行做）
5. 等两边就绪后逐项校验依赖，任何一项不通就报错退出

正常结束长这样：

```
依赖校验：
  model_service  正常  http://host.docker.internal:8100 device=cuda dtype=fp16
  qdrant         正常  mode=remote client=1.19.0 server=1.19.0
  redis          正常  redis://:***@redis:6379/0
  database       正常  mysql+aiomysql://deepresearch:***@mysql:3306/deepresearch
  queue          正常  arq 已连接
Deep Research Agent 已启动。本机入口: http://127.0.0.1/
```

打开 **http://127.0.0.1/** 即可提交研究问题，页面右上角常驻依赖状态条。

> **首次要等**：镜像构建约 3~5 分钟；模型服务冷启动要顺序装载 embed(2.2GB) + rerank(2.2GB) + translate，全部预热完才开始监听 `:8100`，实测约 100 秒。脚本最多等 180 秒，期间控制台停着不动是正常的。超时中断请看[常见问题第 1 条](#1-模型服务起不来model-server-did-not-become-ready)。

常用参数：

```powershell
.\deployment\start-all.ps1 -WorkerCount 4      # 起 4 个 worker
.\deployment\start-all.ps1 -SkipBuild          # 只改了配置没改代码，跳过镜像构建
.\deployment\start-all.ps1 -PythonExe "D:\conda\envs\deepresearch\python.exe"
```

### A5. 停止

```powershell
.\deployment\stop-all.ps1                      # 停服务，容器和数据都留着
.\deployment\stop-all.ps1 -KeepModelServer     # 只重启容器栈（模型服务冷启动要 100 秒，不值得反复重来）
.\deployment\stop-all.ps1 -RemoveContainers    # 连容器一起删，数据卷仍保留
```

数据卷不在脚本处理范围内。确实要清空 MySQL 里的任务和 Qdrant 里的向量时，显式执行：

```powershell
docker compose -f deployment/docker/compose.yaml down -v
```

### A6. 日常运维

```powershell
# 看日志（-f 跟随）
docker compose -f deployment/docker/compose.yaml logs -f worker

# 只扩 worker，不动其他服务
docker compose -f deployment/docker/compose.yaml up -d --scale worker=4

# 依赖体检：/health 恒返回 200，健康度在 body 的 status 字段
curl http://127.0.0.1/health
```

**把模型服务挡在防火墙后面。** 容器经 `host.docker.internal` 连过来走的是宿主机网卡地址而非回环，所以 `:8100` 必须绑 `0.0.0.0`（脚本默认如此）。该端口自身没有鉴权，请加一条只放行本机的入站规则：

```powershell
# 管理员 PowerShell
New-NetFirewallRule -DisplayName "deepresearch model-server (local only)" `
    -Direction Inbound -LocalPort 8100 -Protocol TCP -Action Allow -RemoteAddress LocalSubnet
```

### A7. 凭据轮换

四个基础设施口令都在 `deployment/docker/.env`（不入版本库，`compose.yaml` 里写的是 `${VAR:?...}`，缺任何一个都会直接启动失败，不会悄悄退回弱口令）。

Redis 和 Qdrant 改了 `.env` 再 `docker compose ... up -d` 即可生效。

**MySQL 不行** —— `MYSQL_PASSWORD` 只在数据卷首次初始化时被读取，卷已存在时改它不会改变数据库里的实际口令（表现为改完就连不上）。必须先在库里改，再同步文件：

```powershell
# 1) 用当前的 MYSQL_ROOT_PASSWORD 登进去
docker exec -it deepresearch-docker-mysql-1 mysql -uroot -p

#    mysql> ALTER USER 'deepresearch'@'%' IDENTIFIED BY '新口令';
#    mysql> exit

# 2) 把 deployment/docker/.env 里的 MYSQL_PASSWORD 改成同一个值

# 3) 重建，让 api / worker 拿到新 DSN
docker compose -f deployment/docker/compose.yaml up -d
```

口令不会出现在日志或 `/health` 里：所有 DSN 在输出前都过 `app/core/config.py` 的 `mask_dsn()`。新增任何要打印 DSN 的地方都得照做 —— `/health` 是经 Nginx 公网可达的。

### A8. `/health` 的明细令牌

`/health` 和前端一样经 Nginx 暴露在公网。它的 `dependencies[].detail` 里是内部主机名、端口、数据库用户名、各组件版本号，失败时还有异常原文 —— 对排查有用，对访客则是白送的踩点信息。所以默认**不返回** detail，任何人都只看得到「哪一项坏了」：

```bash
curl -s http://127.0.0.1/health
# {"status":"degraded","failed":["qdrant"],"dependencies":[{"name":"qdrant","ok":false,"detail":"",...}]}
```

要看原因，带上 `deployment/docker/.env` 里的 `HEALTH_DETAIL_TOKEN`（`start-all.ps1` 首次运行时生成；老部署再跑一次会自动补写这一行）：

```powershell
$t = (Select-String deployment\docker\.env -Pattern '^HEALTH_DETAIL_TOKEN=(.+)$').Matches.Groups[1].Value
curl.exe -s -H "X-Health-Token: $t" http://127.0.0.1/health
```

`start-all.ps1` 的依赖校验自己会带这个头，所以启动时那张表照常显示原因。等价的排查入口是 `docker compose -f deployment/docker/compose.yaml logs api` —— 启动时的依赖校验本来就把同样的内容写进了日志，那里需要宿主机权限才看得到。

> 没有按来源 IP 区分内外网，是因为本项目的公网入口是 frp 隧道：frpc 在宿主机上回连 `127.0.0.1:80`，公网流量到 Nginx 时的 `$remote_addr` 和本机访问一模一样，IP 层面根本分不开。

### A9. 不用脚本启动（Linux / macOS）

`start-all.ps1` 只是把下面五步串起来并加了校验。手动做一遍是等价的：

```bash
# 1) 应用配置：填 LLM_API_KEY（同 A2）
cp .env.example .env

# 2) 部署配置：四个基础设施口令必须自己填随机值。
#    compose 里写的是 ${VAR:?...}，缺任何一个都会直接启动失败，不会退回弱口令。
cp deployment/docker/.env.example deployment/docker/.env
python - <<'PY'
import pathlib, re, secrets
p = pathlib.Path("deployment/docker/.env")
keys = ["MYSQL_PASSWORD", "MYSQL_ROOT_PASSWORD", "REDIS_PASSWORD",
        "QDRANT_API_KEY", "HEALTH_DETAIL_TOKEN"]
t = p.read_text(encoding="utf-8")
for k in keys:
    t = re.sub(rf"^{k}=.*$", f"{k}={secrets.token_urlsafe(32)}", t, flags=re.M)
p.write_text(t, encoding="utf-8")
PY

# 3) 宿主机模型服务（要 GPU；必须监听 0.0.0.0，容器走的是宿主网卡而非回环）
pip install -r requirements-model.txt
uvicorn app.model_server:app --host 0.0.0.0 --port 8100 &

# 4) 容器栈
docker compose -f deployment/docker/compose.yaml up -d --build

# 5) 等模型服务预热完（约 100 秒）再体检
curl -s http://127.0.0.1:8100/health
curl -s http://127.0.0.1/health
```

脚本额外做的事只有三件：检查根 `.env` 缺不缺 `LLM_API_KEY`、并行等待两边就绪、最后逐项校验依赖并在任一项不通时报错退出。**最后这条是有意的 —— 降级是运行期的容错能力，不是一次成功部署的可接受终态。** 手动启动时请自己看第 5 步的输出，`status` 不是 `ok` 就别急着用。

对外服务前记得回头看 [A6 的防火墙规则](#a6-日常运维)：`:8100` 自身没有鉴权。

---

## B. 开发模式

前后端各起一个进程，改代码即时生效。默认零外部依赖：`REDIS_URL` 留空时队列 / 缓存 / 限流全部自动降级为进程内，SQLite 存任务，Qdrant 用进程内内存实例。

```powershell
conda create -n deepresearch python=3.11 -y
conda activate deepresearch

# 有 GPU：装 torch 系，走 .env.example 的默认配置（st 嵌入 + 重排 + 跨语言检索）
pip install -r requirements-model.txt --extra-index-url https://download.pytorch.org/whl/cu118
# 无 GPU：只装核心依赖，下面记得改三项配置
# pip install -r requirements.txt

pip install -r requirements-dev.txt        # pytest / ruff / mypy / streamlit
copy .env.example .env                      # 填 LLM_API_KEY，同 A2

cd web && npm ci && cd ..
```

开发模式没有独立的模型服务进程（`MODEL_SERVICE_URL` 留空），模型直接在 uvicorn 进程里加载，所以 torch 系必须装在**本进程**的环境里 —— 这是和 A 方案最大的区别：容器镜像刻意不含 torch，因为那边由宿主机的模型服务代劳。

**终端 ①｜后端**
```powershell
uvicorn app.main:app --port 8000
```
首次启动会预热本地嵌入/重排模型，等十几秒到一两分钟；出现 `Application startup complete` 即就绪（Swagger 在 http://localhost:8000/docs ）。

**终端 ②｜前端**
```powershell
cd web
npm run dev
```

访问 **http://localhost:5173**。Vite 把 `/api`、`/health` 代理到 :8000，两个进程要同时开着。

> **没有 GPU**：`.env` 里改三项 —— `EMBEDDING_PROVIDER=fastembed`、`RERANKER_ENABLED=false`、`XLING_ENABLED=false`，走纯 ONNX/CPU 路径，这样只装 `requirements.txt` 就够。检索质量会下降，但不需要显卡。
> 三项没改全就装了核心依赖的话，首个任务会在嵌入阶段报 `ModuleNotFoundError: sentence_transformers` —— 因为 `.env.example` 的默认值是 GPU 那套。

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
│   │   ├── cache_service.py    # Redis 缓存（降级可恢复）
│   │   ├── health_service.py   # 依赖探测，/health 与启动校验共用
│   │   ├── db.py               # 关系型 SoR 引擎（SQLite / MySQL）
│   │   └── task_service.py     # 任务持久化（关系库 + Redis 热层）
│   └── prompts/                # planner / analyst / fact_checker / report_writer（可编辑 Markdown）
├── web/                        # React + Vite + TS 前端（详见 web/README.md）
├── ui/streamlit_app.py         # Streamlit 界面（legacy）
├── tests/                      # pytest（asyncio_mode=auto）
├── benchmark/                  # 端到端与逐节点基准
├── rag_experiments/            # RAG 检索质量实验
├── docs/                       # 需求 / 架构 / 部署 / 专题改进历程；版本说明在 docs/version/
├── deployment/
│   ├── start-all.ps1           # 唯一启动入口（模型服务 + 容器栈 + 依赖校验）
│   ├── stop-all.ps1
│   └── docker/
│       ├── compose.yaml        # nginx / api / worker×N / mysql / redis / qdrant
│       ├── Dockerfile          # 多阶段：web-build → py-deps → app / web
│       ├── nginx.conf
│       ├── .env.example        # 部署形态参数与基础设施口令（.env 不入库）
│       └── migrate_sqlite_to_mysql.py
├── .env.example
├── pyproject.toml
├── requirements.txt            # api / worker / 容器镜像（刻意不含 torch）
├── requirements-model.txt      # 上者 + torch 系，模型服务专用
└── requirements-dev.txt        # pytest / ruff / mypy / streamlit
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

## 常见问题

排查顺序都是「先看现象落在哪个组件，再看那个组件的日志」。第一步永远是 `curl http://127.0.0.1/health` —— 五个依赖逐项列出 `ok` 与 `detail`，能直接定位到组件。

容器的日志用 compose 取：

```powershell
docker compose -f deployment/docker/compose.yaml logs --tail 100 worker
```

模型服务是 `start-all.ps1` 用 `-WindowStyle Hidden` 起的宿主机进程，不重定向输出，要看日志就按下面第 1 条手动前台重起一次。

### 1. 模型服务起不来：`[model-server] did not become ready`

先区分是**还在加载**还是**加载失败**：

```powershell
Get-Process -Id (Get-Content .\deployment\runtime\pids\model-server.pid) | Select-Object CPU, WorkingSet64
```

CPU 时间和内存持续增长 = 还在装模型，等着就行（冷启动约 100 秒）。数值几分钟不动 = 已经卡死或退出，按下面带日志重跑看真实报错：

```powershell
$R = ".\deployment\runtime"
$env:HF_HOME = "$R\model-cache\huggingface"; $env:HF_HUB_CACHE = "$env:HF_HOME\hub"
python -m uvicorn app.model_server:app --host 127.0.0.1 --port 8100
```

若报错是 `json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)`，说明 **HuggingFace 缓存的 snapshot 链接坏了**——`deployment/runtime/model-cache/huggingface/hub/*/snapshots/` 下的文件变成了 0 字节。HF 缓存里 `snapshots/` 存的是指向 `blobs/` 的链接，用不保留链接的方式复制或移动过缓存目录（比如换盘）就会留下一堆空文件。

确认：

```powershell
Get-ChildItem .\deployment\runtime\model-cache\huggingface\hub -Recurse -File |
    Where-Object { $_.FullName -like "*\snapshots\*" -and $_.Length -eq 0 } |
    Select-Object -ExpandProperty FullName
```

`blobs/` 里的权重通常还是完好的（bge-m3 与 reranker 各约 2.2GB），可以按文件名把 blob 硬链接回 snapshot 恢复，省一次几 GB 的重下；不想麻烦就直接删掉整个坏掉的 `models--*` 目录让它重新下载。

**另一种情况：没有任何报错，就是卡着不动。** 模型全部命中本地缓存时，sentence-transformers / transformers 仍会去 HuggingFace Hub 校验版本（日志里那句 `You are sending unauthenticated requests to the HF Hub` 就是它）。网络不通或代理不稳时，这些请求会挂很久甚至一直挂住，表现为进程活着、内存停在 2GB 左右、CPU 不涨、`:8100` 永远不监听。

模型已经下全的话，直接切离线模式即可，实测启动时间从「无限期卡住」变成 **20 秒**：

```powershell
$env:HF_HUB_OFFLINE = "1"
```

要长期生效就写进 `deployment/start-all.ps1` 设 `HF_HOME` 那一段。注意开了之后不会再自动拉取新模型——换 `EMBEDDING_MODEL` / `RERANKER_MODEL` 时要先临时关掉它把权重下下来。

### 2. 任务一直不完成，最后 `error_message: 任务被取消`

「被取消」是撞上了 `job_timeout`（`app/core/config.py`，默认 1800 秒），不是根因，要找的是**谁把流水线拖慢了**。

最典型的一种：worker 调不通模型服务，静默回退到进程内加载模型。判断方法是看 worker 的内存——正常应在 1GB 以内（worker 设计上根本不 import torch）：

```powershell
docker stats --no-stream --format "{{.Name}}  {{.MemUsage}}"
```

涨到几 GB 就说明回退发生了（容器镜像不含 torch，这时通常直接 ImportError 而不是变慢；宿主机上跑 worker 才会静默回退）。直接压一下模型服务确认：

```python
import httpx
print(httpx.post("http://127.0.0.1:8100/embed", json={"texts": ["测试"]}, timeout=60).status_code)
```

**返回 502（空 body）而 curl 同一地址正常** —— 这是 Windows 系统代理在拦回环请求。httpx 默认 `trust_env=True`，走 `urllib.request.getproxies()`；该函数在 Windows 上除环境变量外**还会读注册表里的系统代理**（Clash 一类工具常设 `127.0.0.1:某端口`），于是连 `127.0.0.1` 的请求也被塞进代理。环境变量里查不到，所以极难反查：

```python
from urllib.request import getproxies, getproxies_environment
print(getproxies_environment())   # {} —— 环境变量干净
print(getproxies())               # 却返回 {'http': 'http://127.0.0.1:xxxxx', ...}
```

本项目指向模型服务的客户端（`app/services/rag_service.py` 的 `_get_http_client`）已显式设 `trust_env=False`。**新增任何指向 `127.0.0.1` / `localhost` 的 httpx 客户端都要照做**；反过来，crawler、搜索、HuggingFace 下载这些出网请求需要这个代理，不要全局关掉。

### 3. 容器起不来：`ports are not available` / `bind: address already in use`

Nginx 容器要占宿主机的 80 端口。先看是谁占着：

```powershell
Get-NetTCPConnection -State Listen -LocalPort 80 | Select-Object OwningProcess |
    ForEach-Object { Get-Process -Id $_.OwningProcess }
```

改端口比抢端口省事——在 `deployment/docker/.env` 里把 `WEB_PORT` 改成 8080 再重跑 `start-all.ps1` 即可。

如果占用者是早期装在 C 盘的旧 Nginx（`C:\nginx-1.28.0`），用**管理员权限**跑清理脚本 —— 它会先停掉那个进程再删目录：

```powershell
.\deployment\remove-legacy-c-drive.ps1 -WhatIf    # 先看它打算做什么
.\deployment\remove-legacy-c-drive.ps1            # 确认后执行
```

删除前会校验替代品确实在位（nginx 容器已被 compose 创建过），校验不过就拒绝执行。所以正常顺序是先跑一次 `start-all.ps1`——即使它因为端口被占而启动失败也没关系，容器建出来就满足条件了。

### 3b. 依赖校验报 `qdrant 失败：配置 remote 但已回退内存模式`

Qdrant 客户端和服务端的版本必须 major 相同、minor 相差不超过 1。超出范围时症状很隐蔽：`get_collections` 正常、集合也读得到，但 `upsert` 直接失败，表现为研究任务照常跑完、报告却没有任何引用。`compose.yaml` 里的镜像版本固定为 `v1.19.0`，升级 `qdrant-client` 时必须同步改它——`/health` 的 qdrant 一项会同时打印两边版本号。

### 4. 提交的中文问题在报告里变成了乱码或 `?`

先看后端实际收到的是什么：`GET /api/research/{id}` 里的 `query` 字段。

- 存的是 `2025??????????` —— **请求体**编码丢失，问题在客户端。PowerShell 的 `Invoke-RestMethod -Body <string>` 不会按 UTF-8 发送，要显式传字节：`-Body ([System.Text.Encoding]::UTF8.GetBytes($json))`。这种任务不会报错，只会拿着一串 `?` 去搜索，产出完全跑题但看着正常的报告。
- 存的是 `2025å¹´...` —— **响应**解码问题，服务端数据是对的。用 curl 或 Python 复核即可。

## 文档

本文只讲**怎么跑起来**。设计取舍、实验数据和被否决的方案都在 [`docs/`](docs/README.md)：

| | |
|---|---|
| [容器化部署](docs/容器化部署.md) | 当前部署形态为什么这么设计（没用过 Docker 也能读），以及踩过的坑 |
| [技术架构](docs/技术架构文档.md) | 8 节点流水线、模块职责、数据流 |
| [检索与证据](docs/检索与证据.md) | 搜什么、从哪搜、收哪些源；一次"答非所问"的全链路根因诊断 |
| [RAG 改进方案](docs/RAG改进方案.md) | 切片 / 重排序 / HyDE、评估体系、GraphRAG-V 否决实验 |
| [LLM 节点改进](docs/LLM节点改进.md) | 提示词与结构化输出演进，八轮 benchmark |
| [模型服务化与显存内存优化](docs/模型服务化与显存内存优化.md) | 多 worker 崩溃根因（是提交内存，不是显存）、FP16、模型服务化 |
| [版本说明](docs/version/) | v0.1.4 ~ v0.2.1 逐版改动 |

## 许可

MIT
