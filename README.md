# 🔬 Deep Research Agent

基于 **LangGraph**、**RAG** 与 **FastAPI** 的自动化深度调研系统。

用户输入一个研究问题，系统自动完成问题分解、联网搜索、内容抓取、RAG 证据构建、多轮分析、事实核查，最终生成带引用来源的结构化研究报告。

## 技术栈

| 模块 | 技术 |
|------|------|
| Agent 编排 | LangGraph |
| LLM 调用 | LiteLLM (OpenAI/Claude/任意 API) |
| API 服务 | FastAPI + Uvicorn |
| 向量数据库 | Qdrant |
| Embedding | fastembed (BGE) |
| 网页抓取 | httpx + BeautifulSoup |
| 搜索引擎 | DuckDuckGo / Tavily / Serper |
| Web UI | Streamlit |
| 配置管理 | Pydantic Settings + .env |

## 快速开始

### 1. 环境准备

```bash
# 激活 conda 环境
conda activate deepresearch

# 安装依赖
pip install -r requirements.txt

# 复制并配置环境变量
cp .env.example .env
# 编辑 .env 设置 LLM_API_KEY 等
```

### 2. 启动 API 服务

```bash
# 开发模式
uvicorn app.main:app --reload --port 8000

# 或 CLI 模式
python -m app.main "你的研究问题"
```

### 4. 启动 Web UI

```bash
streamlit run ui/streamlit_app.py
```

访问 `http://localhost:8501` 使用界面。

## 项目结构

```
deepresearch/
├── app/
│   ├── main.py                 # FastAPI 应用入口
│   ├── api/
│   │   ├── schemas.py          # API 请求/响应模型
│   │   └── routes_research.py  # 研究任务 API 路由
│   ├── core/
│   │   ├── config.py           # 全局配置 (Pydantic Settings)
│   │   ├── logging.py          # 日志配置
│   │   └── exceptions.py       # 统一异常定义
│   ├── graph/
│   │   ├── state.py            # LangGraph 状态定义
│   │   ├── nodes.py            # 工作流节点实现
│   │   └── builder.py          # 图构建
│   ├── services/
│   │   ├── llm_service.py      # LLM 调用服务
│   │   ├── search_service.py   # 搜索服务
│   │   ├── crawler_service.py  # 网页抓取服务
│   │   ├── vector_store.py     # Qdrant 向量数据库
│   │   ├── rag_service.py      # RAG 证据构建与检索
│   │   └── task_service.py     # 任务管理
│   └── prompts/
│       ├── planner.md          # 规划提示词
│       ├── analyst.md          # 分析提示词
│       ├── fact_checker.md     # 核查提示词
│       └── report_writer.md    # 写作提示词
├── ui/
│   └── streamlit_app.py        # Streamlit Web 界面
├── tests/
│   ├── test_api.py
│   ├── test_graph.py
│   ├── test_rag.py
│   └── test_models.py
├── docs/
│   ├── PRD.md                  # 需求文档
│   ├── 技术架构文档.md           # 架构设计
│   └── 开发任务拆解表.md          # 开发进度
├── docker-compose.yml          # Qdrant + API 容器化
├── .env.example                # 环境变量示例
├── pyproject.toml
└── requirements.txt
```

## API 文档

启动 API 后访问 `http://localhost:8000/docs` 查看 Swagger 文档。

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/research` | POST | 创建研究任务 |
| `/api/research/{id}/status` | GET | 查询任务状态 |
| `/api/research/{id}` | GET | 查询任务详情 |
| `/api/research/{id}/report` | GET | 获取研究报告 |
| `/api/research` | GET | 列出所有任务 |
| `/health` | GET | 健康检查 |

## LangGraph 工作流

```
用户输入 → Planner → Retriever → Content Extractor
    → Source Evaluator → Evidence Builder → Analyst
    → Fact Checker → (证据不足? → Retriever 补充搜索)
    → Report Writer → Markdown 报告
```

## 开发阶段

- **MVP** ✅ — 完整流程已实现
- **事实核查与多轮研究** ✅ — Fact Checker + 补充检索
- **测试与优化** ✅ — 单元测试 + 提示词模板
- **产品化扩展** 🔜 — 待后续迭代

## 许可

MIT
