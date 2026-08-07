# Deep Research Agent — 需求与进度

> 需求部分整理自源文件 `参考文档.txt`（本目录）。
> 本文合并了原《开发任务拆解表》，作为需求与完成度的单一入口。

## 项目概述

**技术栈**：LangGraph · RAG（bge-m3 + bge-reranker-v2-m3 + Qdrant）· FastAPI · arq · LLM

**核心目标**：用户输入自然语言问题，系统自动完成深度研究，输出**带引用来源、结构清晰、逻辑完整**的研究报告。

关键约束（决定了后续所有设计）：

- 报告中的每条具体声明都必须能回溯到真实信源 —— 引用体系与事实核查是主线而非附加功能
- 单次研究要在可接受时长内完成 —— 全流程异步 + 节点内并发
- 任一外部依赖（搜索引擎 / Redis / 向量库 / 模型服务）不可用时**降级而非崩溃**

相关文档：[技术架构](技术架构文档.md) · [容器化部署](容器化部署.md) · [版本说明](version/)

---

## 完成度

### 第一阶段：MVP ✅

| 任务 | 文件 |
|---|---|
| FastAPI 应用入口 / API 路由 | `app/main.py`、`app/api/routes_research.py` |
| LangGraph 工作流与状态定义 | `app/graph/builder.py`、`app/graph/state.py` |
| Planner / Retriever / Analyst / Report Writer 节点 | `app/graph/nodes.py` |
| 网页正文抓取 | `app/services/crawler_service.py` |
| Qdrant 向量入库 与 RAG 检索 | `app/services/vector_store.py`、`rag_service.py` |
| 任务管理服务 | `app/services/task_service.py` |
| Web 界面 | `web/`（React + Vite）；`ui/streamlit_app.py` 为 v0.2.0 前的旧界面，保留但不再维护 |

### 第二阶段：事实核查与多轮研究 ✅

Fact Checker 节点、证据不足判断、自动补充检索、最大轮数控制（均在 `app/graph/nodes.py`），任务状态与进度查询 API（`routes_research.py`）。多轮从"原样重跑"演进到"增量补证"的过程见 [LLM 节点改进 §五](LLM节点改进.md) 与 [v0.2.1](version/v0.2.1.md)。

### 第三阶段：评估与优化 ✅

`tests/`（api / graph / rag / models / config / health_redaction）、`app/prompts/*.md` 提示词模板、`app/core/` 的异常/日志/配置。评测体系另有 `benchmark/`（12 任务黄金集 + RAGAS）与 `rag_experiments/`（不改生产代码的对照实验）。

### 第四阶段：产品化

| 项 | 状态 |
|---|---|
| Reranker 集成 | ✅ v0.1.4（bge-reranker-v2-m3） |
| RAGAS 评估 | ✅ v0.1.4（faithfulness / context_precision） |
| 历史任务管理 | ✅ 任务持久化到 SQLite / MySQL，`GET /api/research` 列表 |
| 容器化部署 | ✅ v0.2.x，见 [容器化部署](容器化部署.md)（原计划的根目录 `docker-compose.yml` 已由 `deployment/docker/compose.yaml` 取代） |
| 用户鉴权 | ❌ 未做 —— 公网 `/api` 目前无鉴权，是已知遗留项 |
| WebSocket 实时进度 | ❌ 未做 —— 当前为 HTTP 轮询 |
| PDF / DOCX 导出 | ❌ 未做 |
| 本地文档上传（私有知识源） | ❌ 未做 |
| 多 Agent 协作、管理后台 | ❌ 未做，优先级低 |
