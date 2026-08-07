"""配置管理 — 基于 Pydantic Settings。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置，从环境变量或 .env 文件加载。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM ---
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_max_tokens: int = 16384
    llm_temperature: float = 0.3
    # 进程级全局 LLM 并发闸门（生产）。所有经 LLMService.chat() 的调用共用此上限。
    # 实测 mimo 后端在生产负载（5-10s/请求）下并发 12 全部成功、无 429。
    llm_max_concurrency: int = 12
    # analyst 节点内子问题并发数（单次节点调用内的 LLM 并发）
    analyst_concurrency: int = 8
    # fact_checker 节点内子问题并发数
    fact_checker_concurrency: int = 5
    # benchmark 专用总 LLM 预算：pipeline（含 analyst/fact_checker）+ RAGAS 共享此闸门。
    # 比生产略低（10 < 12），为长时间基准测试中的 API 波动留缓冲。
    llm_benchmark_concurrency: int = 10

    # --- Search ---
    use_tavily: bool = True    # true=Tavily，false=DuckDuckGo
    tavily_api_key: str = ""
    serper_api_key: str = ""
    bing_api_key: str = ""
    max_search_results: int = 5
    # advanced 每请求耗 2 credit 且 raw_content 拉回整页正文（payload 大、易触发套餐
    # 用量上限 ForbiddenError）。basic 仅 1 credit、返回摘要，正文由 content_extractor
    # 爬虫补抓（爬虫有 30s 超时，稳健）。配额紧张/基准测试时可降级为 basic。
    tavily_search_depth: Literal["basic", "advanced"] = "advanced"
    tavily_include_raw_content: bool = True
    # 学术源双路（方案①）：每次查询在通用搜索外，再并发一路限定学术白名单域的 Tavily
    # 检索并合并，让研究/综述源进入候选池；相关性交给 reranker 自过滤（统一生效，不猜
    # 问题类型）。仅 Tavily 路生效。详见 docs/检索与证据.md。
    academic_search_enabled: bool = True

    # --- Qdrant ---
    # memory=进程内实例（零基础设施，单进程内跑完一个任务够用，为历史默认）；
    # remote=连 qdrant_url 指向的服务，向量可持久化并跨进程共享。remote 不可达时
    # 自动回退 memory（降级安全，与 Redis/队列/模型服务一致），实际生效模式见 /health。
    qdrant_mode: Literal["memory", "remote"] = "memory"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "deep_research_chunks"
    # 证据向量保留天数（0=不清理）。memory 模式下向量随进程消失，无需过期策略；
    # remote 模式会一直累积（实测单任务约 2000 chunk ≈ 10MB），必须有时效性。
    # 取 7 天是因为向量只在任务执行期间（含多轮补证，分钟级）被检索，任务完成后
    # 报告已落 MySQL，向量再无读取方——留一周纯粹是给排查留窗口。
    vector_ttl_days: int = 7
    # 过期清理的执行时刻（worker 内 arq cron，UTC）。默认每天 03:00，错开白天使用。
    vector_cleanup_hour: int = 3

    # --- RAG ---
    embedding_provider: Literal["fastembed", "st", "api"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    chunk_size: int = 800
    chunk_overlap: int = 200
    rag_top_k: int = 6
    # 本地模型推理精度（仅 cuda 生效；cpu 下 fp16 多数算子不支持，自动退回 fp32）。
    # fp16 使 bge-m3 显存 2166→1083 MB、reranker 同比减半。对检索质量的影响见
    # rag_experiments/experiment_fp16.py 在 120 题黄金集上的实测。
    model_dtype: Literal["fp32", "fp16"] = "fp32"
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_k: int = 6       # rerank 后保留的 chunk 数
    reranker_retrieve_k: int = 40  # 触发 rerank 时向量检索扩大到的数量（基准实测 20→40：Recall@6 0.808→0.858）
    # --- 跨语言检索（中文 query 经本地 MT 译为英文，中英双路 dense 检索按 cid 合并）---
    # 解决"中文问题→英文 gold"的 dense 跨语言对齐失效（基准实测 Recall@6 0.858→0.892）。
    # 翻译走独立本地 MT（opus-mt，~20ms/题），不占主 LLM 并发池。
    xling_enabled: bool = True
    translation_model: str = "Helsinki-NLP/opus-mt-zh-en"

    # --- 模型服务化（可选，独立 FastAPI 进程集中装载 embed/rerank/translate）---
    # 非空时，worker 全部通过 HTTP 调用模型服务，进程内不 import torch/transformers，
    # 不建 CUDA context（实测每 worker commit 6.9GB→0.75GB）。服务不可达时自动回退
    # 本地加载（降级安全，与 Redis/队列一致）。启动：uvicorn app.model_server:app --port 8100
    model_service_url: str = ""
    model_service_timeout: float = 30.0

    # --- 任务队列（arq，多 worker）---
    # true=优先入 arq 队列由独立 worker 池执行；不可用时回退进程内 asyncio.create_task。
    queue_enabled: bool = True
    worker_max_jobs: int = 4        # 单 worker 进程内并发任务数
    job_timeout: int = 1800         # 单个研究任务超时（秒）

    # --- 持久化（关系型系统记录 SoR）---
    # 默认 SQLite（零基础设施；WAL 支持单机多 worker 共享）。多机部署时改成
    # mysql+aiomysql://user:pass@host/db 或 postgresql+asyncpg://... 即可，代码不变。
    database_url: str = "sqlite+aiosqlite:///output/deepresearch.db"

    # --- 分布式限流（Redis 令牌桶）---
    # 进程内 asyncio.Semaphore 只在单进程有效；多 worker 下用 Redis 令牌桶做「全局速率
    # 上限」，本地信号量继续兜进程内并发。Redis 不可用时 acquire 直接放行（降级安全）。
    llm_rate_limit_enabled: bool = True
    llm_rate_limit_per_sec: float = 12.0   # 全局每秒放行的 LLM 请求数
    llm_rate_burst: int = 12               # 令牌桶容量（突发上限）

    # --- Cache (Redis) ---
    # redis_url 为空 → 缓存整体禁用（CacheService 静默旁路，绝不因缓存故障中断研究）。
    # cache_enabled 是总开关：benchmark 可置 false 以避免缓存扭曲评测（embedding 缓存
    # 确定性、可保持开启，见 benchmark 说明）。cache_version 改动会使全部旧 key 失效。
    redis_url: str = ""
    cache_enabled: bool = True
    cache_version: str = "v1"
    search_cache_ttl: int = 21600      # 搜索结果缓存 6h（省 Tavily 配额）
    emb_cache_ttl: int = 2592000       # embedding 缓存 30d（确定性，可长存）

    # --- App ---
    app_env: Literal["development", "production", "testing"] = "development"
    log_level: str = "INFO"
    max_rounds: int = 2
    request_timeout: int = 30
    max_concurrent_fetches: int = 5
    max_sources_per_round: int = 20

    # --- /health 明细令牌 ---
    # /health 经 Nginx 公网可达，而它的 dependencies[].detail 里是内部主机名、端口、
    # 数据库用户名、各组件版本号和失败时的异常原文——对排查有用，对公网访客则是白送
    # 的踩点信息。为空（默认）时任何调用方都只拿到 name/ok/skipped；配了值以后，带
    # `X-Health-Token: <值>` 的请求才能看到 detail。
    #
    # 不用「按来源 IP 判断内网」那套：本项目的公网入口是 frp 隧道，frpc 在宿主机上
    # 回连 127.0.0.1:80，公网流量到 Nginx 时的 $remote_addr 和本机访问完全一样，
    # 在 IP 层面根本区分不开。
    health_detail_token: str = ""


settings = Settings()


def mask_dsn(dsn: str) -> str:
    """把 DSN 里的口令替换成 ***，供日志与 /health 输出使用。

    容器化部署后 Redis / MySQL 都启用了口令，而口令是写在 URL 里的。这些 URL 原本
    被直接打进日志，而 /health 的 dependencies[].detail 又会把它们返回给调用方——
    该端点经 Nginx 公网可达，等于把基础设施口令挂到公网上。凡是要把 DSN 交给
    日志或响应体的地方，都必须先过这个函数。
    """
    if not dsn or "@" not in dsn:
        return dsn
    scheme, sep, rest = dsn.partition("://")
    if not sep:
        return dsn
    creds, _, host = rest.rpartition("@")
    if ":" not in creds:
        # 只有用户名没有口令，无需隐去
        return dsn
    user, _, _password = creds.partition(":")
    return f"{scheme}://{user}:***@{host}"
