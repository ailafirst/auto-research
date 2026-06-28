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

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "deep_research_chunks"

    # --- RAG ---
    embedding_provider: Literal["fastembed", "st", "api"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    chunk_size: int = 800
    chunk_overlap: int = 200
    rag_top_k: int = 6
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_k: int = 6       # rerank 后保留的 chunk 数
    reranker_retrieve_k: int = 40  # 触发 rerank 时向量检索扩大到的数量（基准实测 20→40：Recall@6 0.808→0.858）
    # --- 跨语言检索（中文 query 经本地 MT 译为英文，中英双路 dense 检索按 cid 合并）---
    # 解决"中文问题→英文 gold"的 dense 跨语言对齐失效（基准实测 Recall@6 0.858→0.892）。
    # 翻译走独立本地 MT（opus-mt，~20ms/题），不占主 LLM 并发池。
    xling_enabled: bool = False
    translation_model: str = "Helsinki-NLP/opus-mt-zh-en"

    # --- App ---
    app_env: Literal["development", "production", "testing"] = "development"
    log_level: str = "INFO"
    max_rounds: int = 2
    request_timeout: int = 30
    max_concurrent_fetches: int = 5
    max_sources_per_round: int = 20

    # --- Secret ---
    secret_key: str = "change-me-in-production"


settings = Settings()
