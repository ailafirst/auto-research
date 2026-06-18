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

    # --- Search ---
    search_provider: str = "duckduckgo"
    tavily_api_key: str = ""
    serper_api_key: str = ""
    bing_api_key: str = ""
    max_search_results: int = 5

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "deep_research_chunks"

    # --- RAG ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    chunk_size: int = 800
    chunk_overlap: int = 120
    rag_top_k: int = 6

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
