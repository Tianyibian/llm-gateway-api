from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Smart AI Support API"
    llm_provider: Literal["auto", "openai", "ollama"] = "auto"
    llm_orchestrator: Literal["native", "langchain"] = "native"
    database_url: str = (
        "postgresql+asyncpg://llm_gateway:llm_gateway_local@"
        "127.0.0.1:5432/llm_gateway"
    )
    database_echo: bool = False
    database_pool_size: int = 5
    database_max_overflow: int = 10
    database_pool_timeout_seconds: float = 30.0
    database_pool_recycle_seconds: int = 1800
    business_data_dir: str = "Business_data"
    knowledge_base_dir: str = "Knowledge Base"
    rag_embedding_provider: Literal["ollama", "openai"] = "ollama"
    rag_embedding_dimensions: int = 768
    rag_chunk_size: int = 1200
    rag_chunk_overlap: int = 200
    rag_retrieval_k: int = 5

    openai_api_key: Optional[SecretStr] = None
    openai_base_url: Optional[str] = None
    openai_chat_model: str = "gpt-5.6-luna"
    openai_reason_model: str = "gpt-5.6-terra"
    openai_recommendation_model: str = "gpt-5.6-terra"
    openai_vision_model: str = "gpt-5.6-luna"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_vision_detail: Literal["low", "high", "original", "auto"] = "high"
    vision_max_image_bytes: int = 10 * 1024 * 1024
    openai_reasoning_effort: Literal[
        "none", "low", "medium", "high", "xhigh", "max"
    ] = "medium"
    openai_timeout_seconds: float = 60.0

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = "qwen3:4b"
    ollama_reason_model: str = "qwen3:4b"
    ollama_recommendation_model: str = "qwen3:4b"
    ollama_embedding_model: str = "embeddinggemma"
    ollama_chat_think: bool = True
    ollama_keep_alive: str = "5m"
    ollama_timeout_seconds: float = 300.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
