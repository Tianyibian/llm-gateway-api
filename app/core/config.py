from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.neo4j"),
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
    graphrag_guardrail_timeout_seconds: float = 90.0
    graphrag_guardrail_min_confidence: float = 0.75
    microsoft_graphrag_enabled: bool = False
    microsoft_graphrag_root: str = ".local/graphrag-ms-business-v1"
    microsoft_graphrag_python: str = ".venv-graphrag/bin/python"
    microsoft_graphrag_timeout_seconds: float = 90
    neo4j_enabled: bool = False
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: Optional[str] = None
    neo4j_password: Optional[SecretStr] = None
    neo4j_database: str = "neo4j"
    neo4j_dataset: str = "aster-business-v1"
    neo4j_query_timeout_seconds: float = 10
    neo4j_plan_timeout_seconds: float = 60
    neo4j_max_estimated_rows: float = Field(default=100_000, gt=0, allow_inf_nan=False)
    graph_answer_timeout_seconds: float = Field(default=120, gt=0, le=600)
    neo4j_load_user: Optional[str] = None
    neo4j_load_password: Optional[SecretStr] = None

    # Snowflake is an optional OLAP backend. PostgreSQL remains the operational
    # conversation database even when this integration is enabled.
    snowflake_enabled: bool = False
    analytics_backend: Literal["neo4j", "snowflake"] = "neo4j"
    snowflake_account: Optional[str] = None
    snowflake_user: Optional[str] = None
    snowflake_auth_method: Literal["password", "key_pair"] = "password"
    snowflake_password: Optional[SecretStr] = None
    snowflake_private_key_file: Optional[str] = None
    snowflake_private_key_passphrase: Optional[SecretStr] = None
    snowflake_warehouse: Optional[str] = None
    snowflake_database: Optional[str] = None
    snowflake_schema: Optional[str] = None
    snowflake_role: Optional[str] = None
    snowflake_load_role: Optional[str] = None
    snowflake_pool_size: int = 2
    snowflake_max_overflow: int = 3
    snowflake_timeout_seconds: int = 30

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
