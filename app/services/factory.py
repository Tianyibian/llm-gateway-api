from __future__ import annotations

import sys
from typing import Any

from openai import AsyncOpenAI

from app.core.config import Settings, get_settings
from app.db.session import async_session_factory
from app.services.base import LLMService, ServiceType
from app.services.assistant_service import AssistantGraphService
from app.services.analytics_planner import AnalyticsPlanner
from app.services.errors import LLMConfigurationError
from app.services.embedding_service import (
    EmbeddingService,
    OllamaEmbeddingService,
    OpenAIEmbeddingService,
)
from app.services.knowledge_service import KnowledgeIngestionService, KnowledgeRetriever
from app.services.langchain_service import LangChainChatService
from app.services.ollama_service import OllamaChatService
from app.services.openai_service import OpenAIResponsesService
from app.services.product_catalog import ProductCatalog
from app.services.query_classifier import QueryClassifier
from app.services.snowflake_analytics import SnowflakeAnalyticsService
from app.services.vision_service import OpenAIVisionService
from app.services.graphrag_guardrail import GraphRAGGuardrail
from app.services.graph_supervisor import GraphRAGSupervisor
from app.services.graph_agents import HierarchicalGraphSupervisor
from app.services.clarification_service import ClarificationService
from app.services.graph_answer import GraphAnswerGenerator
from app.services.microsoft_graphrag import MicrosoftGraphRAGClient, MicrosoftGraphRAGTool, MODES
from app.models.graph_supervisor import SupervisorLimits, GraphTool


class LLMServiceFactory:
    """Create chat/reason/recommendation services from environment configuration."""

    _INSTRUCTIONS = {
        ServiceType.CHAT: None,
        ServiceType.REASON: (
            "Reason carefully before answering. Return a clear final answer and a concise, "
            "user-facing explanation of the important steps. Do not claim to expose hidden "
            "chain-of-thought."
        ),
        ServiceType.RECOMMENDATION: (
            "Give practical recommendations. State the criteria, tradeoffs, and any important "
            "uncertainty behind the recommendation."
        ),
    }

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def resolve_provider(self) -> str:
        """Resolve auto at service-creation time without runtime error fallback."""
        if self.settings.llm_provider != "auto":
            return self.settings.llm_provider

        if self.settings.openai_api_key is None:
            return "ollama"
        key = self.settings.openai_api_key.get_secret_value().strip()
        placeholder_values = {
            "",
            "your_openai_api_key_here",
            "replace_with_your_own_secret_key",
        }
        return "ollama" if key in placeholder_values else "openai"

    def resolve_analytics_backend(self) -> str:
        """Configuration-time selection, never fallback after a database error."""
        return "snowflake" if self.settings.snowflake_enabled and self.settings.analytics_backend == "snowflake" else "neo4j"

    def create(self, service_type: ServiceType) -> LLMService:
        provider = self.resolve_provider()
        if self.settings.llm_orchestrator == "langchain":
            return self._create_langchain(service_type, provider)
        if provider == "openai":
            return self._create_openai(service_type)
        if provider == "ollama":
            return self._create_ollama(service_type)
        raise LLMConfigurationError(f"Unsupported LLM provider: {provider}")

    def create_classifier(self) -> QueryClassifier:
        """Create the LangChain structured-output query classifier."""
        provider = self.resolve_provider()
        model_by_provider = {
            "openai": self.settings.openai_chat_model,
            "ollama": self.settings.ollama_chat_model,
        }
        try:
            model = model_by_provider[provider]
        except KeyError as exc:
            raise LLMConfigurationError(
                f"Unsupported classifier provider: {provider}"
            ) from exc

        model_client = self._build_langchain_model(
            provider=provider,
            model=model,
            service_type=ServiceType.CHAT,
            reasoning_override=False,
        )
        return QueryClassifier.from_model(
            model_client=model_client,
            provider=provider,
            model=model,
            analytics_backend=self.resolve_analytics_backend(),
        )

    def create_graphrag_guardrail(self, *, scope_only=False) -> GraphRAGGuardrail:
        provider = self.resolve_provider()
        model = (self.settings.openai_chat_model if provider == "openai"
                 else self.settings.ollama_chat_model)
        client = self._build_langchain_model(
            provider=provider, model=model, service_type=ServiceType.CHAT,
            reasoning_override=False,
        )
        return self._guardrail_from_model(client, provider, scope_only=scope_only)

    def create_graph_supervisor(self, *, tools=None) -> GraphRAGSupervisor:
        provider = self.resolve_provider()
        model = (self.settings.openai_chat_model if provider == "openai"
                 else self.settings.ollama_chat_model)
        client = self._build_langchain_model(
            provider=provider, model=model, service_type=ServiceType.CHAT,
            reasoning_override=False,
        )
        return HierarchicalGraphSupervisor.from_model(
            client, guardrail=self._guardrail_from_model(client, provider, scope_only=True),
            task_guardrail=self._guardrail_from_model(client, provider),
            tools=self.create_graph_retrieval_tools(client) if tools is None else tools,
            answer_generator=GraphAnswerGenerator.from_model(client, timeout=self.settings.graph_answer_timeout_seconds),
            limits=SupervisorLimits(call_timeout_seconds=90, timeout_seconds=480),
        )

    def create_microsoft_graphrag_client(self) -> MicrosoftGraphRAGClient:
        if not self.settings.microsoft_graphrag_enabled or self.settings.openai_api_key is None:
            raise LLMConfigurationError("Microsoft GraphRAG requires enabled configuration and an OpenAI key.")
        return MicrosoftGraphRAGClient(
            root=self.settings.microsoft_graphrag_root, python=self.settings.microsoft_graphrag_python,
            api_key=self.settings.openai_api_key.get_secret_value(),
            timeout=self.settings.microsoft_graphrag_timeout_seconds,
        )

    def create_neo4j_service(self, model_client=None):
        from app.services.neo4j_service import Neo4jExecutor, TextToCypherService
        executor = Neo4jExecutor(self.settings)
        if model_client is None:
            provider = self.resolve_provider()
            model = self.settings.openai_chat_model if provider == "openai" else self.settings.ollama_chat_model
            model_client = self._build_langchain_model(provider=provider, model=model,
                service_type=ServiceType.CHAT, reasoning_override=False)
        return TextToCypherService.from_model(model_client, executor=executor,
            dataset=self.settings.neo4j_dataset, timeout=self.settings.neo4j_plan_timeout_seconds)

    def create_graph_retrieval_tools(self, model_client=None):
        tools = {}
        if self.settings.microsoft_graphrag_enabled:
            try:
                client = self.create_microsoft_graphrag_client()
                if client.status()["ready"]:
                    tools.update({tool: MicrosoftGraphRAGTool(client, mode) for tool, mode in MODES.items()})
            except LLMConfigurationError:
                pass
        if self.settings.neo4j_enabled:
            try:
                tools[GraphTool.NEO4J] = self.create_neo4j_service(model_client)
            except LLMConfigurationError:
                pass
        return tools

    def _guardrail_from_model(self, client: Any, provider: str, *, scope_only=False) -> GraphRAGGuardrail:
        if provider == "ollama":
            # Bound latency for the graph gate without changing answer generation.
            client = client.model_copy(update={"reasoning": False, "temperature": 0, "num_ctx": 8192})
        return GraphRAGGuardrail.from_model(
            client, timeout_seconds=self.settings.graphrag_guardrail_timeout_seconds,
            min_confidence=self.settings.graphrag_guardrail_min_confidence,
            scope_only=scope_only,
        )

    def create_assistant(self) -> AssistantGraphService:
        """Create the LangGraph assistant and its route-specific dependencies."""
        provider = self.resolve_provider()
        model_by_provider = {
            "openai": self.settings.openai_chat_model,
            "ollama": self.settings.ollama_chat_model,
        }
        try:
            model = model_by_provider[provider]
        except KeyError as exc:
            raise LLMConfigurationError(
                f"Unsupported assistant provider: {provider}"
            ) from exc

        classifier_client = self._build_langchain_model(
            provider=provider,
            model=model,
            service_type=ServiceType.CHAT,
            reasoning_override=False,
        )
        answer_client = self._build_langchain_model(
            provider=provider,
            model=model,
            service_type=ServiceType.CHAT,
            reasoning_override=True if provider == "ollama" else None,
        )
        classifier = QueryClassifier.from_model(
            model_client=classifier_client,
            provider=provider,
            model=model,
            analytics_backend=self.resolve_analytics_backend(),
        )
        analytics_planner = AnalyticsPlanner.from_model(classifier_client)
        analytics_service = None
        if self.resolve_analytics_backend() == "snowflake":
            analytics_service = SnowflakeAnalyticsService.from_settings(self.settings)
        catalog = ProductCatalog(self.settings.business_data_dir)
        knowledge_retriever = self.create_knowledge_retriever()
        try:
            vision_service = self.create_vision_service()
        except LLMConfigurationError:
            vision_service = None
        graph_guardrail = self._guardrail_from_model(classifier_client, provider, scope_only=True)
        return AssistantGraphService.from_model(
            classifier=classifier,
            clarification_service=ClarificationService.from_model(classifier_client, analytics_backend=self.resolve_analytics_backend()),
            model_client=answer_client,
            product_catalog=catalog,
            vision_service=vision_service,
            knowledge_retriever=knowledge_retriever,
            analytics_planner=analytics_planner,
            analytics_service=analytics_service,
            graph_guardrail=graph_guardrail,
            graph_supervisor=HierarchicalGraphSupervisor.from_model(
                classifier_client, guardrail=graph_guardrail,
                task_guardrail=self._guardrail_from_model(classifier_client, provider),
                tools=self.create_graph_retrieval_tools(classifier_client),
                answer_generator=GraphAnswerGenerator.from_model(answer_client, timeout=self.settings.graph_answer_timeout_seconds),
                limits=SupervisorLimits(call_timeout_seconds=90, timeout_seconds=480),
            ),
            provider=provider,
            model=model,
        )

    def create_embedding_service(self) -> EmbeddingService:
        """Create the configured fixed-dimension RAG embedding adapter."""
        dimensions = self.settings.rag_embedding_dimensions
        if dimensions != 768:
            raise LLMConfigurationError(
                "RAG_EMBEDDING_DIMENSIONS must be 768 for the current pgvector schema."
            )

        if self.settings.rag_embedding_provider == "ollama":
            return OllamaEmbeddingService(
                base_url=self.settings.ollama_base_url,
                model=self.settings.ollama_embedding_model,
                dimensions=dimensions,
                timeout_seconds=self.settings.ollama_timeout_seconds,
            )

        if self.settings.openai_api_key is None:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is required when RAG_EMBEDDING_PROVIDER=openai."
            )
        client = AsyncOpenAI(
            api_key=self.settings.openai_api_key.get_secret_value(),
            base_url=self.settings.openai_base_url,
            timeout=self.settings.openai_timeout_seconds,
        )
        return OpenAIEmbeddingService(
            client=client,
            model=self.settings.openai_embedding_model,
            dimensions=dimensions,
        )

    def create_knowledge_retriever(self) -> KnowledgeRetriever:
        """Create the public Knowledge Base pgvector retriever."""
        return KnowledgeRetriever(
            session_factory=async_session_factory,
            embedding_service=self.create_embedding_service(),
            default_k=self.settings.rag_retrieval_k,
        )

    def create_knowledge_ingestion_service(self) -> KnowledgeIngestionService:
        """Create the idempotent local Knowledge Base ingestion pipeline."""
        return KnowledgeIngestionService(
            session_factory=async_session_factory,
            embedding_service=self.create_embedding_service(),
            knowledge_base_dir=self.settings.knowledge_base_dir,
            chunk_size=self.settings.rag_chunk_size,
            chunk_overlap=self.settings.rag_chunk_overlap,
        )

    def create_vision_service(self) -> OpenAIVisionService:
        """Create an OpenAI-only image understanding service."""
        if self.settings.openai_api_key is None:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is required for image analysis. Add it to the local .env file."
            )
        api_key = self.settings.openai_api_key.get_secret_value().strip()
        if api_key in {
            "",
            "your_openai_api_key_here",
            "replace_with_your_own_secret_key",
        }:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is required for image analysis. Add a real key to .env."
            )

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.settings.openai_base_url,
            timeout=self.settings.openai_timeout_seconds,
        )
        return OpenAIVisionService(
            client=client,
            model=self.settings.openai_vision_model,
            detail=self.settings.openai_vision_detail,
        )

    def _create_langchain(
        self,
        service_type: ServiceType,
        provider: str,
    ) -> LLMService:
        model_by_provider = {
            "openai": {
                ServiceType.CHAT: self.settings.openai_chat_model,
                ServiceType.REASON: self.settings.openai_reason_model,
                ServiceType.RECOMMENDATION: self.settings.openai_recommendation_model,
            },
            "ollama": {
                ServiceType.CHAT: self.settings.ollama_chat_model,
                ServiceType.REASON: self.settings.ollama_reason_model,
                ServiceType.RECOMMENDATION: self.settings.ollama_recommendation_model,
            },
        }
        try:
            model = model_by_provider[provider][service_type]
        except KeyError as exc:
            raise LLMConfigurationError(
                f"Unsupported LangChain provider: {provider}"
            ) from exc

        model_client = self._build_langchain_model(
            provider=provider,
            model=model,
            service_type=service_type,
        )
        return LangChainChatService(
            model_client=model_client,
            model=model,
            service_type=service_type,
            provider=provider,
            system_instruction=self._INSTRUCTIONS[service_type],
        )

    def _build_langchain_model(
        self,
        *,
        provider: str,
        model: str,
        service_type: ServiceType,
        reasoning_override: bool | None = None,
    ) -> Any:
        if sys.version_info < (3, 10):
            raise LLMConfigurationError(
                "LangChain 1.x requires Python 3.10 or newer. Use the native "
                "orchestrator or recreate the virtual environment with Python 3.10+."
            )

        if provider == "openai":
            if self.settings.openai_api_key is None:
                raise LLMConfigurationError(
                    "OPENAI_API_KEY is missing. Copy .env.example to .env and add your own key."
                )
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:
                raise LLMConfigurationError(
                    "LangChain OpenAI support is not installed. Run "
                    "'python -m pip install -r requirements-langchain.txt'."
                ) from exc

            options: dict[str, Any] = {
                "model": model,
                "api_key": self.settings.openai_api_key.get_secret_value(),
                "base_url": self.settings.openai_base_url,
                "timeout": self.settings.openai_timeout_seconds,
                "use_responses_api": True,
            }
            if service_type is not ServiceType.CHAT:
                options["reasoning_effort"] = self.settings.openai_reasoning_effort
            return ChatOpenAI(**options)

        if provider == "ollama":
            try:
                from langchain_ollama import ChatOllama
            except ImportError as exc:
                raise LLMConfigurationError(
                    "LangChain Ollama support is not installed. Run "
                    "'python -m pip install -r requirements-langchain.txt'."
                ) from exc

            return ChatOllama(
                model=model,
                base_url=self.settings.ollama_base_url,
                keep_alive=self.settings.ollama_keep_alive,
                reasoning=(
                    reasoning_override
                    if reasoning_override is not None
                    else (
                        self.settings.ollama_chat_think
                        if service_type is ServiceType.CHAT
                        else True
                    )
                ),
            )

        raise LLMConfigurationError(f"Unsupported LangChain provider: {provider}")

    def _create_openai(self, service_type: ServiceType) -> LLMService:
        if self.settings.openai_api_key is None:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is missing. Copy .env.example to .env and add your own key."
            )

        model_by_type = {
            ServiceType.CHAT: self.settings.openai_chat_model,
            ServiceType.REASON: self.settings.openai_reason_model,
            ServiceType.RECOMMENDATION: self.settings.openai_recommendation_model,
        }
        effort = (
            "none"
            if service_type is ServiceType.CHAT
            else self.settings.openai_reasoning_effort
        )
        client = AsyncOpenAI(
            api_key=self.settings.openai_api_key.get_secret_value(),
            base_url=self.settings.openai_base_url,
            timeout=self.settings.openai_timeout_seconds,
        )
        return OpenAIResponsesService(
            client=client,
            model=model_by_type[service_type],
            service_type=service_type,
            reasoning_effort=effort,
            system_instruction=self._INSTRUCTIONS[service_type],
        )

    def _create_ollama(self, service_type: ServiceType) -> LLMService:
        model_by_type = {
            ServiceType.CHAT: self.settings.ollama_chat_model,
            ServiceType.REASON: self.settings.ollama_reason_model,
            ServiceType.RECOMMENDATION: self.settings.ollama_recommendation_model,
        }
        return OllamaChatService(
            base_url=self.settings.ollama_base_url,
            model=model_by_type[service_type],
            service_type=service_type,
            # With Qwen 3, think=True keeps reasoning in message.thinking so the
            # adapter can omit it and stream only message.content.
            think=(
                self.settings.ollama_chat_think
                if service_type is ServiceType.CHAT
                else True
            ),
            keep_alive=self.settings.ollama_keep_alive,
            timeout_seconds=self.settings.ollama_timeout_seconds,
            system_instruction=self._INSTRUCTIONS[service_type],
        )
