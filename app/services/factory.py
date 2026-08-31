from __future__ import annotations

import sys
from typing import Any

from openai import AsyncOpenAI

from app.core.config import Settings, get_settings
from app.services.base import LLMService, ServiceType
from app.services.assistant_service import AssistantGraphService
from app.services.errors import LLMConfigurationError
from app.services.langchain_service import LangChainChatService
from app.services.ollama_service import OllamaChatService
from app.services.openai_service import OpenAIResponsesService
from app.services.product_catalog import ProductCatalog
from app.services.query_classifier import QueryClassifier
from app.services.vision_service import OpenAIVisionService


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
        )
        catalog = ProductCatalog(self.settings.business_data_dir)
        try:
            vision_service = self.create_vision_service()
        except LLMConfigurationError:
            vision_service = None
        return AssistantGraphService.from_model(
            classifier=classifier,
            model_client=answer_client,
            product_catalog=catalog,
            vision_service=vision_service,
            provider=provider,
            model=model,
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
