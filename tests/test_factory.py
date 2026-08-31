import pytest

from app.core.config import Settings
from app.services.base import ServiceType
from app.services.errors import LLMConfigurationError
from app.services.factory import LLMServiceFactory


def test_missing_openai_key_has_actionable_error() -> None:
    factory = LLMServiceFactory(
        Settings(llm_provider="openai", openai_api_key=None)
    )

    with pytest.raises(LLMConfigurationError, match="OPENAI_API_KEY is missing"):
        factory.create(ServiceType.CHAT)


def test_factory_builds_three_service_types() -> None:
    factory = LLMServiceFactory(
        Settings(llm_provider="openai", openai_api_key="test-key")
    )

    chat = factory.create(ServiceType.CHAT)
    reason = factory.create(ServiceType.REASON)
    recommendation = factory.create(ServiceType.RECOMMENDATION)

    assert chat.service_type is ServiceType.CHAT
    assert reason.service_type is ServiceType.REASON
    assert recommendation.service_type is ServiceType.RECOMMENDATION
    assert recommendation.model == "gpt-5.6-terra"


def test_factory_builds_ollama_services_without_openai_key() -> None:
    factory = LLMServiceFactory(Settings(llm_provider="ollama"))

    chat = factory.create(ServiceType.CHAT)
    reason = factory.create(ServiceType.REASON)
    recommendation = factory.create(ServiceType.RECOMMENDATION)

    assert chat.provider == "ollama"
    assert chat.model == "qwen3:4b"
    assert chat._think is True
    assert reason.service_type is ServiceType.REASON
    assert recommendation.service_type is ServiceType.RECOMMENDATION


def test_auto_provider_uses_ollama_without_openai_key() -> None:
    factory = LLMServiceFactory(
        Settings(
            _env_file=None,
            llm_provider="auto",
            openai_api_key=None,
        )
    )

    service = factory.create(ServiceType.CHAT)

    assert factory.resolve_provider() == "ollama"
    assert service.provider == "ollama"


def test_auto_provider_uses_ollama_for_placeholder_key() -> None:
    factory = LLMServiceFactory(
        Settings(
            _env_file=None,
            llm_provider="auto",
            openai_api_key="replace_with_your_own_secret_key",
        )
    )

    assert factory.resolve_provider() == "ollama"


def test_auto_provider_uses_openai_when_key_exists() -> None:
    factory = LLMServiceFactory(
        Settings(
            _env_file=None,
            llm_provider="auto",
            openai_api_key="test-key",
        )
    )

    service = factory.create(ServiceType.REASON)

    assert factory.resolve_provider() == "openai"
    assert service.provider == "openai"


def test_factory_builds_openai_vision_independently_of_default_provider() -> None:
    factory = LLMServiceFactory(
        Settings(
            _env_file=None,
            llm_provider="ollama",
            openai_api_key="test-key",
            openai_vision_model="gpt-test-vision",
        )
    )

    service = factory.create_vision_service()

    assert service.provider == "openai"
    assert service.model == "gpt-test-vision"
    assert service.detail == "high"
