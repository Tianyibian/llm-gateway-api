from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from app.core.config import Settings
from app.models.schemas import Message
from app.services.base import ServiceType
from app.services.factory import LLMServiceFactory
from app.services.langchain_service import LangChainChatService


@dataclass
class FakeChunk:
    text: str = ""
    content: Any = ""


class FakeChatModel:
    def __init__(self, chunks: list[FakeChunk] | None = None) -> None:
        self.chunks = chunks or []
        self.inputs: list[list[tuple[str, str]]] = []

    async def astream(self, messages: list[tuple[str, str]]):
        self.inputs.append(messages)
        for chunk in self.chunks:
            yield chunk


def test_langchain_adapter_maps_roles_and_streams_only_text() -> None:
    async def scenario() -> None:
        model = FakeChatModel(
            [
                FakeChunk(content=[{"type": "reasoning", "text": "hidden"}]),
                FakeChunk(text="Hello"),
                FakeChunk(content=[{"type": "text", "text": " world"}]),
            ]
        )
        service = LangChainChatService(
            model_client=model,
            model="test-model",
            service_type=ServiceType.REASON,
            provider="ollama",
            system_instruction="Answer clearly.",
        )

        output = [
            delta
            async for delta in service.stream(
                [
                    Message(role="developer", content="Be concise."),
                    Message(role="user", content="Hi"),
                    Message(role="assistant", content="Earlier answer"),
                ]
            )
        ]

        assert output == ["Hello", " world"]
        assert model.inputs == [
            [
                ("system", "Answer clearly."),
                ("system", "Be concise."),
                ("human", "Hi"),
                ("ai", "Earlier answer"),
            ]
        ]

    asyncio.run(scenario())


def test_factory_can_select_langchain_without_importing_provider_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeChatModel()
    factory = LLMServiceFactory(
        Settings(
            _env_file=None,
            llm_provider="ollama",
            llm_orchestrator="langchain",
        )
    )
    monkeypatch.setattr(factory, "_build_langchain_model", lambda **_: model)

    service = factory.create(ServiceType.RECOMMENDATION)

    assert isinstance(service, LangChainChatService)
    assert service.provider == "ollama"
    assert service.model == "qwen3:4b"
    assert service.service_type is ServiceType.RECOMMENDATION
