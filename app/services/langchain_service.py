from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from app.models.schemas import Message
from app.services.base import LLMService, ServiceType


class LangChainChatService(LLMService):
    """Adapter from LangChain's chat-model interface to the gateway contract."""

    def __init__(
        self,
        *,
        model_client: Any,
        model: str,
        service_type: ServiceType,
        provider: str,
        system_instruction: str | None = None,
    ) -> None:
        super().__init__(model=model, service_type=service_type, provider=provider)
        self._model_client = model_client
        self._system_instruction = system_instruction

    def _messages(self, messages: Sequence[Message]) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        if self._system_instruction:
            result.append(("system", self._system_instruction))

        role_map = {
            "system": "system",
            "developer": "system",
            "user": "human",
            "assistant": "ai",
        }
        result.extend((role_map[message.role], message.content) for message in messages)
        return result

    @staticmethod
    def _text_from_chunk(chunk: Any) -> str:
        """Return final-answer text while ignoring reasoning/tool content blocks."""
        text = getattr(chunk, "text", None)
        if isinstance(text, str) and text:
            return text

        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""

        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {
                "text",
                "output_text",
            }:
                value = block.get("text", "")
                if isinstance(value, str):
                    parts.append(value)
        return "".join(parts)

    async def stream(self, messages: Sequence[Message]) -> AsyncIterator[str]:
        async for chunk in self._model_client.astream(self._messages(messages)):
            if text := self._text_from_chunk(chunk):
                yield text
