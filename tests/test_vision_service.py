from __future__ import annotations

import asyncio
import base64
from io import BytesIO
from types import SimpleNamespace

from PIL import Image
import pytest

from app.services.errors import InvalidImageError
from app.services.vision_service import OpenAIVisionService


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class FakeEventStream:
    def __init__(self) -> None:
        self._events = iter(
            [
                SimpleNamespace(type="response.created"),
                SimpleNamespace(type="response.output_text.delta", delta="Visible"),
                SimpleNamespace(type="response.output_text.delta", delta=" answer"),
            ]
        )

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeEventStream()


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


def test_vision_service_builds_responses_image_input_and_streams_text() -> None:
    async def scenario() -> None:
        client = FakeClient()
        service = OpenAIVisionService(
            client=client,  # type: ignore[arg-type]
            model="gpt-test",
            detail="high",
        )

        output = [
            delta
            async for delta in service.stream(
                question="What is shown?",
                image_bytes=PNG_BYTES,
                mime_type="image/png",
            )
        ]

        assert output == ["Visible", " answer"]
        request = client.responses.calls[0]
        assert request["model"] == "gpt-test"
        assert request["stream"] is True
        content = request["input"][0]["content"]
        assert content[0] == {"type": "input_text", "text": "What is shown?"}
        assert content[1]["detail"] == "high"
        assert content[1]["image_url"].startswith("data:image/png;base64,")

    asyncio.run(scenario())


def test_image_validation_uses_file_content_not_claimed_mime_type() -> None:
    assert OpenAIVisionService.validate_image(PNG_BYTES) == "image/png"

    with pytest.raises(InvalidImageError, match="not a valid supported image"):
        OpenAIVisionService.validate_image(b"not an image")


def test_image_validation_rejects_animated_gif() -> None:
    output = BytesIO()
    first = Image.new("RGB", (2, 2), "red")
    second = Image.new("RGB", (2, 2), "blue")
    first.save(output, format="GIF", save_all=True, append_images=[second], duration=100)

    with pytest.raises(InvalidImageError, match="Animated GIF"):
        OpenAIVisionService.validate_image(output.getvalue())
