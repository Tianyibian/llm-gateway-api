from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from io import BytesIO
import warnings

from openai import AsyncOpenAI
from PIL import Image, UnidentifiedImageError

from app.services.errors import InvalidImageError


class OpenAIVisionService:
    """Stream image analysis from the OpenAI Responses API."""

    _MIME_BY_FORMAT = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "WEBP": "image/webp",
        "GIF": "image/gif",
    }

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        detail: str,
    ) -> None:
        self._client = client
        self.model = model
        self.detail = detail
        self.provider = "openai"

    @classmethod
    def validate_image(cls, image_bytes: bytes) -> str:
        if not image_bytes:
            raise InvalidImageError("The uploaded image is empty.")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(image_bytes)) as image:
                    image_format = image.format
                    is_animated = bool(getattr(image, "is_animated", False))
                    image.verify()
        except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombWarning) as exc:
            raise InvalidImageError(
                "The upload is not a valid supported image."
            ) from exc

        if image_format not in cls._MIME_BY_FORMAT:
            raise InvalidImageError(
                "Supported image formats are PNG, JPEG, WEBP, and non-animated GIF."
            )
        if image_format == "GIF" and is_animated:
            raise InvalidImageError("Animated GIF files are not supported.")
        return cls._MIME_BY_FORMAT[image_format]

    async def stream(
        self,
        *,
        question: str,
        image_bytes: bytes,
        mime_type: str,
    ) -> AsyncIterator[str]:
        image_base64 = base64.b64encode(image_bytes).decode("ascii")
        image_url = f"data:{mime_type};base64,{image_base64}"
        stream = await self._client.responses.create(
            model=self.model,
            instructions=(
                "Analyze the supplied image and answer the user's question accurately. "
                "Describe uncertainty and do not invent text or details that are not visible. "
                "Do not provide medical diagnoses or claim exact measurements from the image. "
                "Use concise plain text without Markdown formatting."
            ),
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": question},
                        {
                            "type": "input_image",
                            "image_url": image_url,
                            "detail": self.detail,
                        },
                    ],
                }
            ],
            stream=True,
        )

        async for event in stream:
            if event.type == "response.output_text.delta":
                yield event.delta
