from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import httpx

from app.services.errors import EmbeddingServiceError


class EmbeddingService(ABC):
    """Provider-neutral asynchronous text embedding interface."""

    def __init__(self, *, provider: str, model: str, dimensions: int) -> None:
        self.provider = provider
        self.model = model
        self.dimensions = dimensions

    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple document chunks in input order."""

    async def embed_query(self, text: str) -> list[float]:
        """Embed one retrieval query."""
        vectors = await self.embed_documents([text])
        return vectors[0]

    def _validate_vectors(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        expected_count: int,
    ) -> list[list[float]]:
        if len(vectors) != expected_count:
            raise EmbeddingServiceError(
                f"{self.provider} returned {len(vectors)} vectors for "
                f"{expected_count} inputs."
            )
        validated: list[list[float]] = []
        for index, vector in enumerate(vectors):
            if len(vector) != self.dimensions:
                raise EmbeddingServiceError(
                    f"{self.provider} embedding {index} has {len(vector)} dimensions; "
                    f"expected {self.dimensions}."
                )
            validated.append([float(value) for value in vector])
        return validated


class OllamaEmbeddingService(EmbeddingService):
    """Generate local embeddings through Ollama's batch embed endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dimensions: int,
        timeout_seconds: float,
        batch_size: int = 32,
    ) -> None:
        super().__init__(provider="ollama", model=model, dimensions=dimensions)
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.batch_size = batch_size

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        normalized = [text.strip() for text in texts]
        if not normalized or any(not text for text in normalized):
            raise ValueError("Embedding inputs must contain non-blank text.")

        vectors: list[list[float]] = []
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for start in range(0, len(normalized), self.batch_size):
                batch = normalized[start : start + self.batch_size]
                try:
                    response = await client.post(
                        f"{self.base_url}/api/embed",
                        json={"model": self.model, "input": batch, "truncate": True},
                    )
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    raise EmbeddingServiceError(
                        "Ollama embedding request failed. Confirm Ollama is running and "
                        f"run 'ollama pull {self.model}'."
                    ) from exc
                payload = response.json()
                batch_vectors = payload.get("embeddings")
                if not isinstance(batch_vectors, list):
                    raise EmbeddingServiceError(
                        "Ollama returned an invalid embedding response."
                    )
                vectors.extend(batch_vectors)

        return self._validate_vectors(vectors, expected_count=len(normalized))


class OpenAIEmbeddingService(EmbeddingService):
    """Generate embeddings through the OpenAI embeddings API."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        dimensions: int,
        batch_size: int = 128,
    ) -> None:
        super().__init__(provider="openai", model=model, dimensions=dimensions)
        self._client = client
        self.batch_size = batch_size

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        normalized = [text.strip() for text in texts]
        if not normalized or any(not text for text in normalized):
            raise ValueError("Embedding inputs must contain non-blank text.")

        vectors: list[list[float]] = []
        for start in range(0, len(normalized), self.batch_size):
            batch = normalized[start : start + self.batch_size]
            response = await self._client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions,
                encoding_format="float",
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)

        return self._validate_vectors(vectors, expected_count=len(normalized))
