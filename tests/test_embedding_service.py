import asyncio
from types import SimpleNamespace

import pytest

from app.services.embedding_service import OpenAIEmbeddingService
from app.services.errors import EmbeddingServiceError


class FakeEmbeddingsResource:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        dimensions = int(kwargs["dimensions"])
        inputs = kwargs["input"]
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(index)] * dimensions)
                for index, _ in reversed(list(enumerate(inputs)))
            ]
        )


def test_openai_embedding_adapter_batches_and_restores_input_order() -> None:
    async def scenario() -> None:
        resource = FakeEmbeddingsResource()
        service = OpenAIEmbeddingService(
            client=SimpleNamespace(embeddings=resource),
            model="test-embedding-model",
            dimensions=3,
            batch_size=2,
        )

        vectors = await service.embed_documents(["one", "two", "three"])

        assert vectors == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
        assert [call["input"] for call in resource.calls] == [
            ["one", "two"],
            ["three"],
        ]
        assert all(call["dimensions"] == 3 for call in resource.calls)

    asyncio.run(scenario())


def test_embedding_adapter_rejects_wrong_dimensions() -> None:
    service = OpenAIEmbeddingService(
        client=SimpleNamespace(),
        model="test-embedding-model",
        dimensions=3,
    )

    with pytest.raises(EmbeddingServiceError, match="expected 3"):
        service._validate_vectors([[1.0, 2.0]], expected_count=1)
