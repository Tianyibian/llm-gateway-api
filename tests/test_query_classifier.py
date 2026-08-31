from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from app.models.schemas import QueryClassification, QueryRoute
from app.services.query_classifier import QueryClassifier


class FakeClassificationChain:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.inputs: list[dict[str, Any]] = []

    async def ainvoke(self, values: dict[str, Any]) -> Any:
        self.inputs.append(values)
        return self.result


def test_classifier_returns_typed_structured_output() -> None:
    async def scenario() -> None:
        chain = FakeClassificationChain(
            {
                "route": "product_search",
                "reason": "The query asks for product specifications.",
                "confidence": 0.95,
            }
        )
        classifier = QueryClassifier(
            chain=chain,
            provider="ollama",
            model="qwen3:4b",
        )

        result = await classifier.classify("Does this camera support Wi-Fi 6?")

        assert result == QueryClassification(
            route=QueryRoute.PRODUCT_SEARCH,
            reason="The query asks for product specifications.",
            confidence=0.95,
        )
        assert chain.inputs == [
            {
                "query": "Does this camera support Wi-Fi 6?",
                "history": [],
            }
        ]

    asyncio.run(scenario())


def test_classifier_rejects_an_unknown_model_route() -> None:
    async def scenario() -> None:
        classifier = QueryClassifier(
            chain=FakeClassificationChain(
                {
                    "route": "order_search",
                    "reason": "Unsupported route.",
                    "confidence": 0.8,
                }
            ),
            provider="fake",
            model="fake-model",
        )

        with pytest.raises(ValidationError):
            await classifier.classify("Where is my order?")

    asyncio.run(scenario())


def test_classifier_receives_conversation_history_for_follow_up_routing() -> None:
    async def scenario() -> None:
        chain = FakeClassificationChain(
            {
                "route": "product_search",
                "reason": "The follow-up refers to the previously discussed product.",
                "confidence": 0.9,
            }
        )
        classifier = QueryClassifier(chain=chain, provider="fake", model="fake-model")
        history = [
            ("user", "Tell me about Chai."),
            ("assistant", "Chai is a beverage product."),
        ]

        await classifier.classify("What about its stock?", history=history)

        assert chain.inputs == [
            {"query": "What about its stock?", "history": history}
        ]

    asyncio.run(scenario())


def test_classifier_prompt_defines_every_supported_route() -> None:
    for route in QueryRoute:
        assert route.value in QueryClassifier.SYSTEM_PROMPT
