from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from app.models.schemas import QueryClassification, QueryRoute
from app.services.assistant_service import AssistantGraphService
from app.services.knowledge_service import RetrievedKnowledge
from app.services.product_catalog import ProductCatalog, ProductRecord


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeClassifier:
    def __init__(self, route: QueryRoute) -> None:
        self.route = route

    async def classify(self, _: str, *, history=None) -> QueryClassification:
        del history
        return QueryClassification(
            route=self.route,
            reason=f"Selected {self.route.value} for the test.",
            confidence=0.99,
        )


class FakeVisionService:
    async def stream(self, *, question: str, image_bytes: bytes, mime_type: str):
        assert question == "What number is visible?"
        assert image_bytes == b"image-bytes"
        assert mime_type == "image/png"
        yield "The number "
        yield "is 615."


class FakeKnowledgeRetriever:
    async def retrieve(self, query: str) -> list[RetrievedKnowledge]:
        assert query == "Can I return this item?"
        return [
            RetrievedKnowledge(
                content="Eligible products can be returned within 30 days.",
                title="Return Policy",
                source_path="Knowledge Base/Return Policy.pdf#page=1",
                source_file="Knowledge Base/Return Policy.pdf",
                source_type="pdf",
                category="Policy",
                page=1,
                score=0.91,
                chunk_index=0,
            )
        ]


def test_product_catalog_finds_and_enriches_matching_rows() -> None:
    catalog = ProductCatalog(PROJECT_ROOT / "Business_data")

    matches = catalog.search("Show me Philips Hue smart locks and inventory")

    assert matches
    assert matches[0].product_name == "Philips Hue Smart Lock Max"
    assert matches[0].category == "Smart Lock"
    assert matches[0].supplier == "Philips Hue"
    assert matches[0].units_in_stock == 615


def test_product_record_is_immutable() -> None:
    record = ProductRecord(
        product_id=1,
        product_name="Example",
        category="Category",
        supplier="Supplier",
        quantity_per_unit="1 unit",
        unit_price=1.0,
        units_in_stock=1,
        units_on_order=0,
        discontinued=False,
    )

    with pytest.raises(FrozenInstanceError):
        record.product_name = "Changed"  # type: ignore[misc]


def test_product_record_rejects_positional_arguments() -> None:
    with pytest.raises(TypeError):
        ProductRecord(  # type: ignore[misc]
            1,
            "Example",
            "Category",
            "Supplier",
            "1 unit",
            1.0,
            1,
            0,
            False,
        )


def test_general_graph_streams_model_output() -> None:
    fake_module = pytest.importorskip("langchain_core.language_models.fake_chat_models")

    async def scenario() -> None:
        model = fake_module.FakeListChatModel(responses=["A concise general answer."])
        service = AssistantGraphService.from_model(
            classifier=FakeClassifier(QueryRoute.GENERAL_SEARCH),
            model_client=model,
            product_catalog=ProductCatalog(PROJECT_ROOT / "Business_data"),
            provider="fake",
            model="fake-model",
        )

        events = [event async for event in service.stream("What can you do?")]

        assert events[0][0] == "route"
        assert events[0][1]["route"] == "general_search"
        assert "".join(
            payload["content"] for name, payload in events if name == "delta"
        ) == "A concise general answer."

    asyncio.run(scenario())


def test_product_graph_emits_grounding_records() -> None:
    fake_module = pytest.importorskip("langchain_core.language_models.fake_chat_models")

    async def scenario() -> None:
        model = fake_module.FakeListChatModel(responses=["A grounded product answer."])
        service = AssistantGraphService.from_model(
            classifier=FakeClassifier(QueryRoute.PRODUCT_SEARCH),
            model_client=model,
            product_catalog=ProductCatalog(PROJECT_ROOT / "Business_data"),
            provider="fake",
            model="fake-model",
        )

        events = [
            event
            async for event in service.stream(
                "Show me Philips Hue smart locks and inventory."
            )
        ]

        source_event = next(payload for name, payload in events if name == "sources")
        assert source_event["sources"] == ["Business_data/Products.csv"]
        assert source_event["products"][0]["product_name"] == (
            "Philips Hue Smart Lock Max"
        )
        assert "".join(
            payload["content"] for name, payload in events if name == "delta"
        ) == "A grounded product answer."

    asyncio.run(scenario())


def test_return_graph_retrieves_knowledge_and_emits_citations() -> None:
    fake_module = pytest.importorskip("langchain_core.language_models.fake_chat_models")

    async def scenario() -> None:
        model = fake_module.FakeListChatModel(
            responses=["Eligible products can be returned within 30 days [1]."]
        )
        service = AssistantGraphService.from_model(
            classifier=FakeClassifier(QueryRoute.RETURN_SEARCH),
            model_client=model,
            product_catalog=ProductCatalog(PROJECT_ROOT / "Business_data"),
            knowledge_retriever=FakeKnowledgeRetriever(),  # type: ignore[arg-type]
            provider="fake",
            model="fake-model",
        )

        events = [event async for event in service.stream("Can I return this item?")]
        answer = "".join(
            payload["content"] for name, payload in events if name == "delta"
        )

        assert events[0][1]["route"] == "return_search"
        source_event = next(payload for name, payload in events if name == "sources")
        assert source_event["documents"][0]["title"] == "Return Policy"
        assert source_event["documents"][0]["page"] == 1
        assert answer == "Eligible products can be returned within 30 days [1]."

    asyncio.run(scenario())


def test_vision_is_a_streaming_langgraph_branch() -> None:
    fake_module = pytest.importorskip("langchain_core.language_models.fake_chat_models")

    async def scenario() -> None:
        model = fake_module.FakeListChatModel(responses=[])
        service = AssistantGraphService.from_model(
            classifier=FakeClassifier(QueryRoute.GENERAL_SEARCH),
            model_client=model,
            product_catalog=ProductCatalog(PROJECT_ROOT / "Business_data"),
            vision_service=FakeVisionService(),
            provider="fake",
            model="fake-model",
        )

        events = [
            event
            async for event in service.stream(
                "What number is visible?",
                image_bytes=b"image-bytes",
                image_mime_type="image/png",
            )
        ]

        assert events[0][0] == "route"
        assert events[0][1]["route"] == "vision_analysis"
        assert "".join(
            payload["content"] for name, payload in events if name == "delta"
        ) == "The number is 615."

    asyncio.run(scenario())
