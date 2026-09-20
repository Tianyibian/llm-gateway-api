import asyncio
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.models.schemas import AnalyticsPlanningDecision
from app.services.analytics_planner import AnalyticsPlanner


def test_supported_plan_keeps_explicit_2026_dates():
    chain = AsyncMock()
    chain.ainvoke.return_value = {"supported":True, "plan":{
        "intent":"monthly_sales_trend", "start_date":"2026-01-01", "end_date":"2026-12-31", "limit":12, "reason":"Calendar year report",
    }}
    plan = asyncio.run(AnalyticsPlanner(chain=chain).plan("Monthly sales in 2026"))
    assert plan.start_date.year == 2026 and plan.limit == 12


def test_unsupported_question_has_no_executable_plan():
    chain = AsyncMock()
    chain.ainvoke.return_value = {"supported":False, "plan":None}
    assert asyncio.run(AnalyticsPlanner(chain=chain).plan("Only supplier Acme's sales")) is None


@pytest.mark.parametrize("data", [
    {"supported":True, "plan":None},
    {"supported":False, "plan":{"intent":"monthly_sales_trend", "reason":"Wrongly supplied plan"}},
    {"supported":True, "plan":{"intent":"monthly_sales_trend", "reason":"Unsafe", "sql":"DROP TABLE PRODUCTS"}},
    {"supported":False, "plan":None, "sql":"SELECT * FROM CUSTOMERS"},
])
def test_plan_contract_rejects_inconsistent_or_extra_fields(data):
    with pytest.raises(ValidationError):
        AnalyticsPlanningDecision.model_validate(data)


def test_unsupported_analytics_does_not_call_database_or_answer_model():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from app.models.schemas import QueryClassification
    from app.services.assistant_service import AssistantGraphService
    from app.services.product_catalog import ProductCatalog
    classifier = AsyncMock()
    classifier.classify.return_value = QueryClassification(route="analytics_search", reason="Report", confidence=1)
    planner = AsyncMock()
    planner.plan.return_value = None
    database = AsyncMock()
    service = AssistantGraphService.from_model(classifier=classifier,
        model_client=FakeListChatModel(responses=["Must not use this answer"]), product_catalog=ProductCatalog("Business_data"),
        analytics_planner=planner, analytics_service=database, provider="fake", model="fake")
    async def run():
        return [item async for item in service.stream("Only supplier Acme's sales")]
    events = asyncio.run(run())
    database.query.assert_not_awaited()
    assert "No database query was executed" in str(events)
    assert "Must not use this answer" not in str(events)
