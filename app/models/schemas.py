from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Message(BaseModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be blank")
        return value


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=100)


class QueryRoute(str, Enum):
    """Supported destinations for the first-stage query router."""

    GENERAL_SEARCH = "general_search"
    ADDITIONAL_SEARCH = "additional_search"
    PRODUCT_SEARCH = "product_search"
    POLICY_SEARCH = "policy_search"
    ANALYTICS_SEARCH = "analytics_search"
    GRAPH_RAG_SEARCH = "graph_rag_search"


class AnalyticsIntent(str, Enum):
    """Allowlisted business questions supported by the analytics warehouse."""

    TOP_PRODUCTS_BY_REVENUE = "top_products_by_revenue"
    MONTHLY_SALES_TREND = "monthly_sales_trend"
    SUPPLIER_PERFORMANCE = "supplier_performance"
    CATEGORY_PERFORMANCE = "category_performance"


class AnalyticsQueryPlan(BaseModel):
    """Structured, non-SQL plan produced by the LLM analytics planner."""

    model_config = ConfigDict(extra="forbid")

    intent: AnalyticsIntent
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    limit: int = Field(default=10, ge=1, le=20)
    reason: str = Field(min_length=1, max_length=500)

    @field_validator("end_date")
    @classmethod
    def end_date_must_not_precede_start_date(
        cls,
        value: Optional[date],
        info,
    ) -> Optional[date]:
        start_date = info.data.get("start_date")
        if value is not None and start_date is not None and value < start_date:
            raise ValueError("end_date must be on or after start_date")
        return value


class AnalyticsPlanningDecision(BaseModel):
    """Do not force an unsupported question into the closest SQL template."""

    model_config = ConfigDict(extra="forbid")
    supported: bool
    plan: AnalyticsQueryPlan | None

    @model_validator(mode="after")
    def consistent_support(self):
        if self.supported != (self.plan is not None):
            raise ValueError("Supported decisions require a plan; unsupported decisions forbid one")
        return self


class ClassificationRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class QueryClassification(BaseModel):
    # Keep the enum reference bare for OpenAI structured-output compatibility.
    # Route documentation lives on QueryRoute and in the classifier prompt.
    route: QueryRoute
    reason: str = Field(
        min_length=1,
        max_length=500,
        description="A concise, user-facing reason for the selected route.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the route selection, from 0.0 to 1.0.",
    )
    resolved_query: str | None = Field(default=None, min_length=1, max_length=10000,
        description="Standalone question resolved only from explicit user conversation context; null if unnecessary or ambiguous.")


class AssistantRequest(ClassificationRequest):
    """A stateful user query sent through the LangGraph assistant."""

    user_id: str = Field(min_length=1, max_length=255)
    conversation_id: Optional[UUID] = None

    @field_validator("user_id")
    @classmethod
    def assistant_user_id_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("user_id must not be blank")
        return value


class ProductMatch(BaseModel):
    product_id: int
    product_name: str
    category: str
    supplier: str
    quantity_per_unit: str
    unit_price: float
    units_in_stock: int
    units_on_order: int
    discontinued: bool


class ConversationChatRequest(ChatRequest):
    user_id: str = Field(min_length=1, max_length=255)
    conversation_id: Optional[UUID] = None

    @field_validator("user_id")
    @classmethod
    def user_id_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("user_id must not be blank")
        return value

    @field_validator("messages")
    @classmethod
    def last_message_must_be_from_user(cls, value: list[Message]) -> list[Message]:
        if value[-1].role != "user":
            raise ValueError("the final message in a chat turn must have role 'user'")
        return value

    @property
    def current_user_message(self) -> Message:
        return self.messages[-1]


class ConversationCreateRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=255)
    title: Optional[str] = Field(default=None, max_length=200)

    @field_validator("user_id")
    @classmethod
    def user_id_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("user_id must not be blank")
        return value

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None


class ConversationUpdateRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=255)
    title: str = Field(min_length=1, max_length=200)

    @field_validator("user_id", "title")
    @classmethod
    def fields_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime


class ConversationMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    conversation_id: UUID
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
