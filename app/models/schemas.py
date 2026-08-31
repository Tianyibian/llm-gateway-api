from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    RETURN_SEARCH = "return_search"
    PRODUCT_SEARCH = "product_search"


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
    route: QueryRoute = Field(description="The search route selected for the query.")
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
