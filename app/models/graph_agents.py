"""Separate contracts for business delegation and tool selection."""
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from app.models.graphrag import GraphContract


class AgentRole(str, Enum):
    CATALOG = "catalog_agent"
    SALES = "sales_agent"
    REVIEWS = "reviews_agent"


class AgentAssignment(GraphContract):
    agent: AgentRole
    question: str = Field(min_length=1, max_length=2000)
    parent_evidence_ids: list[str] = Field(max_length=12)


class DelegationPlan(GraphContract):
    action: Literal["delegate", "finish", "clarify"]
    tasks: list[AgentAssignment] = Field(max_length=3)
    evidence_ids: list[str] = Field(max_length=12)

    @model_validator(mode="after")
    def consistent_action(self):
        if (self.action == "delegate") != bool(self.tasks):
            raise ValueError("Only delegation contains assignments")
        if (self.action == "finish") != bool(self.evidence_ids):
            raise ValueError("Only finishing selects evidence")
        return self
