from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from app.models.graphrag import GraphContract, GraphMention
from app.models.cypher import CypherExecution


class GraphTool(str, Enum):
    PREDEFINED_CYPHER = "predefined_cypher"
    TEXT_TO_CYPHER = "text_to_cypher"
    # Retained for legacy callers; the application factory registers explicit tools.
    NEO4J = "neo4j_relationships"
    MS_LOCAL = "ms_local_search"
    MS_GLOBAL = "ms_global_search"
    MS_DRIFT = "ms_drift_search"


class GraphTask(GraphContract):
    tool: GraphTool
    question: str = Field(min_length=1, max_length=2000,
        description="A focused business subquestion, not query implementation instructions. Preserve a single focused original question verbatim; do not add fields, search algorithms or output columns.")
    parent_evidence_ids: list[str] = Field(max_length=12)
    entity_mentions: list[GraphMention] = Field(default_factory=list, max_length=12,
        description="Every named entity/category filter in this task, quoted from the task question; use [] for unfiltered queries.")


class GraphPlan(GraphContract):
    action: Literal["retrieve", "finish", "clarify", "data_unavailable"]
    tasks: list[GraphTask] = Field(max_length=3)
    evidence_ids: list[str] = Field(max_length=12)
    missing_product_data: list[Literal["current_price", "live_stock", "specifications", "compatibility"]] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def consistent_action(self):
        if (self.action == "data_unavailable") != bool(self.missing_product_data):
            raise ValueError("Only data_unavailable plans must name missing product capabilities")
        if (self.action == "retrieve") != bool(self.tasks):
            raise ValueError("Only retrieve plans must have tasks")
        if self.action == "finish" and not self.evidence_ids:
            raise ValueError("Finishing requires evidence")
        if self.action != "finish" and self.evidence_ids:
            raise ValueError("Only finishing may select answer evidence")
        return self


class RetrievalEvidence(GraphContract):
    """Adapter-supplied public evidence; IDs assigned by the supervisor separately."""

    source_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=2000)
    entities: list[GraphMention] = Field(max_length=12)
    execution: CypherExecution | None = None


class GraphEvidence(RetrievalEvidence):
    evidence_id: str
    tool: GraphTool
    task_id: str
    parent_evidence_ids: list[str]
    agent_run_id: str | None = None
    agent_role: str | None = None


class SupervisorLimits(GraphContract):
    max_rounds: int = Field(default=3, ge=1, le=6)
    max_tool_calls: int = Field(default=6, ge=1, le=12)
    max_parallel: int = Field(default=3, ge=1, le=3)
    max_evidence_per_call: int = Field(default=3, ge=1, le=5)
    timeout_seconds: float = Field(default=120, gt=0, le=600)
    call_timeout_seconds: float = Field(default=30, gt=0, le=120)


class SupervisorResult(GraphContract):
    status: Literal["complete", "partial", "clarify", "rejected", "unavailable"]
    reason_code: str
    answer: str
    evidence: list[GraphEvidence] = Field(default_factory=list)
    answer_evidence_ids: list[str] = Field(default_factory=list)
    rounds: int = 0
    tool_calls: int = 0
    trace: list[dict] = Field(default_factory=list)
    answer_generation: dict | None = None
    agent_runs: list[dict] = Field(default_factory=list)
