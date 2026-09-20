from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GraphContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GraphGuardrailRequest(GraphContract):
    query: str = Field(min_length=1, max_length=10_000)

    @field_validator("query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class GraphScope(str, Enum):
    IN_SCOPE = "in_scope"
    MIXED = "mixed"
    OUT_OF_SCOPE = "out_of_scope"
    UNCLEAR = "unclear"


class GraphIntent(str, Enum):
    ENTITY_RELATIONSHIPS = "entity_relationships"
    CORPUS_SUMMARY = "corpus_summary"
    EXPLORATORY = "exploratory"
    GRAPH_ANALYTICS = "graph_analytics"


class GraphBackend(str, Enum):
    MICROSOFT_GRAPHRAG = "microsoft_graphrag"
    NEO4J = "neo4j"


class GraphRisk(str, Enum):
    INSTRUCTION_OVERRIDE = "instruction_override"
    SECRET_EXTRACTION = "secret_extraction"
    UNAUTHORIZED_DATA = "unauthorized_data"
    WRITE_OPERATION = "write_operation"


class GraphRelation(GraphContract):
    source_type: str = Field(min_length=1, max_length=60)
    relation: str = Field(min_length=1, max_length=60)
    target_type: str = Field(min_length=1, max_length=60)


class GraphMention(GraphContract):
    text: str = Field(min_length=1, max_length=200)
    entity_type: str = Field(min_length=1, max_length=60)


class GraphScopeAssessment(GraphContract):
    """Untrusted LLM output. Not a permission to execute a tool."""

    scope: GraphScope
    intent: GraphIntent
    backend: GraphBackend
    confidence: float = Field(ge=0, le=1)
    entity_types: list[str] = Field(max_length=8)
    entity_mentions: list[GraphMention] = Field(max_length=8)
    required_relations: list[GraphRelation] = Field(max_length=8)
    risks: list[GraphRisk] = Field(max_length=4)


class GraphPolicy(GraphContract):
    version: str = Field(min_length=1)
    scope_definition: str = Field(min_length=1)
    entity_types: list[str] = Field(min_length=1)
    relations: list[GraphRelation] = Field(min_length=1)
    backend_capabilities: dict[GraphBackend, str]


class GraphGuardrailDecision(GraphContract):
    action: Literal["allow", "clarify", "reject", "unavailable"]
    reason_code: str
    message: str
    policy_version: str
    intent: GraphIntent | None = None
    backend: GraphBackend | None = None
    eligible_tools: list[Literal["query_relationships", "local_search", "global_search", "drift_search"]] = Field(default_factory=list)
    eligible_search_modes: list[Literal["local", "global", "drift"]] = Field(
        default_factory=list
    )
    entity_mentions: list[GraphMention] = Field(default_factory=list)
    # The scope-only endpoint does not resolve entities or inspect adapters.
    # Readiness is reported separately by /api/graphrag/status.
    retrieval_ready: Literal[False] = False
    entity_resolution: Literal["not_performed"] = "not_performed"
