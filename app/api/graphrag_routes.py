from fastapi import APIRouter, Depends, HTTPException, Response

from app.models.graphrag import GraphGuardrailDecision, GraphGuardrailRequest
from app.services.errors import LLMConfigurationError
from app.services.factory import LLMServiceFactory
from app.services.graphrag_guardrail import GraphRAGGuardrail
from app.services.graph_supervisor import GraphRAGSupervisor
from app.models.graph_supervisor import SupervisorResult

router = APIRouter(prefix="/api/graphrag", tags=["graphrag"])


@router.get("/neo4j/status", summary="Inspect the Neo4j business snapshot without an LLM call")
async def neo4j_status():
    from app.services.neo4j_service import Neo4jExecutor
    try:
        _, snapshot = await Neo4jExecutor(LLMServiceFactory().settings).run()
        return {"ready": True, "backend": "neo4j", "query_modes": ["template", "text_to_cypher"], "snapshot": snapshot}
    except Exception:
        return {"ready": False, "backend": "neo4j", "query_modes": []}


@router.get("/status", summary="Inspect verified Microsoft GraphRAG index readiness")
def graph_index_status():
    factory = LLMServiceFactory()
    if not factory.settings.microsoft_graphrag_enabled:
        return {"ready": False, "backend": "microsoft_graphrag", "available_modes": []}
    try:
        return factory.create_microsoft_graphrag_client().status()
    except LLMConfigurationError:
        return {"ready": False, "backend": "microsoft_graphrag", "available_modes": []}


def get_graph_supervisor() -> GraphRAGSupervisor:
    try:
        return LLMServiceFactory().create_graph_supervisor()
    except LLMConfigurationError as exc:
        raise HTTPException(503, "GraphRAG supervisor is not configured.") from exc


@router.post("/query", response_model=SupervisorResult,
             summary="Run the guarded, bounded GraphRAG supervisor")
async def run_graph_query(
    request: GraphGuardrailRequest, response: Response,
    service: GraphRAGSupervisor = Depends(get_graph_supervisor),
) -> SupervisorResult:
    result = await service.run(request.query)
    if result.status == "unavailable":
        response.status_code = 503
    return result


def get_graph_guardrail() -> GraphRAGGuardrail:
    try:
        return LLMServiceFactory().create_graphrag_guardrail(scope_only=True)
    except LLMConfigurationError as exc:
        raise HTTPException(503, "GraphRAG scope model is not configured.") from exc


@router.post("/guardrail", response_model=GraphGuardrailDecision,
             summary="Evaluate GraphRAG scope without retrieving or saving a turn")
async def evaluate_graph_scope(
    request: GraphGuardrailRequest,
    response: Response,
    service: GraphRAGGuardrail = Depends(get_graph_guardrail),
) -> GraphGuardrailDecision:
    decision = await service.evaluate(request.query)
    if decision.action == "unavailable":
        response.status_code = 503
    return decision
