from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.services.graphrag_guardrail import GraphRAGGuardrail, scope_stop_message
from app.services.graph_supervisor import GraphRAGSupervisor


class GraphBranchState(TypedDict, total=False):
    query: str
    graphrag_search_mode: str
    graph_guardrail_decision: dict
    answer: str
    graph_supervisor_result: dict


def build_graphrag_branch(guardrail: GraphRAGGuardrail, supervisor: GraphRAGSupervisor | None = None):
    """Only approved GraphRAG queries enter the adaptive supervisor."""
    async def assess(state: GraphBranchState) -> GraphBranchState:
        decision = await guardrail.evaluate(state["query"])
        payload = decision.model_dump(mode="json")
        from langgraph.config import get_stream_writer
        get_stream_writer()({"event": "guardrail", "payload": payload})
        return {"graph_guardrail_decision": payload}

    def respond(state: GraphBranchState) -> GraphBranchState:
        decision = state["graph_guardrail_decision"]
        return {"answer": decision["message"] if decision["action"] == "allow"
                else scope_stop_message(decision["action"])}

    async def supervise(state: GraphBranchState) -> GraphBranchState:
        selected = supervisor.with_search_mode(state.get("graphrag_search_mode", "local"))
        result = await selected.run_approved(
            state["query"],
            scope_approved=state["graph_guardrail_decision"]["action"] == "allow",
        )
        return {"graph_supervisor_result": result.model_dump(mode="json"), "answer": result.answer}

    graph = StateGraph(GraphBranchState)
    graph.add_node("assess_graph_scope", assess)
    graph.add_node("graph_scope_response", respond)
    graph.add_edge(START, "assess_graph_scope")
    if supervisor is not None:
        graph.add_node("supervise_graph_retrieval", supervise)
        graph.add_conditional_edges("assess_graph_scope", lambda state:
            "supervise" if state["graph_guardrail_decision"]["action"] == "allow" else "respond",
            {"supervise": "supervise_graph_retrieval", "respond": "graph_scope_response"})
        graph.add_edge("supervise_graph_retrieval", END)
    else:
        graph.add_edge("assess_graph_scope", "graph_scope_response")
    graph.add_edge("graph_scope_response", END)
    return graph.compile(name="graphrag-scope-gate")
