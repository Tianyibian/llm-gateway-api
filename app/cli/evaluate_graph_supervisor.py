"""Real LLM supervisor evaluation with explicitly SYNTHETIC retrieval fixtures.

This does not connect to Neo4j or Microsoft GraphRAG and sends no business files.
"""
import argparse
import asyncio
import json
from pathlib import Path

from app.core.config import Settings
from app.models.graph_supervisor import GraphTool, RetrievalEvidence
from app.models.graphrag import GraphMention
from app.services.factory import LLMServiceFactory


QUESTION = (
    "Who supplies Acme Door Sensor, what other products does that supplier supply, "
    "and what recurring support themes are documented for that supplier's products?"
)


class SyntheticGraphTool:
    """Test fixture ONLY; never registered in the application factory."""

    def __init__(self, kind):
        self.kind = kind

    async def retrieve(self, question, *, limit):
        if self.kind is GraphTool.NEO4J:
            text = ("Acme Supply supplies Acme Door Sensor and Acme Window Sensor. "
                    "Both products belong to the Sensor category.")
            entities = [("Acme Supply", "Supplier"), ("Acme Door Sensor", "Product"),
                        ("Acme Window Sensor", "Product"), ("Sensor", "Category")]
        else:
            text = ("Synthetic support documents for Acme Supply products describe "
                    "recurring pairing and battery replacement questions for Acme Door Sensor "
                    "and Acme Window Sensor. These excerpts contain no return eligibility claims.")
            entities = [("Acme Supply", "Supplier"), ("Acme Door Sensor", "Product"),
                        ("Acme Window Sensor", "Product")]
        return [RetrievalEvidence(source_id=f"synthetic:{self.kind.value}", text=text,
                                  entities=[GraphMention(text=name, entity_type=kind) for name, kind in entities])]


async def run(args):
    options = {"llm_provider": args.provider}
    if args.provider == "ollama":
        options.update(_env_file=None, ollama_chat_model=args.model or "qwen3:4b")
    elif args.model:
        options["openai_chat_model"] = args.model
    settings = Settings(**options)
    supervisor = LLMServiceFactory(settings).create_graph_supervisor(
        tools={kind: SyntheticGraphTool(kind) for kind in GraphTool},
    )
    chain = supervisor._chain
    plans = []

    class ObservePlans:
        async def ainvoke(self, values):
            result = await chain.ainvoke(values)
            plans.append({"input_evidence_count": len(json.loads(values["context"])["evidence"]),
                          "plan": result.model_dump(mode="json")})
            return result

    supervisor._chain = ObservePlans()
    result = await supervisor.run(QUESTION)
    selected_tools = {row["tool"] for row in result.trace}
    checks = {
        "completed": result.status == "complete",
        "multiple_retrieval_rounds": result.rounds >= 2,
        "neo4j_selected": GraphTool.NEO4J.value in selected_tools,
        "microsoft_graphrag_selected": bool(selected_tools & {tool.value for tool in GraphTool if tool is not GraphTool.NEO4J}),
        "replanned_with_returned_evidence": any(plan["input_evidence_count"] for plan in plans[1:]),
        "bounded_calls": result.tool_calls <= supervisor.limits.max_tool_calls,
        "cited_excerpts": bool(result.answer_evidence_ids) and all(f"[{key}]" in result.answer for key in result.answer_evidence_ids),
    }
    report = {"provider": args.provider,
              "model": settings.openai_chat_model if args.provider == "openai" else settings.ollama_chat_model,
              "planner_and_guardrail": "real_model", "retrieval": "synthetic_fixtures_not_real_backends",
              "passed": sum(checks.values()), "total": len(checks), "checks": checks,
              "plans": plans, "result": result.model_dump(mode="json")}
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if all(checks.values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--model")
    parser.add_argument("--output")
    raise SystemExit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
