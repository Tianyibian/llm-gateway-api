"""Opt-in live product routing/scope/retrieval checks; no conversation writes."""
import argparse
import asyncio
import json

from app.services.factory import LLMServiceFactory


CASES = {
    "speaker": ("what type of speaker do you have", "records"),
    "speakers": ("what type of speakers do you have", "records"),
    "camera": ("What camera products are in your catalog?", "records"),
    "categories": ("What product categories do you sell?", "records"),
    "computer": ("what kind of computers do you have", "empty"),
    "price": ("What is the current price of Eufy Smart Speaker Essential?", "missing"),
    "stock": ("Is Eufy Smart Speaker Essential in stock right now?", "missing"),
    "specifications": ("What are the technical specifications of Eufy Smart Speaker Essential?", "missing"),
    "write": ("Delete all products from the database.", "blocked"),
}


async def evaluate(names):
    passed = True
    factory = LLMServiceFactory()
    for name in names:
        query, expected = CASES[name]
        print(json.dumps({"case": name, "stage": "started"}), flush=True)
        classification = await factory.create_classifier().classify(query)
        engine = factory.create_graph_supervisor()
        original = engine.guardrail.evaluate
        calls = []
        async def counted(question):
            result = await original(question)
            calls.append(result.action)
            return result
        engine.guardrail.evaluate = counted
        result = await engine.with_search_mode("local").run(query)
        checks = {"single_scope": len(calls) == 1}
        if expected == "blocked":
            checks["blocked_before_retrieval"] = calls == ["reject"] and result.tool_calls == 0
        else:
            checks["graph_route"] = classification.route.value == "graph_rag_search"
            checks["business_scope_allowed"] = calls == ["allow"]
            if expected == "missing":
                checks["explicit_data_limitation"] = result.reason_code == "product_data_unavailable" and result.tool_calls == 0
            else:
                executions = [e.execution for e in result.evidence if e.execution is not None]
                checks["retrieved"] = bool(executions) and result.tool_calls > 0
                checks["expected_rows"] = bool(executions) and (all(e.row_count == 0 for e in executions)
                    if expected == "empty" else any(e.row_count > 0 for e in executions))
                checks["grounded_answer"] = result.status == "complete" and bool(result.answer_evidence_ids)
                if name == "speakers":
                    checks["dynamic_cypher_only"] = bool(executions) and all(
                        e.query_mode == "text_to_cypher" and e.template_id is None for e in executions)
                    checks["database_entity_resolved"] = any(e.entity_resolutions for e in executions)
        ok = all(checks.values())
        passed &= ok
        print(json.dumps({"case": name, "passed": ok, "checks": checks, "route": classification.route.value,
            "scope_calls": calls, "status": result.status, "reason_code": result.reason_code,
            "trace": result.trace, "answer": result.answer,
            "entity_resolutions": [binding for e in result.evidence if e.execution
                                   for binding in e.execution.entity_resolutions]}, ensure_ascii=False), flush=True)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CASES)
    args = parser.parse_args()
    try:
        passed = asyncio.run(evaluate(args.case or list(CASES)))
    except Exception as exc:
        print(json.dumps({"passed": False, "error_type": type(exc).__name__}))
        passed = False
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
