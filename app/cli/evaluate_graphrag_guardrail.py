"""Opt-in, real-model scope/backend evaluation. No retrieval or cloud indexing."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from app.core.config import Settings
from app.services.factory import LLMServiceFactory


ROOT = Path(__file__).resolve().parents[2]


async def run(args: argparse.Namespace) -> int:
    # Provider selection is explicit. Local runs never fall back to a paid API.
    options = {
        "llm_provider": args.provider,
        "ollama_timeout_seconds": args.timeout,
        "openai_timeout_seconds": args.timeout,
        "graphrag_guardrail_timeout_seconds": args.timeout,
    }
    if args.provider == "ollama":
        options.update(_env_file=None, ollama_chat_model=args.model or "qwen3:4b")
    elif args.model:
        options["openai_chat_model"] = args.model
    settings = Settings(**options)
    model = settings.openai_chat_model if args.provider == "openai" else settings.ollama_chat_model
    service = LLMServiceFactory(settings).create_graphrag_guardrail()
    chain = service._chain
    captured = {}

    class ObservedChain:
        async def ainvoke(self, values):
            try:
                result = await chain.ainvoke(values)
            except Exception as exc:
                # Never log exception messages, request headers, or credentials.
                captured["error"] = {"type": type(exc).__name__,
                                     "status_code": getattr(exc, "status_code", None)}
                raise
            captured["assessment"] = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
            return result

    service._chain = ObservedChain()
    cases = json.loads((ROOT / "tests/fixtures/graphrag_scope_cases.json").read_text())
    if args.case:
        cases = [case for case in cases if case["id"] in args.case]
        if len(cases) != len(set(args.case)):
            raise ValueError("Unknown case ID")
    results = []
    for case in cases:
        captured.clear()
        start = perf_counter()
        decision = await service.evaluate(case["query"])
        passed = (
            decision.action in case["actions"]
            and ("backend" not in case or decision.backend == case["backend"])
            and ("reason_code" not in case or decision.reason_code == case["reason_code"])
            and not decision.retrieval_ready
        )
        row = {
            "id": case["id"], "passed": passed,
            "expected_actions": case["actions"], "expected_backend": case.get("backend"),
            "decision": decision.model_dump(mode="json"),
            "elapsed_seconds": round(perf_counter() - start, 3),
        }
        results.append(row)
        if args.include_assessments:
            row["model_assessment"] = captured.get("assessment")
            row["provider_error"] = captured.get("error")
        print(json.dumps(row), flush=True)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider, "model": model,
        "kind": "live_model_scope_and_backend_evaluation",
        "retrieval_executed": False,
        "passed": sum(row["passed"] for row in results), "total": len(results),
        "results": results,
    }
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Live evaluation: {report['passed']}/{report['total']} passed", flush=True)
    return 0 if report["passed"] == report["total"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--model", help="Defaults to qwen3:4b for Ollama or OPENAI_CHAT_MODEL for OpenAI")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--case", action="append")
    parser.add_argument("--output")
    parser.add_argument("--include-assessments", action="store_true")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
