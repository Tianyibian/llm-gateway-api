"""Opt-in REAL OpenAI + Microsoft GraphRAG + supervisor + FastAPI test. No mocks."""
import argparse
import asyncio
import json
from pathlib import Path

import httpx

from app.main import app


async def run(output):
    # In-process ASGI transport exercises the real route, dependencies, model,
    # isolated GraphRAG process, and index. No service overrides are installed.
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=360) as client:
        status = await client.get("/api/graphrag/status")
        response = await client.post("/api/graphrag/query", json={
            "query": "Summarize the recurring support themes described in the sampled Eufy Smart Speaker Essential reviews."
        })
    result = response.json()
    checks = {
        "index_ready": status.status_code == 200 and status.json().get("ready") is True,
        "api_success": response.status_code == 200,
        "supervisor_complete": result.get("status") == "complete",
        "real_graph_calls": result.get("tool_calls", 0) > 0,
        "real_evidence_ids": bool(result.get("evidence")) and all(
            e["source_id"].startswith("microsoft_graphrag:") for e in result.get("evidence", [])
        ),
        "known_answer_citations": bool(result.get("answer_evidence_ids")) and all(
            f"[{key}]" in result.get("answer", "") for key in result.get("answer_evidence_ids", [])
        ),
        "no_neo4j_or_fake_tools": all(row["tool"].startswith("ms_") and not row.get("error") for row in result.get("trace", [])),
    }
    report = {"kind": "live_microsoft_graphrag_fastapi_supervisor_no_mocks",
              "passed": sum(checks.values()), "total": len(checks), "checks": checks,
              "index_status": status.json(), "http_status": response.status_code, "result": result}
    if output:
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "result"}, indent=2))
    return 0 if all(checks.values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    raise SystemExit(asyncio.run(run(parser.parse_args().output)))


if __name__ == "__main__":
    main()
