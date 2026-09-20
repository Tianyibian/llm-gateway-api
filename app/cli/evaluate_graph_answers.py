"""Opt-in live SSE/MapReduce checks. Uses model tokens and persists test conversations."""
from __future__ import annotations

import argparse
import asyncio
from datetime import date
from decimal import Decimal
import json
import re
from uuid import uuid4

import httpx

from app.core.config import Settings
from app.models.schemas import AnalyticsQueryPlan
from app.services.csv_analytics import CsvAnalyticsBaseline


async def evaluate(base_url: str, include_microsoft: bool):
    user = f"mapreduce-live-{uuid4().hex[:8]}"
    queries = [
        "Using Neo4j, show monthly sales revenue for 2025. Answer in English.",
        "Using Neo4j, count products by supplier, top 5. Answer in English.",
    ]
    if include_microsoft:
        queries.append("Summarize recurring support themes in the sampled Eufy Smart Speaker Essential reviews. Answer in English.")
    async with httpx.AsyncClient(base_url=base_url, timeout=450) as client:
        for query in queries:
            response = await client.post("/api/assistant", json={"user_id": user, "query": query})
            response.raise_for_status()
            events = []
            for block in response.text.split("\n\n"):
                lines = block.splitlines()
                name = next((line[7:] for line in lines if line.startswith("event: ")), None)
                data = next((line[6:] for line in lines if line.startswith("data: ")), None)
                if name and data:
                    events.append((name, data if data == "[DONE]" else json.loads(data)))
            answer = "".join(payload["content"] for name, payload in events if name == "delta")
            supervisor = next((payload for name, payload in events if name == "supervisor"), {})
            print(json.dumps({"query": query, "events": [name for name, _ in events],
                              "answer": answer, "generation": supervisor.get("answer_generation"),
                              "reason_code": supervisor.get("reason_code"), "trace": supervisor.get("trace")}, ensure_ascii=False), flush=True)
            assert any(name == "done" for name, _ in events), "Stream did not complete"
            assert not any(name == "error" for name, _ in events), "Stream returned an error"
            assert (supervisor.get("answer_generation") or {}).get("status") == "complete", "Answer generation failed"
            assert any(name == "answer_generation" for name, _ in events), "Missing progress events"
            assert "[E" in answer and "Retrieved evidence:" not in answer, "Missing final synthesis"
            evidence = supervisor["evidence"]
            if "monthly" in query:
                assert evidence[0]["execution"]["query_mode"] == "template"
                plan = AnalyticsQueryPlan(intent="monthly_sales_trend", start_date=date(2025, 1, 1),
                                          end_date=date(2025, 12, 31), limit=12, reason="Live answer validation")
                rows = CsvAnalyticsBaseline(Settings().business_data_dir).query(plan).rows
                numbers = {Decimal(value.replace(",", "")) for value in re.findall(r"\d[\d,]*(?:\.\d+)?", answer)}
                assert all(Decimal(str(row["revenue"])) in numbers for row in rows), "Missing or incorrect monthly revenue"
                print("PASS: all 12 monthly revenue values match the independent CSV baseline", flush=True)
            elif "count products" in query:
                assert evidence[0]["execution"]["query_mode"] == "text_to_cypher"
                assert "neo4j_explain_read_only" in evidence[0]["execution"]["checks"]
            else:
                assert any(item["tool"].startswith("ms_") for item in evidence), "Microsoft GraphRAG was not used"
            conversation = next(payload["conversation_id"] for name, payload in events if name == "metadata")
            history = await client.get(f"/api/conversations/{conversation}/messages", params={"user_id": user})
            history.raise_for_status()
            assert len(history.json()) == 2 and history.json()[1]["content"] == answer, "Persisted answer differs"
            print(json.dumps({"passed": True, "conversation_id": conversation, "user_id": user,
                              "checks": "SSE progress, cited answer, strategy, persistence"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--include-microsoft", action="store_true")
    args = parser.parse_args()
    asyncio.run(evaluate(args.base_url, args.include_microsoft))
