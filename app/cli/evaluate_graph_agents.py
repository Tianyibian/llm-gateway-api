"""Opt-in live three-specialist acceptance test; incurs model cost and saves a test turn."""
import argparse
import asyncio
import json
from uuid import uuid4

import httpx


QUESTION = ("For Eufy Smart Speaker Essential, identify its supplier, report its total discounted "
            "sales revenue in 2025, and summarize recurring support themes in its sampled reviews. Answer in English.")


async def evaluate(base_url):
    user = "agents-live-" + uuid4().hex[:8]
    events = []
    async with httpx.AsyncClient(base_url=base_url, timeout=650) as client:
        async with client.stream("POST", "/api/assistant", json={"user_id": user, "query": QUESTION}) as response:
            response.raise_for_status()
            event = None
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: "):
                    data = line[6:]
                    payload = data if data == "[DONE]" else json.loads(data)
                    events.append((event, payload))
                    if event in {"route", "agent", "answer_generation", "error"}:
                        print(json.dumps({"event": event, "payload": payload}), flush=True)
        result = next((payload for name, payload in events if name == "supervisor"), {})
        answer = "".join(payload["content"] for name, payload in events if name == "delta")
        print(json.dumps({"status": result.get("status"), "reason_code": result.get("reason_code"),
                          "agent_runs": result.get("agent_runs"), "trace": result.get("trace"),
                          "answer_generation": result.get("answer_generation"), "answer": answer}), flush=True)
        assert result.get("status") == "complete", "Supervisor did not complete all goals"
        runs = result["agent_runs"]
        assert {run["agent"] for run in runs} == {"catalog_agent", "sales_agent", "reviews_agent"}
        assert all(run["status"] == "complete" and 1 <= run["tool_calls"] <= 2 for run in runs)
        assert 3 <= result["tool_calls"] <= 6
        for step in result["trace"]:
            expected = step["tool"].startswith("ms_") if step["agent"] == "reviews_agent" else step["tool"] == "neo4j_relationships"
            assert expected, "Specialist used an unexpected tool"
        assert result["answer_generation"]["status"] == "complete" and "[E" in answer
        assert any(name == "agent" for name, _ in events) and any(name == "done" for name, _ in events)
        assert not any(name == "error" for name, _ in events)
        conversation = next(payload["conversation_id"] for name, payload in events if name == "metadata")
        history = await client.get(f"/api/conversations/{conversation}/messages", params={"user_id": user})
        history.raise_for_status()
        assert len(history.json()) == 2 and history.json()[1]["content"] == answer
        print(json.dumps({"passed": True, "user_id": user, "conversation_id": conversation,
                          "checks": "three specialists, allowed tools, budgets, SSE, final synthesis, persistence"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    asyncio.run(evaluate(parser.parse_args().base_url))
