"""Live hybrid/filter/rerank/context smoke test; no saved chat or database writes."""
import asyncio
import json

from app.db.session import engine
from app.models.policy_filters import PolicyMetadataFilters
from app.services.factory import LLMServiceFactory


async def main():
    try:
        factory = LLMServiceFactory()
        service = factory.create_assistant()
        filters = PolicyMetadataFilters(source_types=["pdf"])
        events = [item async for item in service.stream("What is the company's return policy?", policy_filters=filters)]
        documents = [doc for name, payload in events if name == "sources" for doc in payload.get("documents", [])]
        answer = "".join(payload["content"] for name, payload in events if name == "delta")
        guard = [payload for name, payload in events if name == "policy_guardrail"]
        checks = {
            "single_allowed_policy_guard": len(guard) == 1 and guard[0].get("allowed") is True,
            "filtered_pdf_context": bool(documents) and all(doc["source_type"] == "pdf" for doc in documents),
            "real_reranker": bool(documents) and all(doc["score_type"] == "cross_encoder" and doc["reranker_model"] for doc in documents),
            "top_k": 0 < len(documents) <= factory.settings.rag_retrieval_k,
            "descending_scores": [doc["score"] for doc in documents] == sorted([doc["score"] for doc in documents], reverse=True),
            "cited_answer": bool(answer) and "[1]" in answer,
        }
        print(json.dumps({"passed": all(checks.values()), "checks": checks,
            "ranking": [{key: doc[key] for key in ("title", "hybrid_rank", "reranker_rank", "score")} for doc in documents],
            "answer": answer}, ensure_ascii=False), flush=True)
        return all(checks.values())
    except Exception as exc:
        print(json.dumps({"passed": False, "error_type": type(exc).__name__}), flush=True)
        return False
    finally:
        await engine.dispose()


if __name__ == "__main__":
    raise SystemExit(0 if asyncio.run(main()) else 1)
