"""Local, bounded Cross-Encoder inference; no remote code or runtime downloads."""
import asyncio
from functools import lru_cache
import math
from pathlib import Path
import threading

from app.services.errors import KnowledgeBaseNotReadyError

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
_inference_lock = threading.Lock()


@lru_cache(maxsize=2)
def _load(path: str):
    from sentence_transformers import CrossEncoder
    import torch
    torch.set_num_threads(2)
    return CrossEncoder(path, device="cpu", max_length=512,
                        local_files_only=True, trust_remote_code=False,
                        model_kwargs={"use_safetensors": True},
                        activation_fn=torch.nn.Identity())


class PolicyCrossEncoder:
    def __init__(self, *, model_path: str, timeout: float = 60):
        self.model_path = str(Path(model_path).resolve())
        self.timeout = timeout
        self.model = MODEL_ID

    def _score(self, query, documents):
        # A timed-out worker retains the lock until inference actually finishes.
        # New requests fail busy instead of piling up background inference jobs.
        if not _inference_lock.acquire(blocking=False):
            raise RuntimeError("Reranker is busy")
        try:
            if not Path(self.model_path, "model.safetensors").is_file():
                raise RuntimeError("Reranker weights have not been prepared")
            scores = _load(self.model_path).predict(
                [(query, document) for document in documents], batch_size=8,
                show_progress_bar=False, convert_to_numpy=True)
            return [float(score) for score in scores]
        finally:
            _inference_lock.release()

    async def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        if len(documents) > 200:
            raise ValueError("Too many reranking candidates")
        try:
            scores = await asyncio.wait_for(asyncio.to_thread(self._score, query, documents), self.timeout)
            if len(scores) != len(documents) or any(not math.isfinite(score) for score in scores):
                raise ValueError("Invalid reranker output")
            return scores
        except Exception:
            raise KnowledgeBaseNotReadyError(
                "The policy reranker is unavailable. Prepare its local model or retry later; no unreranked answer was generated."
            ) from None
