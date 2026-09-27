"""Download pinned safe weights locally; no policy documents leave this machine."""
from huggingface_hub import snapshot_download
from app.core.config import Settings
from app.services.policy_reranker import MODEL_ID, MODEL_REVISION, _load


def main():
    path = snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION,
        local_dir=Settings().rag_reranker_model_path,
        allow_patterns=["*.json", "vocab.txt", "model.safetensors", "README.md"],
        token=False)
    model = _load(path)
    scores = model.predict([("How do I return an item?", "Return instructions and refund eligibility.")])
    print({"ready": True, "model": MODEL_ID, "revision": MODEL_REVISION, "smoke_score": float(scores[0])})


if __name__ == "__main__":
    main()
