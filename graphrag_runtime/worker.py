"""Isolated worker for the REAL Microsoft GraphRAG 3.1.2 Python APIs.

Run with .venv-graphrag/bin/python. stdout is one JSON result; progress goes to
stderr. Never import this module into the FastAPI process: the vendor config
loader changes cwd and has its own dependency/model singletons.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import redirect_stdout
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import time


REPO = Path(__file__).resolve().parents[1]
TABLES = ("documents", "text_units", "entities", "relationships", "communities", "community_reports")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root, source):
    if root.exists():
        raise FileExistsError("Choose a new workspace")
    files = sorted((source / "input").glob("*.txt"))
    if not 1 <= len(files) <= 64 or sum(p.stat().st_size for p in files) > 200_000:
        raise ValueError("Small-corpus limit exceeded")
    if any(p.is_symlink() for p in files):
        raise ValueError("Symlink inputs are not allowed")
    root.mkdir(parents=True)
    (root / "input").mkdir()
    (root / "prompts").mkdir()
    for path in files:
        shutil.copyfile(path, root / "input" / path.name)
    shutil.copyfile(REPO / "graphrag_runtime/settings.yaml", root / "settings.yaml")
    shutil.copyfile(REPO / "graphrag_runtime/business_extract_graph.txt", root / "prompts/business_extract_graph.txt")
    manifest = {
        "scope": "approved_local_business_sample", "privacy_review_required": True,
        "approval": "explicit_cli_approval", "source": source.name,
        "documents": {p.name: digest(p) for p in files},
        "limitations": "Sampled catalog and unverified reviews; not authoritative policies or population statistics.",
    }
    (root / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"prepared": True, "document_count": len(files), "index_built": False}


def check_inputs(root):
    manifest = json.loads((root / "corpus_manifest.json").read_text())
    files = {p.name: digest(p) for p in (root / "input").glob("*.txt")}
    if files != manifest["documents"]:
        raise ValueError("Approved inputs changed")
    return manifest


def configure(root):
    from dotenv import dotenv_values
    from graphrag.config.load_config import load_config

    key = os.environ.get("GRAPHRAG_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        key = dotenv_values(REPO / ".env").get("OPENAI_API_KEY")
    if not key or key.startswith("your_"):
        raise ValueError("OpenAI credential missing")
    os.environ["GRAPHRAG_API_KEY"] = key
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    # No .env file containing credentials is copied into the indexing workspace.
    return load_config(root)


def inspect_index(root):
    import pandas as pd
    import lancedb

    counts = {name: len(pd.read_parquet(root / "output" / f"{name}.parquet")) for name in TABLES}
    vector_path = root / "output/lancedb"
    vector_counts = {}
    if vector_path.is_dir():
        db = lancedb.connect(str(vector_path))
        for name in db.list_tables().tables:
            table = db.open_table(name)
            if table.schema.field("vector").type.list_size != 1536:
                raise ValueError("Unexpected embedding dimension")
            vector_counts[name] = table.count_rows()
    return {"package": "graphrag", "version": version("graphrag"), "counts": counts,
            "vector_store": "lancedb", "vector_dimensions": 1536,
            "vector_counts": vector_counts,
            "ready": all(counts.values()) and len(vector_counts) == 3 and all(vector_counts.values())}


def verify(root):
    check_inputs(root)
    result = inspect_index(root)
    if not result["ready"]:
        raise ValueError("Incomplete index")
    result.update(config_sha256=digest(root / "settings.yaml"),
                  corpus_sha256=digest(root / "corpus_manifest.json"),
                  artifacts_sha256={name: digest(root / "output" / f"{name}.parquet") for name in TABLES})
    (root / "verified_index.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


async def build(root):
    import graphrag.api as api
    from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks

    check_inputs(root)
    if (root / "index_report.json").exists():
        raise FileExistsError("Index already built; prepare a new version")
    config = configure(root)

    class Progress(NoopWorkflowCallbacks):
        def workflow_start(self, name, instance):
            print(f"Starting {name}", file=sys.stderr, flush=True)

        def workflow_end(self, name, instance):
            print(f"Finished {name}", file=sys.stderr, flush=True)

    started = time.monotonic()
    outputs = await api.build_index(config=config, method="standard", callbacks=[Progress()])
    failed = [out.workflow for out in outputs if out.error is not None]
    if failed:
        return {"ready": False, "failed_workflows": failed,
                "error_types": [type(out.error).__name__ for out in outputs if out.error is not None]}
    result = {**inspect_index(root), "elapsed_seconds": round(time.monotonic() - started, 2),
              "workflows": [out.workflow for out in outputs],
              "config_sha256": digest(root / "settings.yaml"),
              "corpus_sha256": digest(root / "corpus_manifest.json")}
    if not result["ready"]:
        raise ValueError("Incomplete index")
    (root / "index_report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


async def query(root, method, question):
    import graphrag.api as api
    import pandas as pd

    check_inputs(root)
    if not (root / "verified_index.json").exists():
        raise ValueError("Index not verified")
    verified = json.loads((root / "verified_index.json").read_text())
    if verified["corpus_sha256"] != digest(root / "corpus_manifest.json"):
        raise ValueError("Verified corpus changed")
    if verified["config_sha256"] != digest(root / "settings.yaml") or any(
        verified["artifacts_sha256"][name] != digest(root / "output" / f"{name}.parquet") for name in TABLES
    ):
        raise ValueError("Verified index changed")
    config = configure(root)
    frames = {name: pd.read_parquet(root / "output" / f"{name}.parquet") for name in TABLES}
    common = {"config": config, "entities": frames["entities"], "communities": frames["communities"],
              "community_reports": frames["community_reports"], "community_level": 0,
              "response_type": (
                  "A concise answer with exact table-ID citations. For Sources, cite only the id column of the Sources context table, "
                  "NEVER Review IDs embedded in source text. Never cite a question or an invented source name. "
                  "Treat reviews as unverified sample opinions, not authoritative policies. If evidence for a product "
                  "is missing from the retrieved context, say it is insufficient; do not claim the entire index has no such evidence."
              ),
              "query": question}
    if method == "global":
        answer, context = await api.global_search(**common, dynamic_community_selection=False)
    elif method == "local":
        answer, context = await api.local_search(**common, text_units=frames["text_units"],
                                                 relationships=frames["relationships"], covariates=None)
    elif method == "drift":
        answer, context = await api.drift_search(**common, text_units=frames["text_units"],
                                                 relationships=frames["relationships"])
    else:
        raise ValueError("Unsupported graph search mode")
    def records(value):
        if isinstance(value, pd.DataFrame):
            return json.loads(value.to_json(orient="records"))
        if isinstance(value, dict):
            return {key: records(item) for key, item in value.items()}
        if isinstance(value, list):
            return [records(item) for item in value]
        return value
    from evidence import candidates, validate_citations
    serialized = records(context)
    evidence = candidates(serialized, {name: records(frame) for name, frame in frames.items()}, answer)
    citations = validate_citations(answer, serialized)
    safe_answer = answer if citations["valid"] else (
        "The generated answer contained unverifiable citations. Verified retrieved excerpts:\n" +
        "\n".join(f"[{e['source_id']}] {e['text']}" for e in evidence)
    )
    return {"backend": "microsoft_graphrag", "method": method,
            "answer": safe_answer, "generated_answer": answer, "citation_validation": citations,
            "context": serialized, "evidence": evidence,
            "version": version("graphrag"), "corpus_sha256": verified["corpus_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "index", "verify", "status", "query"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--source")
    parser.add_argument("--approve-reviewed-inputs", action="store_true")
    parser.add_argument("--method", choices=["local", "global", "drift"], default="local")
    parser.add_argument("--query")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve() if args.output else None
    if version("graphrag") != "3.1.2":
        raise RuntimeError("Use the pinned GraphRAG runtime")
    try:
        with redirect_stdout(sys.stderr):
            if args.action == "prepare":
                if not args.source or not args.approve_reviewed_inputs:
                    raise ValueError("Review and explicitly approve the input corpus first")
                result = prepare(root, Path(args.source).resolve())
            elif args.action == "index":
                result = asyncio.run(asyncio.wait_for(build(root), timeout=900))
            elif args.action == "status":
                result = inspect_index(root)
            elif args.action == "verify":
                result = verify(root)
            else:
                question = args.query if args.query is not None else sys.stdin.read(10001)
                if not question.strip() or len(question) > 10_000:
                    raise ValueError("Invalid query")
                result = asyncio.run(asyncio.wait_for(query(root, args.method, question), timeout=90))
        if output:
            output.write_text(json.dumps(result, indent=2) + "\n")
        printed = ({"backend": result.get("backend"), "method": result.get("method"),
                    "answer_characters": len(str(result.get("answer", ""))),
                    "evidence_count": len(result.get("evidence", [])), "output": str(output)}
                   if output and args.action == "query" else result)
        print(json.dumps(printed))
        if result.get("ready") is False:
            raise SystemExit(1)
    except Exception as exc:
        # Provider exceptions can contain request data; never print them or keys.
        print(json.dumps({"error": type(exc).__name__, "status_code": getattr(exc, "status_code", None)}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
