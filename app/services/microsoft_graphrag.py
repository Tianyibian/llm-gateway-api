"""Async process adapter for a version-pinned Microsoft GraphRAG index."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

from app.models.graph_supervisor import GraphTool, RetrievalEvidence
from app.services.errors import LLMConfigurationError


REPO = Path(__file__).resolve().parents[2]
MODES = {GraphTool.MS_LOCAL: "local", GraphTool.MS_GLOBAL: "global", GraphTool.MS_DRIFT: "drift"}


class MicrosoftGraphRAGClient:
    def __init__(self, *, root: str, python: str, api_key: str, timeout: float = 90):
        self.root = (REPO / root).resolve()
        self.python = (REPO / python).absolute()
        self.api_key = api_key
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(1)

    def status(self):
        try:
            report = json.loads((self.root / "verified_index.json").read_text())
            if not self.python.is_file() or not report["ready"] or report["version"] != "3.1.2":
                raise ValueError("Runtime unavailable")
            if set(report["artifacts_sha256"]) != {"documents", "entities", "relationships", "text_units", "communities", "community_reports"}:
                raise ValueError("Incomplete artifacts")
            if not (self.root / "output/lancedb").is_dir():
                raise ValueError("Missing vector store")
            if hashlib.sha256((self.root / "corpus_manifest.json").read_bytes()).hexdigest() != report["corpus_sha256"]:
                raise ValueError("Corpus manifest changed")
            if hashlib.sha256((self.root / "settings.yaml").read_bytes()).hexdigest() != report["config_sha256"]:
                raise ValueError("Config changed")
            for name, expected in report["artifacts_sha256"].items():
                if name not in {"documents", "entities", "relationships", "text_units", "communities", "community_reports"}:
                    raise ValueError("Invalid artifact")
                if hashlib.sha256((self.root / "output" / f"{name}.parquet").read_bytes()).hexdigest() != expected:
                    raise ValueError("Artifact changed")
            return {"ready": True, "backend": "microsoft_graphrag", "version": "3.1.2",
                    "counts": report["counts"], "vector_counts": report["vector_counts"],
                    "vector_dimensions": report["vector_dimensions"],
                    "available_modes": ["local", "global", "drift"]}
        except (OSError, ValueError, KeyError, TypeError):
            return {"ready": False, "backend": "microsoft_graphrag", "available_modes": []}

    async def query(self, question: str, mode: str):
        if mode not in MODES.values() or not question.strip() or len(question) > 10_000:
            raise ValueError("Invalid graph query")
        if not self.status()["ready"]:
            raise LLMConfigurationError("Microsoft GraphRAG index is unavailable or changed.")
        async with self._semaphore:
            env = {**os.environ, "GRAPHRAG_API_KEY": self.api_key}
            # Questions travel over stdin; neither credentials nor user content
            # are interpolated into a shell command or written to a temp file.
            process = await asyncio.create_subprocess_exec(
                str(self.python), str(REPO / "graphrag_runtime/worker.py"), "query",
                "--root", str(self.root), "--method", mode,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL, env=env,
            )
            try:
                stdout, _ = await asyncio.wait_for(process.communicate(question.encode()), timeout=self.timeout)
                if process.returncode or len(stdout) > 2_000_000:
                    raise RuntimeError("Graph retrieval failed")
                result = json.loads(stdout)
                if result.get("backend") != "microsoft_graphrag" or not isinstance(result.get("evidence"), list):
                    raise RuntimeError("Invalid graph response")
                return result
            except BaseException:
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()
                raise


class MicrosoftGraphRAGTool:
    def __init__(self, client: MicrosoftGraphRAGClient, mode: str):
        if mode not in MODES.values():
            raise ValueError("Unsupported graph mode")
        self.client, self.mode = client, mode

    async def retrieve(self, question: str, *, limit: int) -> list[RetrievalEvidence]:
        if not 1 <= limit <= 5:
            raise ValueError("Invalid evidence limit")
        result = await self.client.query(question, self.mode)
        return [RetrievalEvidence.model_validate({key: row[key] for key in ("source_id", "text", "entities")})
                for row in result["evidence"][:limit]]
