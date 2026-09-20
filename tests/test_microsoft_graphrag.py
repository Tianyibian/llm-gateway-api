import asyncio
import hashlib
import json
import sys

import pytest

from app.models.graph_supervisor import GraphTool
from app.services.errors import LLMConfigurationError
from app.services.microsoft_graphrag import MicrosoftGraphRAGClient, MicrosoftGraphRAGTool
from graphrag_runtime.evidence import candidates, validate_citations


@pytest.fixture
def client(tmp_path):
    (tmp_path / "output/lancedb").mkdir(parents=True)
    (tmp_path / "settings.yaml").write_text("test: config")
    (tmp_path / "corpus_manifest.json").write_text("{}")
    artifacts = {}
    for name in ["documents", "entities", "relationships", "text_units", "communities", "community_reports"]:
        path = tmp_path / "output" / f"{name}.parquet"
        path.write_bytes(b"fixture")
        artifacts[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {"ready": True, "version": "3.1.2", "counts": {"documents": 1},
              "vector_counts": {"entity_description": 1}, "vector_dimensions": 1536,
              "config_sha256": hashlib.sha256((tmp_path / "settings.yaml").read_bytes()).hexdigest(),
              "corpus_sha256": hashlib.sha256((tmp_path / "corpus_manifest.json").read_bytes()).hexdigest(),
              "artifacts_sha256": artifacts}
    (tmp_path / "verified_index.json").write_text(json.dumps(report))
    return MicrosoftGraphRAGClient(root=str(tmp_path), python=sys.executable, api_key="test-secret", timeout=0.02)


def test_readiness_requires_verified_files(client):
    assert client.status()["ready"]
    (client.root / "output/entities.parquet").write_bytes(b"changed")
    assert not client.status()["ready"]


@pytest.mark.parametrize("filename", ["settings.yaml", "corpus_manifest.json", "verified_index.json"])
def test_changed_or_missing_metadata_is_not_ready(client, filename):
    (client.root / filename).write_text("changed")
    assert not client.status()["ready"]


def test_query_rejects_changed_index_before_process_launch(client):
    (client.root / "settings.yaml").write_text("changed")
    with pytest.raises(LLMConfigurationError):
        asyncio.run(client.query("Summarize themes", "global"))


@pytest.mark.parametrize("mode,question", [("shell", "x"), ("local", " "), ("global", "x" * 10001)])
def test_query_inputs_are_bounded(client, mode, question):
    with pytest.raises(ValueError):
        asyncio.run(client.query(question, mode))


class FakeProcess:
    returncode = None
    killed = False
    delay = 0
    payload = {"backend": "microsoft_graphrag", "evidence": []}

    async def communicate(self, value):
        self.input = value
        await asyncio.sleep(self.delay)
        self.returncode = 0
        return json.dumps(self.payload).encode(), None

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        return self.returncode


def test_process_uses_stdin_and_no_shell_or_credential_arguments(client, monkeypatch):
    process, captured = FakeProcess(), {}

    async def spawn(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    question = "Explain $(not-a-command); support themes"
    result = asyncio.run(client.query(question, "global"))
    assert result["backend"] == "microsoft_graphrag"
    assert process.input.decode() == question
    assert question not in captured["args"] and "test-secret" not in captured["args"]
    assert captured["kwargs"]["env"]["GRAPHRAG_API_KEY"] == "test-secret"


def test_timeout_kills_real_worker_boundary(client, monkeypatch):
    process = FakeProcess()
    process.delay = 1

    async def spawn(*args, **kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(client.query("Explain support themes", "local"))
    assert process.killed


def test_tool_returns_only_bounded_evidence_not_unverified_generated_answer():
    class Client:
        async def query(self, question, mode):
            return {"answer": "Unverified generated claim", "evidence": [
                {"source_id": f"ms:source:{i}", "text": "Original excerpt", "entities": [],
                 "source_document_ids": ["doc:1"]} for i in range(5)
            ]}

    rows = asyncio.run(MicrosoftGraphRAGTool(Client(), "local").retrieve("query", limit=2))
    assert len(rows) == 2 and all(row.text == "Original excerpt" for row in rows)


def test_evidence_resolves_persisted_ids_not_model_invented_ids():
    tables = {"entities": [{"title": "ACME SENSOR", "type": "PRODUCT"}],
              "text_units": [{"id": "uuid-1", "human_readable_id": 2, "text": "Acme Sensor works.", "document_id": "doc-1"}],
              "community_reports": []}
    result = candidates({"sources": [{"id": "999"}, {"id": "2"}]}, tables, "[Data: Sources (2)]")
    assert len(result) == 1
    assert result[0]["source_id"] == "microsoft_graphrag:text_unit:uuid-1"
    assert result[0]["source_document_ids"] == ["doc-1"]
    assert result[0]["entities"][0]["entity_type"] == "Product"


@pytest.mark.parametrize("answer,valid", [
    ("Claim [Data: Sources (2)]", True),
    ("Claim [Data: Sources (2, 466)]", False),
    ("Claim [Data: My invented document]", False),
    ("Claim without evidence", False),
    ("Claim [Data: Reports (0); Sources (2)]", True),
])
def test_generated_citations_are_checked_against_context_ids(answer, valid):
    result = validate_citations(answer, {"sources": [{"id": "2"}], "reports": [{"id": "0"}]})
    assert result["valid"] is valid


def test_only_microsoft_adapters_are_registered_for_real_index(monkeypatch, client):
    from app.core.config import Settings
    from app.services.factory import LLMServiceFactory
    factory = LLMServiceFactory(Settings(_env_file=None, microsoft_graphrag_enabled=True))
    monkeypatch.setattr(factory, "create_microsoft_graphrag_client", lambda: client)
    registered = factory.create_graph_retrieval_tools()
    assert set(registered) == {GraphTool.MS_LOCAL, GraphTool.MS_GLOBAL, GraphTool.MS_DRIFT}
    assert GraphTool.NEO4J not in registered
