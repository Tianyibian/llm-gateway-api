"""Local read-only previews: real Parquet fixtures, no model calls."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from app.cli.inspect_graphrag import preview


RUNTIME = Path(__file__).resolve().parents[1] / ".venv-graphrag/bin/python"


def runtime_preview(root, **kwargs):
    result = subprocess.run([
        str(RUNTIME), "-c",
        "import json, sys; from app.cli.inspect_graphrag import preview; "
        "print(preview(sys.argv[1], 'entities', **json.loads(sys.argv[2])))",
        str(root), json.dumps(kwargs),
    ], check=True, capture_output=True, text=True, timeout=15)
    return result.stdout


@pytest.fixture
def workspace(tmp_path):
    if not RUNTIME.is_file():
        pytest.skip("Install the isolated GraphRAG runtime for real Parquet checks")
    (tmp_path / "output").mkdir()
    path = tmp_path / "output/entities.parquet"
    subprocess.run([
        str(RUNTIME), "-c",
        "import sys; import pyarrow as pa; import pyarrow.parquet as pq; "
        "pq.write_table(pa.table({'title': ['FIRST', 'SECOND'], "
        "'type': ['Product', 'Product'], 'description': ['First item', 'Second item']}), sys.argv[1])",
        str(path),
    ], check=True, capture_output=True, text=True, timeout=15)
    return tmp_path, path


def test_preview_reads_bounded_rows_without_changing_file(workspace):
    root, path = workspace
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = runtime_preview(root, limit=1)
    assert "2 rows" in result and "FIRST" in result and "SECOND" not in result
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_schema_does_not_include_row_values(workspace):
    root, _ = workspace
    result = runtime_preview(root, schema_only=True)
    assert "title: string" in result and "FIRST" not in result


def test_column_selection(workspace):
    root, _ = workspace
    assert "First item" not in runtime_preview(root, columns=["title"])
    result = subprocess.run([
        str(RUNTIME), "-m", "app.cli.inspect_graphrag", "--root", str(root),
        "--table", "entities", "--columns", "unknown",
    ], capture_output=True, text=True, timeout=15)
    assert result.returncode == 1 and "Cannot preview" in result.stderr


@pytest.mark.parametrize("table,limit", [("../private", 5), ("entities", 0), ("entities", 101)])
def test_invalid_inputs_are_rejected(workspace, table, limit):
    root, _ = workspace
    with pytest.raises(ValueError):
        preview(root, table, limit=limit)
