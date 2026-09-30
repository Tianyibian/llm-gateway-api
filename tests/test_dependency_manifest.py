"""Keep the default installation usable without an optional requirements file."""
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("package", [
    "langchain", "langchain-core", "langgraph", "langchain-openai",
    "langchain-ollama", "numpy", "neo4j", "cryptography",
])
def test_assistant_dependencies_are_explicit_in_root_requirements(package):
    declarations = {
        re.split(r"[<>=!~\[;\s]", line.strip(), maxsplit=1)[0]
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "-"))
    }
    assert package in declarations


def test_legacy_langchain_entrypoint_reuses_root_requirements():
    lines = [
        line.strip()
        for line in (ROOT / "requirements/langchain.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert lines == ["-r ../requirements.txt"]
