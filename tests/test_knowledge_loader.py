from collections import Counter
from pathlib import Path

import pytest

from app.services.knowledge_loader import KnowledgeBaseLoader, KnowledgeChunker


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_loader_reads_all_supported_knowledge_formats() -> None:
    documents = KnowledgeBaseLoader(PROJECT_ROOT / "Knowledge Base").load()
    type_counts = Counter(document.metadata["source_type"] for document in documents)
    visibility_counts = Counter(
        document.metadata["visibility"] for document in documents
    )

    assert type_counts == {"faq": 300, "html": 150, "pdf": 173, "docx": 3}
    assert visibility_counts == {"public": 623, "internal": 3}
    assert all(document.page_content.strip() for document in documents)
    assert all(document.metadata["source_path"] for document in documents)


def test_internal_support_documents_are_not_marked_public() -> None:
    documents = KnowledgeBaseLoader(PROJECT_ROOT / "Knowledge Base").load()
    internal_titles = {
        document.metadata["title"]
        for document in documents
        if document.metadata["visibility"] == "internal"
    }

    assert internal_titles == {
        "Customer Service SOP",
        "Escalation Process",
        "Refund Process",
    }


def test_chunker_adds_overlap_and_preserves_metadata() -> None:
    document = next(
        item
        for item in KnowledgeBaseLoader(PROJECT_ROOT / "Knowledge Base").load()
        if len(item.page_content) > 1500
    )
    chunker = KnowledgeChunker(chunk_size=500, chunk_overlap=100)

    chunks = chunker.split_documents([document])

    assert len(chunks) > 1
    assert all(0 < len(chunk.page_content) <= 500 for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(
        range(len(chunks))
    )
    assert all(
        chunk.metadata["source_path"] == document.metadata["source_path"]
        for chunk in chunks
    )


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(199, 0), (500, -1), (500, 500)],
)
def test_chunker_rejects_invalid_settings(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    with pytest.raises(ValueError):
        KnowledgeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
