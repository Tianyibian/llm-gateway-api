from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import KnowledgeChunk, KnowledgeDocument
from app.services.embedding_service import EmbeddingService
from app.services.errors import KnowledgeBaseNotReadyError
from app.services.knowledge_loader import KnowledgeBaseLoader, KnowledgeChunker


@dataclass(frozen=True, kw_only=True)
class IngestionReport:
    sources_loaded: int
    sources_indexed: int
    sources_skipped: int
    sources_deleted: int
    chunks_indexed: int
    embedding_provider: str
    embedding_model: str


@dataclass(frozen=True, kw_only=True)
class KnowledgeStatus:
    documents: int
    chunks: int
    public_documents: int
    embedding_provider: str
    embedding_model: str


@dataclass(frozen=True, kw_only=True)
class RetrievedKnowledge:
    content: str
    title: str
    source_path: str
    source_file: str
    source_type: str
    category: str | None
    page: int | None
    score: float
    chunk_index: int

    def citation(self) -> dict[str, object]:
        return {
            "title": self.title,
            "source_path": self.source_path,
            "source_file": self.source_file,
            "source_type": self.source_type,
            "category": self.category,
            "page": self.page,
            "score": round(self.score, 4),
            "chunk_index": self.chunk_index,
        }


class KnowledgeIngestionService:
    """Idempotently load, chunk, embed, and persist the local Knowledge Base."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        embedding_service: EmbeddingService,
        knowledge_base_dir: str | Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._session_factory = session_factory
        self.embedding_service = embedding_service
        self.loader = KnowledgeBaseLoader(knowledge_base_dir)
        self.chunker = KnowledgeChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @staticmethod
    def _checksum(document: Document) -> str:
        payload = json.dumps(
            {
                "content": document.page_content,
                "metadata": document.metadata,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _document_id(source_path: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"aster-knowledge:{source_path}"))

    @staticmethod
    def _chunk_id(source_path: str, chunk_index: int) -> str:
        return str(
            uuid5(NAMESPACE_URL, f"aster-knowledge:{source_path}:chunk:{chunk_index}")
        )

    async def ingest(self) -> IngestionReport:
        documents = self.loader.load()
        loaded_by_source = {
            str(document.metadata["source_path"]): document
            for document in documents
        }

        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(
                        KnowledgeDocument.source_path,
                        KnowledgeDocument.checksum,
                        KnowledgeDocument.embedding_provider,
                        KnowledgeDocument.embedding_model,
                    )
                )
            ).all()
        existing = {
            source_path: (checksum, provider, model)
            for source_path, checksum, provider, model in rows
        }

        pending: list[Document] = []
        skipped = 0
        for source_path, document in loaded_by_source.items():
            expected = (
                self._checksum(document),
                self.embedding_service.provider,
                self.embedding_service.model,
            )
            if existing.get(source_path) == expected:
                skipped += 1
            else:
                pending.append(document)

        chunks_by_source: dict[str, list[Document]] = {}
        all_chunks: list[Document] = []
        for document in pending:
            chunks = self.chunker.split_documents([document])
            source_path = str(document.metadata["source_path"])
            chunks_by_source[source_path] = chunks
            all_chunks.extend(chunks)

        vectors = (
            await self.embedding_service.embed_documents(
                [chunk.page_content for chunk in all_chunks]
            )
            if all_chunks
            else []
        )
        vector_cursor = 0
        now = datetime.now(timezone.utc)

        stale_sources = set(existing) - set(loaded_by_source)
        changed_sources = {str(document.metadata["source_path"]) for document in pending}
        async with self._session_factory.begin() as session:
            sources_to_remove = stale_sources | changed_sources
            if sources_to_remove:
                await session.execute(
                    delete(KnowledgeDocument).where(
                        KnowledgeDocument.source_path.in_(sources_to_remove)
                    )
                )
                await session.flush()

            for document in pending:
                metadata = dict(document.metadata)
                source_path = str(metadata["source_path"])
                document_id = self._document_id(source_path)
                knowledge_document = KnowledgeDocument(
                    id=document_id,
                    source_path=source_path,
                    source_type=str(metadata.get("source_type", "unknown")),
                    title=str(metadata.get("title", Path(source_path).stem)),
                    category=(str(metadata["category"]) if metadata.get("category") else None),
                    visibility=str(metadata.get("visibility", "public")),
                    checksum=self._checksum(document),
                    embedding_provider=self.embedding_service.provider,
                    embedding_model=self.embedding_service.model,
                    document_metadata=metadata,
                    updated_at=now,
                )
                session.add(knowledge_document)

                for chunk in chunks_by_source[source_path]:
                    chunk_index = int(chunk.metadata["chunk_index"])
                    session.add(
                        KnowledgeChunk(
                            id=self._chunk_id(source_path, chunk_index),
                            document_id=document_id,
                            chunk_index=chunk_index,
                            content=chunk.page_content,
                            character_count=len(chunk.page_content),
                            chunk_metadata=dict(chunk.metadata),
                            embedding=vectors[vector_cursor],
                        )
                    )
                    vector_cursor += 1

        return IngestionReport(
            sources_loaded=len(documents),
            sources_indexed=len(pending),
            sources_skipped=skipped,
            sources_deleted=len(stale_sources),
            chunks_indexed=len(all_chunks),
            embedding_provider=self.embedding_service.provider,
            embedding_model=self.embedding_service.model,
        )


class KnowledgeRetriever:
    """Retrieve public Knowledge Base chunks using pgvector cosine distance."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        embedding_service: EmbeddingService,
        default_k: int = 5,
    ) -> None:
        self._session_factory = session_factory
        self.embedding_service = embedding_service
        self.default_k = default_k

    async def status(self) -> KnowledgeStatus:
        compatible_documents = (
            KnowledgeDocument.embedding_provider == self.embedding_service.provider,
            KnowledgeDocument.embedding_model == self.embedding_service.model,
        )
        async with self._session_factory() as session:
            documents = await session.scalar(
                select(func.count(KnowledgeDocument.id)).where(*compatible_documents)
            )
            chunks = await session.scalar(
                select(func.count(KnowledgeChunk.id))
                .join(KnowledgeDocument)
                .where(*compatible_documents)
            )
            public_documents = await session.scalar(
                select(func.count(KnowledgeDocument.id)).where(
                    *compatible_documents,
                    KnowledgeDocument.visibility == "public",
                )
            )
        return KnowledgeStatus(
            documents=int(documents or 0),
            chunks=int(chunks or 0),
            public_documents=int(public_documents or 0),
            embedding_provider=self.embedding_service.provider,
            embedding_model=self.embedding_service.model,
        )

    async def retrieve(
        self,
        query: str,
        *,
        k: int | None = None,
        category: str | None = None,
    ) -> list[RetrievedKnowledge]:
        query_vector = await self.embedding_service.embed_query(query)
        distance = KnowledgeChunk.embedding.cosine_distance(query_vector).label(
            "distance"
        )
        statement = (
            select(KnowledgeChunk, KnowledgeDocument, distance)
            .join(
                KnowledgeDocument,
                KnowledgeDocument.id == KnowledgeChunk.document_id,
            )
            .where(
                KnowledgeDocument.visibility == "public",
                KnowledgeDocument.embedding_provider == self.embedding_service.provider,
                KnowledgeDocument.embedding_model == self.embedding_service.model,
            )
            .order_by(distance)
            .limit(k or self.default_k)
        )
        if category:
            statement = statement.where(KnowledgeDocument.category == category)

        async with self._session_factory() as session:
            rows = (await session.execute(statement)).all()
        if not rows:
            raise KnowledgeBaseNotReadyError(
                "The Knowledge Base has no searchable chunks. Run the ingestion command."
            )

        results: list[RetrievedKnowledge] = []
        for chunk, document, raw_distance in rows:
            metadata = document.document_metadata or {}
            score = max(0.0, min(1.0, 1.0 - float(raw_distance)))
            results.append(
                RetrievedKnowledge(
                    content=chunk.content,
                    title=document.title,
                    source_path=document.source_path,
                    source_file=str(metadata.get("source_file", document.source_path)),
                    source_type=document.source_type,
                    category=document.category,
                    page=(int(metadata["page"]) if metadata.get("page") else None),
                    score=score,
                    chunk_index=chunk.chunk_index,
                )
            )
        return results
