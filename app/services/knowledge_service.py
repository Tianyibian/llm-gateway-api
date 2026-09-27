from __future__ import annotations

from dataclasses import dataclass
import asyncio
import math
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import KnowledgeChunk, KnowledgeDocument
from app.models.policy_filters import PolicyMetadataFilters
from app.services.embedding_service import EmbeddingService
from app.services.errors import KnowledgeBaseNotReadyError
from app.services.knowledge_loader import KnowledgeBaseLoader, KnowledgeChunker
from app.services.ensemble_retriever import bm25_rank, reciprocal_rank_fusion


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
    score_type: str = "cosine_similarity"
    vector_rank: int | None = None
    bm25_rank: int | None = None
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    hybrid_rank: int | None = None
    reranker_rank: int | None = None
    reranker_model: str | None = None

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
            "score_type": self.score_type,
            "vector_rank": self.vector_rank,
            "bm25_rank": self.bm25_rank,
            "vector_score": self.vector_score,
            "bm25_score": self.bm25_score,
            "rrf_score": self.rrf_score,
            "hybrid_rank": self.hybrid_rank,
            "reranker_rank": self.reranker_rank,
            "reranker_model": self.reranker_model,
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
    """Hybrid RRF candidates -> metadata narrowing -> reranking -> top context.

    This bounded prototype scans the small public corpus. It does not build a
    BM25 index from vector-only candidates or silently truncate the corpus.
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        embedding_service: EmbeddingService,
        default_k: int = 5,
        candidate_k: int = 20,
        max_corpus_chunks: int = 10000,
        reranker=None,
        context_max_characters: int = 16000,
    ) -> None:
        self._session_factory = session_factory
        self.embedding_service = embedding_service
        self.default_k = default_k
        self.candidate_k = candidate_k
        self.max_corpus_chunks = max_corpus_chunks
        self.reranker = reranker
        self.context_max_characters = context_max_characters

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
        filters: PolicyMetadataFilters | None = None,
    ) -> list[RetrievedKnowledge]:
        if not query.strip():
            raise ValueError("A non-empty retrieval query is required")
        result_k = self.default_k if k is None else k
        if not 1 <= result_k <= 100:
            raise ValueError("Retrieval k must be between 1 and 100")
        metadata_filters = PolicyMetadataFilters.model_validate(
            filters.model_dump() if isinstance(filters, PolicyMetadataFilters) else ({} if filters is None else filters)
        )
        query_vector = await self.embedding_service.embed_query(query)
        distance = KnowledgeChunk.embedding.cosine_distance(query_vector).label(
            "distance"
        )
        statement = (
            select(KnowledgeChunk.id, KnowledgeChunk.content,
                   KnowledgeChunk.chunk_index, KnowledgeDocument, distance)
            .join(
                KnowledgeDocument,
                KnowledgeDocument.id == KnowledgeChunk.document_id,
            )
            .where(
                KnowledgeDocument.visibility == "public",
                KnowledgeDocument.embedding_provider == self.embedding_service.provider,
                KnowledgeDocument.embedding_model == self.embedding_service.model,
            )
            # Read the complete bounded corpus independently of ANN candidates.
            .order_by(KnowledgeChunk.id)
            .limit(self.max_corpus_chunks + 1)
        )
        async with self._session_factory() as session:
            rows = (await session.execute(statement)).all()
        if not rows:
            if category or metadata_filters.model_dump(exclude_none=True):
                # No matching eligible documents is not an ingestion failure and
                # must never fall back to a broader unfiltered search.
                return []
            raise KnowledgeBaseNotReadyError(
                "The Knowledge Base has no searchable chunks. Run the ingestion command."
            )
        if len(rows) > self.max_corpus_chunks:
            raise KnowledgeBaseNotReadyError(
                "The Knowledge Base exceeds the bounded BM25 corpus limit. "
                "Configure an indexed lexical backend before searching this corpus."
            )

        candidate_k = max(self.candidate_k, result_k)
        by_id = {str(row[0]): row for row in rows}
        vector_rows = sorted(rows, key=lambda row: (float(row[4]), str(row[0])))
        vector_ids = [str(row[0]) for row in vector_rows[:candidate_k]]
        lexical = await asyncio.to_thread(
            bm25_rank, query,
            {str(row[0]): f"{row[3].title}\n{row[1]}" for row in rows},
            limit=candidate_k,
        )
        lexical_ids = [key for key, _ in lexical]
        vector_ranks = {key: rank for rank, key in enumerate(vector_ids, 1)}
        lexical_ranks = {key: rank for rank, key in enumerate(lexical_ids, 1)}
        lexical_scores = dict(lexical)
        # Fuse the entire bounded union before applying user metadata constraints.
        # Permissions and embedding compatibility were already enforced in SQL.
        fused = reciprocal_rank_fusion(vector_ids, lexical_ids, limit=2 * candidate_k)
        hybrid_ranks = {key: rank for rank, (key, _) in enumerate(fused, 1)}
        def permitted(key):
            document = by_id[key][3]
            return ((not category or document.category == category)
                and (metadata_filters.categories is None or document.category in metadata_filters.categories)
                and (metadata_filters.source_types is None or document.source_type in metadata_filters.source_types)
                and (metadata_filters.source_paths is None or document.source_path in metadata_filters.source_paths))
        filtered = [(key, score) for key, score in fused if permitted(key)]
        if not filtered:
            return []  # Never broaden filters or pad context with excluded records.
        scores = {}
        if self.reranker is not None:
            values = await self.reranker.score(query, [
                f"{by_id[key][3].title}\n{by_id[key][1]}" for key, _ in filtered])
            if len(values) != len(filtered) or any(not math.isfinite(float(value)) for value in values):
                raise KnowledgeBaseNotReadyError("The policy reranker returned invalid scores.")
            scores = {key: float(value) for (key, _), value in zip(filtered, values)}
            filtered.sort(key=lambda item: (-scores[item[0]], hybrid_ranks[item[0]], item[0]))

        results: list[RetrievedKnowledge] = []
        remaining = self.context_max_characters
        for rank, (key, score) in enumerate(filtered, 1):
            _, content, chunk_index, document, raw_distance = by_id[key]
            metadata = document.document_metadata or {}
            # Preserve whole evidence chunks; bound context characters including
            # a conservative allowance for source headers/citations (not tokens).
            size = len(content) + len(document.title) + len(str(metadata.get("source_file", document.source_path))) + 150
            if size > remaining:
                continue
            remaining -= size
            results.append(
                RetrievedKnowledge(
                    content=content,
                    title=document.title,
                    source_path=document.source_path,
                    source_file=str(metadata.get("source_file", document.source_path)),
                    source_type=document.source_type,
                    category=document.category,
                    page=(int(metadata["page"]) if metadata.get("page") else None),
                    score=scores.get(key, score),
                    chunk_index=chunk_index,
                    score_type="cross_encoder" if self.reranker is not None else "hybrid_rrf",
                    vector_rank=vector_ranks.get(key),
                    bm25_rank=lexical_ranks.get(key),
                    vector_score=1.0 - float(raw_distance),
                    bm25_score=lexical_scores.get(key),
                    rrf_score=score,
                    hybrid_rank=hybrid_ranks[key],
                    reranker_rank=rank if self.reranker is not None else None,
                    reranker_model=self.reranker.model if self.reranker is not None else None,
                )
            )
            if len(results) >= result_k:
                break
        if not results:
            raise KnowledgeBaseNotReadyError("No complete policy excerpt fits the configured context budget.")
        return results
