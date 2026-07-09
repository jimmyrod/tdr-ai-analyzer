from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.config import Settings
from app.embeddings import cosine_similarity
from app.schemas import TextChunk


@dataclass
class VectorRecord:
    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any]


class LocalVectorStore:
    """Small JSON-backed vector store used as a reliable local fallback."""

    def __init__(self, path: Path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.records_path = self.path / "vectors.json"
        self._records: list[VectorRecord] = []
        self._load()

    def _load(self) -> None:
        if not self.records_path.exists():
            return
        try:
            raw_records = json.loads(self.records_path.read_text(encoding="utf-8"))
            self._records = [VectorRecord(**record) for record in raw_records]
        except Exception:
            self._records = []

    def _persist(self) -> None:
        self.records_path.write_text(
            json.dumps([asdict(record) for record in self._records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def reset_document(self, document_name: str) -> None:
        self._records = [
            record for record in self._records if record.metadata.get("document_name") != document_name
        ]
        self._persist()

    def add_chunks(
        self,
        document_name: str,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> None:
        self.reset_document(document_name)
        for chunk, embedding in zip(chunks, embeddings):
            metadata = dict(chunk.metadata)
            metadata["document_name"] = document_name
            self._records.append(
                VectorRecord(
                    id=f"{document_name}:{chunk.index}",
                    text=chunk.text,
                    embedding=embedding,
                    metadata=metadata,
                )
            )
        self._persist()

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_name: str | None = None,
    ) -> list[VectorRecord]:
        candidates = self._records
        if document_name:
            candidates = [
                record for record in candidates if record.metadata.get("document_name") == document_name
            ]
        scored = [
            (cosine_similarity(query_embedding, record.embedding), record)
            for record in candidates
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for score, record in scored[:top_k] if score > 0]


class SupabaseVectorStore:
    """Vector store backed by a Supabase Postgres table with pgvector."""

    TABLE = "document_chunks"
    MATCH_FUNCTION = "match_document_chunks"

    def __init__(self, settings: Settings):
        from supabase import create_client

        self.client = create_client(settings.supabase_url, settings.supabase_secret_key)

    def reset_document(self, document_name: str) -> None:
        self.client.table(self.TABLE).delete().eq("document_name", document_name).execute()

    def add_chunks(
        self,
        document_name: str,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> None:
        self.reset_document(document_name)
        if not chunks:
            return
        rows = [
            {
                "id": f"{document_name}:{chunk.index}",
                "document_name": document_name,
                "chunk_index": chunk.index,
                "text": chunk.text,
                "embedding": embedding,
                "metadata": dict(chunk.metadata),
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]
        self.client.table(self.TABLE).insert(rows).execute()

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_name: str | None = None,
    ) -> list[VectorRecord]:
        response = self.client.rpc(
            self.MATCH_FUNCTION,
            {
                "query_embedding": query_embedding,
                "match_count": top_k,
                "filter_document_name": document_name,
            },
        ).execute()
        return [
            VectorRecord(
                id=row["id"],
                text=row["text"],
                # match_document_chunks() doesn't return the stored vector (not needed by callers).
                embedding=[],
                metadata={**row.get("metadata", {}), "document_name": row["document_name"]},
            )
            for row in (response.data or [])
        ]


class VectorStore:
    """Facade that prefers Supabase/pgvector and falls back to a local JSON store."""

    def __init__(self, path: Path, settings: Settings | None = None):
        self.path = path
        self.settings = settings
        self.local = LocalVectorStore(path)
        self._supabase: SupabaseVectorStore | None = None

    def _remote(self) -> SupabaseVectorStore | None:
        if not self.settings or not self.settings.has_supabase:
            return None
        if self._supabase is None:
            self._supabase = SupabaseVectorStore(self.settings)
        return self._supabase

    def add_chunks(
        self,
        document_name: str,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> None:
        remote = self._remote()
        if remote:
            try:
                remote.add_chunks(document_name, chunks, embeddings)
                return
            except Exception:
                # Keep the app usable even if Supabase is unreachable or misconfigured.
                pass
        self.local.add_chunks(document_name, chunks, embeddings)

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_name: str | None = None,
    ) -> list[VectorRecord]:
        remote = self._remote()
        if remote:
            try:
                return remote.similarity_search(query_embedding, top_k, document_name)
            except Exception:
                pass
        return self.local.similarity_search(query_embedding, top_k, document_name)
