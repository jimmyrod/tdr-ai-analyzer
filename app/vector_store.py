from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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


class VectorStore:
    """Facade prepared for ChromaDB, with JSON fallback for portability."""

    def __init__(self, path: Path):
        self.path = path
        self.backend = LocalVectorStore(path)

    def add_chunks(
        self,
        document_name: str,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> None:
        self.backend.add_chunks(document_name, chunks, embeddings)

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_name: str | None = None,
    ) -> list[VectorRecord]:
        return self.backend.similarity_search(query_embedding, top_k, document_name)
