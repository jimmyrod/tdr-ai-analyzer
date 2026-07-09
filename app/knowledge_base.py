from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings, get_settings
from app.schemas import Solution
from app.text_cleaner import normalize_for_matching

SOLUTIONS_TABLE = "solutions"
SOLUTIONS_FIELDS = (
    "id",
    "nombre",
    "categoria",
    "descripcion",
    "caracteristicas_principales",
    "requisitos_que_cubre",
    "restricciones",
    "modalidad",
    "observaciones",
)


class KnowledgeBase:
    def __init__(self, solutions: list[Solution]):
        self._solutions = solutions

    @classmethod
    def load(cls, path: Path | None = None, settings: Settings | None = None) -> "KnowledgeBase":
        settings = settings or get_settings()
        path = path or settings.knowledge_base_path
        if settings.has_supabase:
            try:
                return cls._load_from_supabase(settings)
            except Exception:
                # Keep the app usable even if Supabase is unreachable or misconfigured.
                pass
        if not path.exists():
            return cls([])
        records = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_records(records)

    @classmethod
    def _load_from_supabase(cls, settings: Settings) -> "KnowledgeBase":
        from supabase import create_client

        client = create_client(settings.supabase_url, settings.supabase_secret_key)
        response = client.table(SOLUTIONS_TABLE).select(",".join(SOLUTIONS_FIELDS)).execute()
        records = [
            {
                **row,
                "caracteristicas_principales": row.get("caracteristicas_principales") or [],
                "requisitos_que_cubre": row.get("requisitos_que_cubre") or [],
                "restricciones": row.get("restricciones") or [],
            }
            for row in (response.data or [])
        ]
        return cls.from_records(records)

    @classmethod
    def from_records(cls, records: list[dict]) -> "KnowledgeBase":
        return cls([Solution(**record) for record in records])

    def all(self) -> list[Solution]:
        return list(self._solutions)

    def by_category(self, category: str) -> list[Solution]:
        wanted = normalize_for_matching(category)
        return [
            solution
            for solution in self._solutions
            if normalize_for_matching(solution.categoria) == wanted
        ]

    def search(self, query: str, category: str | None = None, limit: int = 5) -> list[Solution]:
        query_terms = set(normalize_for_matching(query).split())
        candidates = self.by_category(category) if category else self.all()
        scored: list[tuple[float, Solution]] = []

        for solution in candidates:
            haystack = normalize_for_matching(
                " ".join(
                    [
                        solution.nombre,
                        solution.categoria,
                        solution.descripcion,
                        " ".join(solution.caracteristicas_principales),
                        " ".join(solution.requisitos_que_cubre),
                        solution.modalidad,
                    ]
                )
            )
            haystack_terms = set(haystack.split())
            overlap = len(query_terms & haystack_terms)
            if category and normalize_for_matching(solution.categoria) == normalize_for_matching(category):
                overlap += 3
            if overlap > 0:
                scored.append((overlap, solution))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [solution for _, solution in scored[:limit]]
