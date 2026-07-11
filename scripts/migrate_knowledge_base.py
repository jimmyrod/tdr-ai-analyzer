from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.embeddings import EmbeddingProvider
from app.knowledge_base import SOLUTIONS_TABLE


def _solution_text(record: dict) -> str:
    return " ".join(
        [
            record.get("nombre", ""),
            record.get("categoria", ""),
            record.get("descripcion", ""),
            " ".join(record.get("caracteristicas_principales", [])),
            " ".join(record.get("requisitos_que_cubre", [])),
            record.get("modalidad", ""),
        ]
    )


def main() -> None:
    settings = get_settings()
    if not settings.has_supabase:
        raise SystemExit("SUPABASE_URL / SUPABASE_SECRET_KEY no configurados en .env")

    from supabase import create_client

    records = json.loads(settings.knowledge_base_path.read_text(encoding="utf-8"))
    embedder = EmbeddingProvider(settings)
    embeddings = embedder.embed_texts([_solution_text(record) for record in records])

    rows = [
        {
            "id": record["id"],
            "nombre": record["nombre"],
            "categoria": record["categoria"],
            "descripcion": record["descripcion"],
            "caracteristicas_principales": record.get("caracteristicas_principales", []),
            "requisitos_que_cubre": record.get("requisitos_que_cubre", []),
            "restricciones": record.get("restricciones", []),
            "modalidad": record.get("modalidad", ""),
            "observaciones": record.get("observaciones", ""),
            "embedding": embedding,
        }
        for record, embedding in zip(records, embeddings)
    ]

    client = create_client(settings.supabase_url, settings.supabase_secret_key)
    client.table(SOLUTIONS_TABLE).upsert(rows).execute()
    print(f"{len(rows)} soluciones sincronizadas en Supabase (tabla '{SOLUTIONS_TABLE}').")


if __name__ == "__main__":
    main()
