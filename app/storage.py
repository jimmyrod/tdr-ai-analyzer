from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.config import get_settings
from app.schemas import AnalysisResult, EvaluationMetrics, ExpertEvaluation


class AnalysisStorage:
    def __init__(self, database_path: Path | None = None):
        self.database_path = database_path or get_settings().database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    recommended_solution TEXT NOT NULL,
                    analysis_seconds REAL NOT NULL,
                    demo_mode INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS expert_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    expert_solution TEXT NOT NULL,
                    expert_category TEXT NOT NULL,
                    recommendation_status TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (analysis_id) REFERENCES analyses(id)
                )
                """
            )

    def save_analysis(self, result: AnalysisResult) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO analyses (
                    document_name, category, recommended_solution, analysis_seconds,
                    demo_mode, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    result.nombre_documento,
                    result.categoria_tecnologica,
                    result.solucion_recomendada.nombre,
                    result.tiempo_analisis_segundos,
                    1 if result.modo_demo else 0,
                    json.dumps(result.to_dict(), ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)

    def save_evaluation(
        self,
        analysis_id: int | None,
        evaluation: ExpertEvaluation,
        metrics: EvaluationMetrics,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO expert_evaluations (
                    analysis_id, expert_solution, expert_category,
                    recommendation_status, metrics_json, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    evaluation.solucion_recomendada_experto,
                    evaluation.categoria_correcta,
                    evaluation.recomendacion_ia,
                    json.dumps(metrics.__dict__, ensure_ascii=False),
                    json.dumps(evaluation.__dict__, ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)

    def list_recent_analyses(self, limit: int = 10) -> list[dict]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT id, document_name, category, recommended_solution,
                       analysis_seconds, demo_mode, created_at
                FROM analyses
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]
