import json

from app.evaluator import calculate_metrics
from app.exporter import export_analysis_json, export_analysis_markdown
from app.schemas import AnalysisResult, RecommendedSolution, Requirement


def _sample_result() -> AnalysisResult:
    return AnalysisResult(
        nombre_documento="tdr_demo.txt",
        resumen_general="Resumen breve.",
        objeto_requerimiento="Contratar backup cloud.",
        categoria_tecnologica="Backup y recuperación",
        requisitos_tecnicos=[
            Requirement(
                id="REQ-001",
                descripcion="Debe realizar copias automaticas.",
                tipo="tecnico",
                prioridad="alta",
                fragmento_fuente="copias automaticas diarias",
            )
        ],
        productos_o_servicios_esperados=["Servicio de backup"],
        solucion_recomendada=RecommendedSolution(
            nombre="Backup cloud",
            categoria="Backup y recuperación",
            justificacion="Cubre copias automaticas y recuperacion.",
            nivel_confianza="alta",
        ),
        alternativas=[],
        datos_faltantes_o_ambiguos=["No se especifica retencion."],
        observaciones="Analisis preliminar.",
    )


def test_evaluator_calculates_precision_recall_f1_and_matches_expert():
    metrics = calculate_metrics(
        extracted_requirements=["backup diario", "recuperacion"],
        correct_requirements=["backup diario", "recuperacion", "retencion"],
        predicted_category="Backup y recuperación",
        expert_category="Backup y recuperación",
        predicted_solution="Backup cloud",
        expert_solution="Backup cloud",
        analysis_seconds=120,
        manual_minutes=60,
    )

    assert metrics.precision_requisitos == 1
    assert round(metrics.recall_requisitos, 2) == 0.67
    assert round(metrics.f1_score, 2) == 0.8
    assert metrics.exactitud_clasificacion == 1
    assert metrics.coincidencia_con_experto == 1
    assert metrics.reduccion_tiempo_estimada >= 0.9


def test_exporter_writes_json_and_markdown(tmp_path):
    result = _sample_result()

    json_path = export_analysis_json(result, tmp_path)
    md_path = export_analysis_markdown(result, tmp_path)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = md_path.read_text(encoding="utf-8")

    assert data["nombre_documento"] == "tdr_demo.txt"
    assert "## Solución recomendada" in markdown
    assert "Backup cloud" in markdown
