import json

from pypdf import PdfReader

from app.offline_evaluation import (
    EvaluationCase,
    evaluate_analysis_result,
    load_evaluation_cases,
    load_evaluation_cases_payload,
    summarize_evaluation,
    write_evaluation_outputs,
)
from app.schemas import AnalysisResult, RecommendedSolution, Requirement


def _analysis_result(
    category: str = "Backup y recuperación",
    solution: str = "Backup cloud",
    requirements: list[str] | None = None,
) -> AnalysisResult:
    return AnalysisResult(
        nombre_documento="case-001.txt",
        resumen_general="Resumen.",
        objeto_requerimiento="Contratar servicio de backup cloud.",
        categoria_tecnologica=category,
        requisitos_tecnicos=[
            Requirement(
                id=f"REQ-{index:03d}",
                descripcion=requirement,
                tipo="tecnico",
                prioridad="alta",
                fragmento_fuente=requirement,
            )
            for index, requirement in enumerate(
                requirements or ["backup diario", "retención configurable"],
                start=1,
            )
        ],
        productos_o_servicios_esperados=["Servicio de backup"],
        solucion_recomendada=RecommendedSolution(
            nombre=solution,
            categoria=category,
            justificacion="Coincide con requisitos de respaldo.",
            nivel_confianza="alta",
        ),
        alternativas=["VPS administrado"],
        datos_faltantes_o_ambiguos=[],
        observaciones="Evaluación de prueba.",
        tiempo_analisis_segundos=30.0,
    )


def test_load_evaluation_cases_reads_embedded_text_and_document_path(tmp_path):
    document_path = tmp_path / "tdr.txt"
    document_path.write_text("Texto del TDR desde archivo.", encoding="utf-8")
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            [
                {
                    "id": "embedded",
                    "document_name": "embedded.txt",
                    "text": "Texto embebido.",
                    "expected_category": "Business Intelligence",
                    "expected_solution": "Microsoft Power BI Pro",
                    "expected_requirements": ["dashboard"],
                    "groups": {"provider": "Microsoft"},
                    "split": "validation",
                },
                {
                    "id": "file",
                    "document_name": "file.txt",
                    "document_path": str(document_path),
                    "expected_category": "Backup y recuperación",
                    "expected_solution": "Backup cloud",
                    "expected_requirements": ["backup diario"],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cases = load_evaluation_cases(cases_path)

    assert [case.case_id for case in cases] == ["embedded", "file"]
    assert cases[0].text == "Texto embebido."
    assert cases[0].groups["provider"] == "Microsoft"
    assert cases[0].split == "validation"
    assert cases[1].text == "Texto del TDR desde archivo."


def test_load_evaluation_cases_payload_resolves_relative_document_paths_from_base_dir(tmp_path):
    document_path = tmp_path / "documents" / "tdr.txt"
    document_path.parent.mkdir()
    document_path.write_text("Texto cargado desde ruta relativa.", encoding="utf-8")

    cases = load_evaluation_cases_payload(
        [
            {
                "id": "relative-file",
                "document_name": "tdr.txt",
                "document_path": "documents/tdr.txt",
                "expected_category": "Backup y recuperación",
                "expected_solution": "Backup cloud",
                "expected_requirements": ["backup diario"],
            }
        ],
        base_dir=tmp_path,
    )

    assert cases == [
        EvaluationCase(
            case_id="relative-file",
            document_name="tdr.txt",
            text="Texto cargado desde ruta relativa.",
            expected_category="Backup y recuperación",
            expected_solution="Backup cloud",
            expected_requirements=["backup diario"],
            groups={
                "category": "Backup y recuperación",
                "length": "corto",
                "provider": "generico",
            },
            split="validation",
        )
    ]


def test_evaluate_analysis_result_returns_metrics_and_group_metadata():
    case = EvaluationCase(
        case_id="case-001",
        document_name="case-001.txt",
        text="Debe incluir backup diario y retención configurable.",
        expected_category="Backup y recuperación",
        expected_solution="Backup cloud",
        expected_requirements=["backup diario", "retención configurable", "restauración"],
        groups={"category": "Backup", "length": "corto", "provider": "genérico"},
        split="validation",
    )

    row = evaluate_analysis_result(case, _analysis_result())

    assert row.case_id == "case-001"
    assert row.precision_requisitos == 1.0
    assert round(row.recall_requisitos, 2) == 0.67
    assert round(row.f1_score, 2) == 0.8
    assert row.exactitud_clasificacion == 1.0
    assert row.coincidencia_con_experto == 1.0
    assert row.groups["provider"] == "genérico"
    assert row.predicted_solution == "Backup cloud"


def test_summarize_evaluation_computes_overall_baselines_and_group_gaps():
    good_case = EvaluationCase(
        case_id="good",
        document_name="good.txt",
        text="Debe incluir backup diario.",
        expected_category="Backup y recuperación",
        expected_solution="Backup cloud",
        expected_requirements=["backup diario"],
        groups={"category": "Backup", "provider": "genérico"},
        split="train",
    )
    weak_case = EvaluationCase(
        case_id="weak",
        document_name="weak.txt",
        text="Debe incluir dashboard.",
        expected_category="Business Intelligence",
        expected_solution="Microsoft Power BI Pro",
        expected_requirements=["dashboard"],
        groups={"category": "BI", "provider": "Microsoft"},
        split="validation",
    )
    rows = [
        evaluate_analysis_result(good_case, _analysis_result()),
        evaluate_analysis_result(
            weak_case,
            _analysis_result(
                category="Licenciamiento de software",
                solution="Microsoft 365 Business Standard",
                requirements=["correo corporativo"],
            ),
        ),
    ]

    summary = summarize_evaluation(rows)

    assert summary["overall"]["case_count"] == 2
    assert summary["overall"]["exactitud_clasificacion"] == 0.5
    assert summary["benchmarking"]["modelo_actual"]["case_count"] == 2
    assert "baseline_aleatorio" in summary["benchmarking"]
    assert summary["by_split"]["train"]["exactitud_clasificacion"] == 1.0
    assert summary["by_split"]["validation"]["exactitud_clasificacion"] == 0.0
    assert summary["fairness"]["category"]["f1_score_gap"] > 0


def test_write_evaluation_outputs_generates_downloadable_pdf_report(tmp_path):
    case = EvaluationCase(
        case_id="pdf-case",
        document_name="pdf-case.txt",
        text="Debe incluir backup diario.",
        expected_category="Backup y recuperación",
        expected_solution="Backup cloud",
        expected_requirements=["backup diario"],
        groups={"category": "Backup", "provider": "genérico"},
        split="validation",
    )
    rows = [evaluate_analysis_result(case, _analysis_result())]
    summary = summarize_evaluation(rows)

    paths = write_evaluation_outputs(rows, summary, tmp_path, prefix="demo_pdf")

    pdf_path = paths["report_pdf"]
    assert pdf_path.name == "demo_pdf_informe.pdf"
    assert pdf_path.read_bytes().startswith(b"%PDF")
    reader = PdfReader(str(pdf_path))
    extracted_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "Evaluacion y optimizacion del modelo IA" in extracted_text
    assert "Benchmarking comparativo" in extracted_text
    assert "Backup cloud" in extracted_text
