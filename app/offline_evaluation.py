from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable
from xml.sax.saxutils import escape

from app.document_loader import load_document
from app.evaluator import calculate_metrics
from app.schemas import AnalysisResult


METRIC_FIELDS = [
    "precision_requisitos",
    "recall_requisitos",
    "f1_score",
    "exactitud_clasificacion",
    "coincidencia_con_experto",
    "tiempo_analisis_segundos",
    "reduccion_tiempo_estimada",
]


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    document_name: str
    text: str
    expected_category: str
    expected_solution: str
    expected_requirements: list[str]
    groups: dict[str, str]
    split: str = "validation"


@dataclass(frozen=True)
class EvaluationRow:
    case_id: str
    document_name: str
    split: str
    expected_category: str
    predicted_category: str
    expected_solution: str
    predicted_solution: str
    requirement_count_expected: int
    requirement_count_predicted: int
    precision_requisitos: float
    recall_requisitos: float
    f1_score: float
    exactitud_clasificacion: float
    coincidencia_con_experto: float
    tiempo_analisis_segundos: float
    reduccion_tiempo_estimada: float
    groups: dict[str, str]


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    cases_path = Path(path)
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    return load_evaluation_cases_payload(payload, base_dir=cases_path.parent)


def load_evaluation_cases_payload(payload: list[dict], base_dir: str | Path) -> list[EvaluationCase]:
    if not isinstance(payload, list):
        raise ValueError("El archivo de casos debe contener una lista JSON.")

    base_path = Path(base_dir)
    cases: list[EvaluationCase] = []
    for index, item in enumerate(payload, start=1):
        text = item.get("text", "").strip()
        document_path = item.get("document_path", "").strip()
        if not text and document_path:
            text = load_document(_resolve_document_path(document_path, base_path))
        if not text:
            raise ValueError(f"El caso {item.get('id', index)} no tiene text ni document_path.")

        expected_requirements = item.get("expected_requirements") or []
        if not expected_requirements:
            raise ValueError(f"El caso {item.get('id', index)} no tiene expected_requirements.")

        case_id = str(item.get("id") or f"case-{index:03d}")
        document_name = str(item.get("document_name") or f"{case_id}.txt")
        groups = {str(key): str(value) for key, value in (item.get("groups") or {}).items()}
        groups = _with_default_groups(
            groups,
            expected_category=str(item["expected_category"]),
            text=text,
        )
        cases.append(
            EvaluationCase(
                case_id=case_id,
                document_name=document_name,
                text=text,
                expected_category=str(item["expected_category"]),
                expected_solution=str(item["expected_solution"]),
                expected_requirements=[str(requirement) for requirement in expected_requirements],
                groups=groups,
                split=str(item.get("split") or "validation"),
            )
        )
    return cases


def run_evaluation(
    cases: Iterable[EvaluationCase],
    analyze: Callable[[EvaluationCase], AnalysisResult],
    manual_minutes: float = 60.0,
) -> list[EvaluationRow]:
    rows: list[EvaluationRow] = []
    for case in cases:
        result = analyze(case)
        rows.append(evaluate_analysis_result(case, result, manual_minutes=manual_minutes))
    return rows


def evaluate_analysis_result(
    case: EvaluationCase,
    result: AnalysisResult,
    manual_minutes: float = 60.0,
) -> EvaluationRow:
    metrics = calculate_metrics(
        extracted_requirements=[requirement.descripcion for requirement in result.requisitos_tecnicos],
        correct_requirements=case.expected_requirements,
        predicted_category=result.categoria_tecnologica,
        expert_category=case.expected_category,
        predicted_solution=result.solucion_recomendada.nombre,
        expert_solution=case.expected_solution,
        analysis_seconds=result.tiempo_analisis_segundos,
        manual_minutes=manual_minutes,
    )
    return EvaluationRow(
        case_id=case.case_id,
        document_name=case.document_name,
        split=case.split,
        expected_category=case.expected_category,
        predicted_category=result.categoria_tecnologica,
        expected_solution=case.expected_solution,
        predicted_solution=result.solucion_recomendada.nombre,
        requirement_count_expected=len(case.expected_requirements),
        requirement_count_predicted=len(result.requisitos_tecnicos),
        precision_requisitos=metrics.precision_requisitos,
        recall_requisitos=metrics.recall_requisitos,
        f1_score=metrics.f1_score,
        exactitud_clasificacion=metrics.exactitud_clasificacion,
        coincidencia_con_experto=metrics.coincidencia_con_experto,
        tiempo_analisis_segundos=metrics.tiempo_analisis_segundos,
        reduccion_tiempo_estimada=metrics.reduccion_tiempo_estimada,
        groups=case.groups,
    )


def summarize_evaluation(rows: list[EvaluationRow]) -> dict:
    overall = _aggregate(rows)
    return {
        "overall": overall,
        "benchmarking": _benchmark_summary(rows, overall),
        "by_split": _summarize_by_split(rows),
        "fairness": _fairness_summary(rows),
        "generalization": _generalization_summary(rows),
    }


def write_evaluation_outputs(
    rows: list[EvaluationRow],
    summary: dict,
    output_dir: str | Path,
    prefix: str = "evaluacion_modelo",
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows_path = output_path / f"{prefix}_casos.csv"
    summary_path = output_path / f"{prefix}_resumen.json"
    report_path = output_path / f"{prefix}_informe.md"
    pdf_path = output_path / f"{prefix}_informe.pdf"

    _write_rows_csv(rows, rows_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_markdown_report(rows, summary), encoding="utf-8")
    _write_pdf_report(rows, summary, pdf_path)
    return {
        "rows_csv": rows_path,
        "summary_json": summary_path,
        "report_markdown": report_path,
        "report_pdf": pdf_path,
    }


def _resolve_document_path(document_path: str, base_dir: Path) -> Path:
    path = Path(document_path)
    if path.is_absolute():
        return path
    candidates = [base_dir / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _with_default_groups(groups: dict[str, str], expected_category: str, text: str) -> dict[str, str]:
    defaults = dict(groups)
    defaults.setdefault("category", expected_category)
    defaults.setdefault("length", _length_bucket(text))
    defaults.setdefault("provider", _provider_bucket(text))
    return defaults


def _length_bucket(text: str) -> str:
    words = len(text.split())
    if words < 180:
        return "corto"
    if words < 650:
        return "medio"
    return "largo"


def _provider_bucket(text: str) -> str:
    normalized = text.lower()
    providers = ["microsoft", "google", "fortinet", "sophos", "kaspersky", "eset", "adobe", "zoom"]
    for provider in providers:
        if provider in normalized:
            return provider.title()
    return "generico"


def _aggregate(rows: list[EvaluationRow]) -> dict[str, float]:
    if not rows:
        return {"case_count": 0}
    values = {"case_count": len(rows)}
    for field in METRIC_FIELDS:
        values[field] = round(mean(getattr(row, field) for row in rows), 4)
    return values


def _benchmark_summary(rows: list[EvaluationRow], overall: dict) -> dict:
    if not rows:
        return {"modelo_actual": overall}

    categories = {row.expected_category for row in rows}
    solutions = {row.expected_solution for row in rows}
    majority_category_count = max(
        sum(1 for row in rows if row.expected_category == category) for category in categories
    )
    return {
        "modelo_actual": overall,
        "baseline_aleatorio": {
            "case_count": len(rows),
            "exactitud_clasificacion": round(1 / max(len(categories), 1), 4),
            "coincidencia_con_experto": round(1 / max(len(solutions), 1), 4),
            "nota": "Estimacion teorica si se elige una clase o solucion al azar.",
        },
        "baseline_mayoritario_categoria": {
            "case_count": len(rows),
            "exactitud_clasificacion": round(majority_category_count / len(rows), 4),
            "nota": "Estimacion si siempre se predice la categoria mas frecuente del dataset.",
        },
    }


def _summarize_by_split(rows: list[EvaluationRow]) -> dict:
    return {
        split: _aggregate(split_rows)
        for split, split_rows in _group_by(rows, lambda row: row.split).items()
    }


def _fairness_summary(rows: list[EvaluationRow]) -> dict:
    group_keys = sorted({key for row in rows for key in row.groups})
    summary = {}
    for key in group_keys:
        grouped = _group_by(rows, lambda row, group_key=key: row.groups.get(group_key, "sin_grupo"))
        group_metrics = {group: _aggregate(group_rows) for group, group_rows in grouped.items()}
        summary[key] = {
            "groups": group_metrics,
            "f1_score_gap": _metric_gap(group_metrics, "f1_score"),
            "exactitud_clasificacion_gap": _metric_gap(group_metrics, "exactitud_clasificacion"),
            "coincidencia_con_experto_gap": _metric_gap(group_metrics, "coincidencia_con_experto"),
        }
    return summary


def _generalization_summary(rows: list[EvaluationRow]) -> dict:
    by_split = _summarize_by_split(rows)
    train = by_split.get("train")
    validation = by_split.get("validation")
    if not train or not validation:
        return {
            "estado": "insuficiente",
            "nota": "Use split='train' y split='validation' para estimar brechas de generalizacion.",
        }
    return {
        "estado": "calculado",
        "f1_train_validation_gap": round(train.get("f1_score", 0.0) - validation.get("f1_score", 0.0), 4),
        "accuracy_train_validation_gap": round(
            train.get("exactitud_clasificacion", 0.0) - validation.get("exactitud_clasificacion", 0.0),
            4,
        ),
        "interpretacion": _generalization_label(train, validation),
    }


def _generalization_label(train: dict, validation: dict) -> str:
    train_f1 = train.get("f1_score", 0.0)
    validation_f1 = validation.get("f1_score", 0.0)
    gap = train_f1 - validation_f1
    if train_f1 < 0.55 and validation_f1 < 0.55:
        return "posible_underfitting"
    if gap > 0.2:
        return "posible_overfitting"
    return "ajuste_razonable"


def _group_by(rows: list[EvaluationRow], key_fn: Callable[[EvaluationRow], str]) -> dict[str, list[EvaluationRow]]:
    groups: dict[str, list[EvaluationRow]] = {}
    for row in rows:
        groups.setdefault(str(key_fn(row)), []).append(row)
    return groups


def _metric_gap(group_metrics: dict[str, dict], metric: str) -> float:
    values = [metrics.get(metric, 0.0) for metrics in group_metrics.values() if metrics.get("case_count", 0)]
    if not values:
        return 0.0
    return round(max(values) - min(values), 4)


def _write_rows_csv(rows: list[EvaluationRow], path: Path) -> None:
    fieldnames = [
        "case_id",
        "document_name",
        "split",
        "expected_category",
        "predicted_category",
        "expected_solution",
        "predicted_solution",
        "requirement_count_expected",
        "requirement_count_predicted",
        *METRIC_FIELDS,
        "groups_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            payload["groups_json"] = json.dumps(payload.pop("groups"), ensure_ascii=False)
            writer.writerow(payload)


def _write_pdf_report(rows: list[EvaluationRow], summary: dict, path: Path) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except Exception:
        from app.exporter import _write_minimal_pdf

        _write_minimal_pdf(
            path,
            [
                "Evaluacion y optimizacion del modelo IA",
                "Benchmarking comparativo, analisis de sesgo y diagnostico de generalizacion",
                f"Casos evaluados: {summary.get('overall', {}).get('case_count', 0)}",
            ],
        )
        return

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0b3f44"),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#0b6b6f"),
            spaceBefore=12,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallCell",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=8.5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverMeta",
            parent=styles["BodyText"],
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#26383d"),
        )
    )

    doc = SimpleDocTemplate(
        str(path),
        pagesize=LETTER,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.6 * inch,
        title="Evaluacion y optimizacion del modelo IA",
        author="TDR AI Analyzer",
    )

    story = [
        Spacer(1, 0.6 * inch),
        Paragraph("Evaluacion y optimizacion del modelo IA", styles["ReportTitle"]),
        Paragraph(
            "Benchmarking comparativo, analisis de sesgo y diagnostico de overfitting / underfitting",
            styles["CoverMeta"],
        ),
        Spacer(1, 0.28 * inch),
        Paragraph("Proyecto: TDR AI Analyzer", styles["CoverMeta"]),
        Paragraph("Modelo IA para analisis de terminos de referencia tecnologicos", styles["CoverMeta"]),
        Paragraph(f"Fecha de generacion: {date.today().isoformat()}", styles["CoverMeta"]),
        Spacer(1, 0.35 * inch),
        _metrics_table(summary.get("overall", {}), styles),
        PageBreak(),
        Paragraph("1. Resumen ejecutivo", styles["SectionHeading"]),
        Paragraph(
            "Este informe consolida los resultados calculados desde el portal de evaluacion del modelo. "
            "Incluye indicadores de extraccion de requisitos, clasificacion tecnologica, coincidencia con "
            "criterio experto, benchmarking, fairness por grupos y diagnostico train vs validation.",
            styles["BodyText"],
        ),
        Paragraph("2. Benchmarking comparativo", styles["SectionHeading"]),
        _benchmark_pdf_table(summary, styles),
        Paragraph("3. Diagnostico train vs validation", styles["SectionHeading"]),
        _split_pdf_table(summary, styles),
        Paragraph(
            f"Interpretacion: {summary.get('generalization', {}).get('interpretacion', summary.get('generalization', {}).get('estado', 'sin datos'))}.",
            styles["BodyText"],
        ),
        Paragraph("4. Analisis de sesgo por grupos", styles["SectionHeading"]),
    ]

    fairness = summary.get("fairness", {})
    if fairness:
        for group_key, payload in fairness.items():
            story.append(Paragraph(f"Grupo: {group_key}", styles["Heading3"]))
            story.append(
                Paragraph(
                    "Brecha F1: "
                    f"{payload.get('f1_score_gap', 0):.4f} | "
                    "Brecha clasificacion: "
                    f"{payload.get('exactitud_clasificacion_gap', 0):.4f} | "
                    "Brecha solucion: "
                    f"{payload.get('coincidencia_con_experto_gap', 0):.4f}",
                    styles["BodyText"],
                )
            )
            story.append(_fairness_pdf_table(payload, styles))
            story.append(Spacer(1, 8))
    else:
        story.append(Paragraph("No existen grupos suficientes para analizar sesgo.", styles["BodyText"]))

    story.extend(
        [
            Paragraph("5. Resultados por caso", styles["SectionHeading"]),
            _cases_pdf_table(rows, styles),
            Paragraph("6. Conclusiones para optimizacion", styles["SectionHeading"]),
            Paragraph(
                "Los resultados deben usarse para priorizar mejoras donde exista menor F1 de requisitos, "
                "mayor brecha entre grupos o diferencia relevante entre entrenamiento y validacion. "
                "La recomendacion generada por IA sigue siendo preliminar y requiere validacion humana.",
                styles["BodyText"],
            ),
        ]
    )

    doc.build(story, onFirstPage=_draw_pdf_footer, onLaterPages=_draw_pdf_footer)


def _metrics_table(overall: dict, styles) -> object:
    data = [
        ["Indicador", "Valor"],
        ["Casos evaluados", str(overall.get("case_count", 0))],
        ["F1 promedio de requisitos", _metric_value(overall.get("f1_score", 0))],
        ["Exactitud de clasificacion", _metric_value(overall.get("exactitud_clasificacion", 0))],
        ["Coincidencia con experto", _metric_value(overall.get("coincidencia_con_experto", 0))],
        ["Tiempo promedio de analisis", f"{overall.get('tiempo_analisis_segundos', 0):.2f}s"],
    ]
    return _pdf_table(data, styles, col_widths=[260, 160])


def _benchmark_pdf_table(summary: dict, styles) -> object:
    data = [["Modelo", "Casos", "F1 req.", "Exactitud", "Solucion", "Nota"]]
    for name, metrics in summary.get("benchmarking", {}).items():
        data.append(
            [
                name,
                str(metrics.get("case_count", 0)),
                _metric_value(metrics.get("f1_score", "")),
                _metric_value(metrics.get("exactitud_clasificacion", "")),
                _metric_value(metrics.get("coincidencia_con_experto", "")),
                metrics.get("nota", ""),
            ]
        )
    return _pdf_table(data, styles, col_widths=[95, 38, 50, 55, 55, 190])


def _split_pdf_table(summary: dict, styles) -> object:
    data = [["Split", "Casos", "Precision", "Recall", "F1 req.", "Exactitud", "Solucion"]]
    for split, metrics in summary.get("by_split", {}).items():
        data.append(
            [
                split,
                str(metrics.get("case_count", 0)),
                _metric_value(metrics.get("precision_requisitos", 0)),
                _metric_value(metrics.get("recall_requisitos", 0)),
                _metric_value(metrics.get("f1_score", 0)),
                _metric_value(metrics.get("exactitud_clasificacion", 0)),
                _metric_value(metrics.get("coincidencia_con_experto", 0)),
            ]
        )
    return _pdf_table(data, styles, col_widths=[70, 45, 65, 55, 60, 70, 70])


def _fairness_pdf_table(group_payload: dict, styles) -> object:
    data = [["Valor de grupo", "Casos", "F1", "Exactitud", "Solucion"]]
    for group, metrics in group_payload.get("groups", {}).items():
        data.append(
            [
                group,
                str(metrics.get("case_count", 0)),
                _metric_value(metrics.get("f1_score", 0)),
                _metric_value(metrics.get("exactitud_clasificacion", 0)),
                _metric_value(metrics.get("coincidencia_con_experto", 0)),
            ]
        )
    return _pdf_table(data, styles, col_widths=[170, 45, 60, 70, 70])


def _cases_pdf_table(rows: list[EvaluationRow], styles) -> object:
    data = [["Caso", "Split", "Categoria esperada", "Categoria predicha", "Solucion predicha", "F1"]]
    for row in rows:
        data.append(
            [
                row.case_id,
                row.split,
                row.expected_category,
                row.predicted_category,
                row.predicted_solution,
                _metric_value(row.f1_score),
            ]
        )
    return _pdf_table(data, styles, col_widths=[80, 50, 110, 110, 115, 40])


def _pdf_table(data: list[list[object]], styles, col_widths: list[int]) -> object:
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle

    wrapped = [
        [_pdf_cell(value, styles["SmallCell"]) if row_index else _pdf_cell(value, styles["SmallCell"], bold=True) for value in row]
        for row_index, row in enumerate(data)
    ]
    table = Table(wrapped, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b6b6f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c8d6d3")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#fbfdfc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _pdf_cell(value: object, style, bold: bool = False) -> object:
    from reportlab.platypus import Paragraph

    text = escape(str(value))
    if bold:
        text = f"<b>{text}</b>"
    return Paragraph(text, style)


def _metric_value(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _draw_pdf_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColorRGB(0.28, 0.35, 0.36)
    canvas.drawString(40, 24, "TDR AI Analyzer - Evaluacion y optimizacion del modelo IA")
    canvas.drawRightString(572, 24, f"Pagina {doc.page}")
    canvas.restoreState()


def _render_markdown_report(rows: list[EvaluationRow], summary: dict) -> str:
    overall = summary.get("overall", {})
    lines = [
        "# Evaluacion y optimizacion del modelo IA",
        "",
        "## Resumen ejecutivo",
        "",
        f"- Casos evaluados: {overall.get('case_count', 0)}",
        f"- F1 promedio de requisitos: {overall.get('f1_score', 0):.4f}",
        f"- Exactitud de clasificacion: {overall.get('exactitud_clasificacion', 0):.4f}",
        f"- Coincidencia con experto: {overall.get('coincidencia_con_experto', 0):.4f}",
        f"- Tiempo promedio de analisis: {overall.get('tiempo_analisis_segundos', 0):.2f}s",
        "",
        "## Benchmarking",
        "",
        "| Modelo | Casos | Exactitud categoria | Coincidencia solucion | Nota |",
        "|---|---:|---:|---:|---|",
    ]
    for name, metrics in summary.get("benchmarking", {}).items():
        lines.append(
            "| "
            f"{name} | "
            f"{metrics.get('case_count', 0)} | "
            f"{metrics.get('exactitud_clasificacion', 0):.4f} | "
            f"{metrics.get('coincidencia_con_experto', 0):.4f} | "
            f"{metrics.get('nota', '')} |"
        )

    lines.extend(
        [
            "",
            "## Diagnostico train vs validation",
            "",
            "| Split | Casos | F1 requisitos | Exactitud categoria | Coincidencia solucion |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split, metrics in summary.get("by_split", {}).items():
        lines.append(
            f"| {split} | {metrics.get('case_count', 0)} | "
            f"{metrics.get('f1_score', 0):.4f} | "
            f"{metrics.get('exactitud_clasificacion', 0):.4f} | "
            f"{metrics.get('coincidencia_con_experto', 0):.4f} |"
        )

    generalization = summary.get("generalization", {})
    lines.extend(
        [
            "",
            f"Interpretacion: `{generalization.get('interpretacion', generalization.get('estado', 'sin_datos'))}`.",
            "",
            "## Analisis de sesgo por grupos",
            "",
        ]
    )
    for group_key, payload in summary.get("fairness", {}).items():
        lines.extend(
            [
                f"### Grupo: {group_key}",
                "",
                f"- Brecha F1: {payload.get('f1_score_gap', 0):.4f}",
                f"- Brecha exactitud categoria: {payload.get('exactitud_clasificacion_gap', 0):.4f}",
                f"- Brecha coincidencia solucion: {payload.get('coincidencia_con_experto_gap', 0):.4f}",
                "",
                "| Valor de grupo | Casos | F1 | Exactitud categoria | Coincidencia solucion |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for group_value, metrics in payload.get("groups", {}).items():
            lines.append(
                f"| {group_value} | {metrics.get('case_count', 0)} | "
                f"{metrics.get('f1_score', 0):.4f} | "
                f"{metrics.get('exactitud_clasificacion', 0):.4f} | "
                f"{metrics.get('coincidencia_con_experto', 0):.4f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Resultados por caso",
            "",
            "| Caso | Split | Categoria esperada | Categoria predicha | Solucion esperada | Solucion predicha | F1 |",
            "|---|---|---|---|---|---|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.case_id} | {row.split} | {row.expected_category} | {row.predicted_category} | "
            f"{row.expected_solution} | {row.predicted_solution} | {row.f1_score:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)
