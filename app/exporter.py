from __future__ import annotations

import json
import re
from pathlib import Path

from app.schemas import AnalysisResult


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("_") or "analisis"


def export_analysis_json(result: AnalysisResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{_slug(Path(result.nombre_documento).stem)}_analysis.json"
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def export_analysis_markdown(result: AnalysisResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{_slug(Path(result.nombre_documento).stem)}_analysis.md"
    req_lines = "\n".join(
        f"| {req.id} | {req.tipo} | {req.prioridad} | {req.descripcion} |"
        for req in result.requisitos_tecnicos
    )
    alternatives = "\n".join(f"- {item}" for item in result.alternativas) or "- Sin alternativas."
    missing = "\n".join(f"- {item}" for item in result.datos_faltantes_o_ambiguos) or "- No identificado."

    markdown = f"""# Análisis automático de TDR

**Documento:** {result.nombre_documento}

**Categoría tecnológica:** {result.categoria_tecnologica}

**Modo demo:** {"Sí" if result.modo_demo else "No"}

## Resumen general
{result.resumen_general}

## Objeto o necesidad principal
{result.objeto_requerimiento}

## Requisitos técnicos identificados
| ID | Tipo | Prioridad | Descripción |
| --- | --- | --- | --- |
{req_lines}

## Productos o servicios esperados
{chr(10).join(f"- {item}" for item in result.productos_o_servicios_esperados) or "- No identificado."}

## Solución recomendada
**Nombre:** {result.solucion_recomendada.nombre}

**Categoría:** {result.solucion_recomendada.categoria}

**Nivel de confianza:** {result.solucion_recomendada.nivel_confianza}

### Justificación técnica
{result.solucion_recomendada.justificacion}

## Alternativas
{alternatives}

## Datos faltantes o ambiguos
{missing}

## Observaciones
{result.observaciones}
"""
    path.write_text(markdown, encoding="utf-8")
    return path


def export_analysis_pdf(result: AnalysisResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{_slug(Path(result.nombre_documento).stem)}_analysis.pdf"
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        doc = SimpleDocTemplate(str(path), pagesize=LETTER)
        styles = getSampleStyleSheet()
        story = [
            Paragraph("Análisis automático de TDR", styles["Title"]),
            Paragraph(f"Documento: {result.nombre_documento}", styles["Normal"]),
            Paragraph(f"Categoría: {result.categoria_tecnologica}", styles["Normal"]),
            Spacer(1, 12),
            Paragraph("Resumen general", styles["Heading2"]),
            Paragraph(result.resumen_general, styles["BodyText"]),
            Paragraph("Solución recomendada", styles["Heading2"]),
            Paragraph(result.solucion_recomendada.nombre, styles["Heading3"]),
            Paragraph(result.solucion_recomendada.justificacion, styles["BodyText"]),
            Paragraph("Advertencia", styles["Heading2"]),
            Paragraph(
                "La recomendación generada por IA es preliminar y no reemplaza la revisión de un especialista técnico.",
                styles["BodyText"],
            ),
        ]
        doc.build(story)
    except Exception:
        _write_minimal_pdf(
            path,
            [
                "Analisis automatico de TDR",
                f"Documento: {result.nombre_documento}",
                f"Categoria: {result.categoria_tecnologica}",
                f"Solucion recomendada: {result.solucion_recomendada.nombre}",
                "La recomendacion generada por IA es preliminar.",
            ],
        )
    return path


def _write_minimal_pdf(path: Path, lines: list[str]) -> None:
    escaped_lines = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in lines]
    text_commands = ["BT /F1 12 Tf 72 740 Td"]
    for index, line in enumerate(escaped_lines):
        if index:
            text_commands.append("0 -18 Td")
        text_commands.append(f"({line}) Tj")
    text_commands.append("ET")
    stream = "\n".join(text_commands).encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj_id, obj in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{obj_id} 0 obj\n".encode("ascii"))
        content.extend(obj)
        content.extend(b"\nendobj\n")
    xref_offset = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    content.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode(
            "ascii"
        )
    )
    path.write_bytes(bytes(content))
