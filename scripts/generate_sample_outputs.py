from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import ensure_directories, get_settings
from app.document_loader import load_document
from app.exporter import export_analysis_json, export_analysis_markdown, export_analysis_pdf
from app.rag_engine import RAGEngine


def main() -> None:
    settings = get_settings()
    ensure_directories(settings)
    example_path = settings.project_root / "data" / "examples" / "tdr_backup_cloud.txt"
    text = load_document(example_path)
    engine = RAGEngine(settings=settings)
    result = engine.analyze_document(example_path.name, text)
    export_analysis_json(result, settings.outputs_json_dir)
    export_analysis_markdown(result, settings.outputs_markdown_dir)
    export_analysis_pdf(result, settings.outputs_pdf_dir)
    _write_sample_pdf_from_text(
        settings.project_root / "data" / "examples" / "tdr_backup_cloud.pdf",
        example_path.read_text(encoding="utf-8"),
    )


def _write_sample_pdf_from_text(path: Path, text: str) -> None:
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        doc = SimpleDocTemplate(str(path), pagesize=LETTER)
        styles = getSampleStyleSheet()
        story = [Paragraph("TDR simulado - Backup cloud", styles["Title"]), Spacer(1, 12)]
        for paragraph in text.splitlines():
            if paragraph.strip():
                story.append(Paragraph(paragraph, styles["BodyText"]))
                story.append(Spacer(1, 6))
        doc.build(story)
    except Exception:
        from app.exporter import _write_minimal_pdf

        lines = [line for line in text.splitlines() if line.strip()][:20]
        _write_minimal_pdf(path, lines)


if __name__ == "__main__":
    main()
