from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import ensure_directories, get_settings
from app.offline_evaluation import (
    load_evaluation_cases,
    run_evaluation,
    summarize_evaluation,
    write_evaluation_outputs,
)
from app.rag_engine import RAGEngine


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ejecuta la evaluacion offline del modelo sobre casos etiquetados."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "data" / "evaluations" / "evaluation_cases.example.json",
        help="Ruta al JSON con casos etiquetados.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "evaluation",
        help="Directorio donde se guardan CSV, JSON y Markdown.",
    )
    parser.add_argument(
        "--prefix",
        default="evaluacion_modelo",
        help="Prefijo de los archivos generados.",
    )
    args = parser.parse_args()

    settings = get_settings()
    ensure_directories(settings)
    cases = load_evaluation_cases(args.cases)
    engine = RAGEngine(settings=settings)

    rows = run_evaluation(
        cases,
        analyze=lambda case: engine.analyze_document(case.document_name, case.text),
        manual_minutes=settings.manual_analysis_minutes,
    )
    summary = summarize_evaluation(rows)
    paths = write_evaluation_outputs(rows, summary, args.output_dir, prefix=args.prefix)

    overall = summary["overall"]
    print("Evaluacion completada")
    print(f"Casos: {overall.get('case_count', 0)}")
    print(f"F1 requisitos: {overall.get('f1_score', 0):.4f}")
    print(f"Exactitud clasificacion: {overall.get('exactitud_clasificacion', 0):.4f}")
    print(f"Coincidencia con experto: {overall.get('coincidencia_con_experto', 0):.4f}")
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
