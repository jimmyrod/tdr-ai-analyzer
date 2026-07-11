from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.document_loader import load_document
from app.evaluation_dataset_builder import (
    build_case_from_text,
    extract_pdf_text,
    is_candidate_tdr_text,
)


DOWNLOADS_DIR = Path.home() / "Downloads"
OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluations" / "evaluation_cases_large_july_chatgpt_web.json"
SUMMARY_PATH = PROJECT_ROOT / "data" / "evaluations" / "evaluation_cases_large_july_chatgpt_web_summary.json"

EXCLUDE_NAME_TERMS = [
    "factura",
    "comprobante",
    "retencion",
    "retención",
    "cedula",
    "cédula",
    "certificado de experiencia",
    "evaluacion_modelo_informe",
    "proyecto_integrador",
    "analisis_comparativo",
    "analisis_planificacion",
    "workshop_smart",
    "preparacion_procesamiento",
    "eda_proyecto",
    "receipt-2026",
]

PREFERRED_REAL_TDR_NAMES = {
    "5847685.pdf",
    "6478252 (1).pdf",
    "7449707.pdf",
    "7690656.pdf",
    "7711346.pdf",
    "7745298.pdf",
    "7750087.pdf",
    "7753009.pdf",
    "7756352.pdf",
    "7758560.pdf",
    "7764471.pdf",
    "7765921.pdf",
    "7779018.pdf",
    "7786672.pdf",
    "7802985.pdf",
    "7813467.pdf",
}

PREFERRED_PROFORMA_PREFIXES = [
    "PR2607-2846",
    "PR2607-2854",
    "PR2607-2858",
    "PR2607-2863",
    "PR2607-2866",
    "PR2607-2867",
    "PR2607-2869",
    "PR2607-2876",
    "PR2607-2877",
    "PR2607-2878",
    "PR2607-2879",
    "PR2607-2880",
    "PR2607-2884",
    "PR2607-2885",
]


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sources = _collect_sources()
    cases = []
    for index, source in enumerate(sources, start=1):
        text = _load_text(source["path"])
        if not text.strip():
            continue
        case = build_case_from_text(
            text=text,
            document_name=source["path"].name,
            source_path=str(source["path"]),
            case_id=f"july-{index:03d}-{_slug(source['path'].stem)}",
            split=_split_for_index(index),
            document_type=source["document_type"],
        )
        cases.append(case)

    OUTPUT_PATH.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = _summary(cases, sources)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"cases={len(cases)}")
    print(f"dataset={OUTPUT_PATH}")
    print(f"summary={SUMMARY_PATH}")
    return 0


def _collect_sources() -> list[dict]:
    candidates: list[dict] = []
    seen_keys: set[str] = set()

    for path in _iter_pdf_sources():
        key = _canonical_key(path.name)
        if key in seen_keys:
            continue
        name = path.name
        lower_name = name.lower()
        if any(term in lower_name for term in EXCLUDE_NAME_TERMS):
            continue

        document_type = ""
        if name in PREFERRED_REAL_TDR_NAMES:
            document_type = "real_tdr"
        elif any(name.startswith(prefix) and "signed" not in lower_name for prefix in PREFERRED_PROFORMA_PREFIXES):
            document_type = "proforma"

        if not document_type:
            try:
                preview = extract_pdf_text(path)[:7000]
            except Exception:
                continue
            if is_candidate_tdr_text(preview, name):
                document_type = "real_tdr"
            else:
                continue

        seen_keys.add(key)
        candidates.append({"path": path, "document_type": document_type})

    for example_path in sorted((PROJECT_ROOT / "data" / "examples").glob("tdr_*.txt")):
        key = _canonical_key(example_path.name)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        candidates.append({"path": example_path, "document_type": "synthetic_example"})

    candidates.sort(key=lambda item: (item["document_type"], item["path"].name))
    return candidates


def _iter_pdf_sources() -> list[Path]:
    paths: list[Path] = []
    for folder in [PROJECT_ROOT / "data" / "uploads", DOWNLOADS_DIR]:
        if not folder.exists():
            continue
        paths.extend(path for path in folder.glob("*.pdf") if _is_july_or_project_upload(path))
    return sorted(paths, key=lambda path: (path.name.lower(), str(path.parent).lower()))


def _is_july_or_project_upload(path: Path) -> bool:
    if PROJECT_ROOT in path.parents:
        return True
    return datetime.fromtimestamp(path.stat().st_mtime) >= datetime(2026, 7, 1)


def _load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return extract_pdf_text(path)
    return load_document(path)


def _canonical_key(name: str) -> str:
    key = name.lower()
    key = key.replace("-signed", "")
    key = re.sub(r"\s*\(\d+\)", "", key)
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return key


def _split_for_index(index: int) -> str:
    mod = index % 10
    if mod in {1, 2, 3, 4, 5, 6}:
        return "train"
    if mod in {7, 8}:
        return "validation"
    return "test"


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")[:55] or "case"


def _summary(cases: list[dict], sources: list[dict]) -> dict:
    by_category: dict[str, int] = {}
    by_type: dict[str, int] = {}
    by_split: dict[str, int] = {}
    for case in cases:
        by_category[case["expected_category"]] = by_category.get(case["expected_category"], 0) + 1
        by_type[case["groups"]["document_type"]] = by_type.get(case["groups"]["document_type"], 0) + 1
        by_split[case["split"]] = by_split.get(case["split"], 0) + 1
    return {
        "case_count": len(cases),
        "source_count": len(sources),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(OUTPUT_PATH),
        "by_category": dict(sorted(by_category.items())),
        "by_document_type": dict(sorted(by_type.items())),
        "by_split": dict(sorted(by_split.items())),
        "note": (
            "Dataset construido desde PDFs locales de julio y ejemplos del proyecto. "
            "Las etiquetas se generaron como curación experta inicial basada en objeto, "
            "producto, proveedor y requisitos detectados; se recomienda revisión humana final."
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
