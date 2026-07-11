from __future__ import annotations

from app.schemas import EvaluationMetrics
from app.text_cleaner import normalize_for_matching


def _matches(predicted: str, expected: str) -> bool:
    predicted_norm = normalize_for_matching(predicted).strip()
    expected_norm = normalize_for_matching(expected).strip()
    if not predicted_norm or not expected_norm:
        return False
    return predicted_norm == expected_norm or predicted_norm in expected_norm or expected_norm in predicted_norm


def calculate_metrics(
    extracted_requirements: list[str],
    correct_requirements: list[str],
    predicted_category: str,
    expert_category: str,
    predicted_solution: str,
    expert_solution: str,
    analysis_seconds: float,
    manual_minutes: float = 60.0,
) -> EvaluationMetrics:
    matched = 0
    used_expected: set[int] = set()

    for predicted in extracted_requirements:
        for index, expected in enumerate(correct_requirements):
            if index in used_expected:
                continue
            if _matches(predicted, expected):
                matched += 1
                used_expected.add(index)
                break

    precision = matched / len(extracted_requirements) if extracted_requirements else 0.0
    recall = matched / len(correct_requirements) if correct_requirements else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    manual_seconds = max(manual_minutes * 60, 1.0)
    reduction = max(0.0, min(1.0, (manual_seconds - analysis_seconds) / manual_seconds))

    return EvaluationMetrics(
        precision_requisitos=round(precision, 4),
        recall_requisitos=round(recall, 4),
        f1_score=round(f1, 4),
        exactitud_clasificacion=1.0 if _matches(predicted_category, expert_category) else 0.0,
        coincidencia_con_experto=1.0 if _matches(predicted_solution, expert_solution) else 0.0,
        tiempo_analisis_segundos=round(analysis_seconds, 2),
        reduccion_tiempo_estimada=round(reduction, 4),
    )
