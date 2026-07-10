from __future__ import annotations

from app.config import Settings
from app.knowledge_base import KnowledgeBase
from app.schemas import RecommendationResult, Solution
from app.text_cleaner import normalize_for_matching


class RecommendationEngine:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base

    def recommend_by_vector(
        self, query_embedding: list[float], settings: Settings, top_k: int = 5
    ) -> RecommendationResult:
        matches = self.knowledge_base.search_by_vector(query_embedding, settings, top_k=top_k)
        if not matches:
            return RecommendationResult(
                recommended=None,
                alternatives=[],
                confidence=0.0,
                rationale="La busqueda vectorial no encontro soluciones en la base de conocimiento.",
            )

        best_solution, best_similarity = matches[0]
        alternatives = [solution for solution, _ in matches[1:4]]
        confidence = max(0.0, min(0.95, best_similarity))
        rationale = (
            f"La solucion '{best_solution.nombre}' ({best_solution.categoria}) fue la mas "
            f"cercana semanticamente a los requisitos del TDR, con una similitud de "
            f"{best_similarity:.2f} sobre la base de conocimiento vectorizada en Supabase."
        )

        return RecommendationResult(
            recommended=best_solution,
            alternatives=alternatives,
            confidence=round(confidence, 2),
            rationale=rationale,
        )

    def recommend(self, category: str, requirements: list[str]) -> RecommendationResult:
        candidates = self.knowledge_base.all()
        query = normalize_for_matching(" ".join(requirements + [category]))
        query_terms = {term for term in query.split() if len(term) > 2}
        scored: list[tuple[float, Solution, list[str]]] = []

        for solution in candidates:
            solution_text = normalize_for_matching(
                " ".join(
                    [
                        solution.nombre,
                        solution.categoria,
                        solution.descripcion,
                        " ".join(solution.caracteristicas_principales),
                        " ".join(solution.requisitos_que_cubre),
                        " ".join(solution.restricciones),
                        solution.modalidad,
                    ]
                )
            )
            solution_terms = {term for term in solution_text.split() if len(term) > 2}
            matches = sorted(query_terms & solution_terms)
            score = float(len(matches))

            if normalize_for_matching(solution.categoria) == normalize_for_matching(category):
                score += 5.0
            if normalize_for_matching(solution.nombre) in query:
                score += 4.0
            for covered in solution.requisitos_que_cubre:
                if normalize_for_matching(covered) in query:
                    score += 2.0

            if score > 0:
                scored.append((score, solution, matches[:8]))

        if not scored:
            return RecommendationResult(
                recommended=None,
                alternatives=[],
                confidence=0.0,
                rationale="No se encontro una solucion compatible en la base de conocimiento.",
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_solution, best_matches = scored[0]
        alternatives = [solution for _, solution, _ in scored[1:4]]
        confidence = min(0.95, best_score / max(best_score + 6.0, 1.0))
        rationale = (
            f"La solucion '{best_solution.nombre}' coincide con la categoria "
            f"'{best_solution.categoria}' y cubre terminos clave: "
            f"{', '.join(best_matches) if best_matches else 'categoria y modalidad solicitada'}."
        )

        return RecommendationResult(
            recommended=best_solution,
            alternatives=alternatives,
            confidence=round(confidence, 2),
            rationale=rationale,
        )
