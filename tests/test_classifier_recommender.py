from app.classifier import classify_requirement
from app.config import get_settings
from app.knowledge_base import KnowledgeBase
from app.recommender import RecommendationEngine
from app.schemas import Solution


def test_classifier_detects_cybersecurity_edr():
    text = (
        "Se requiere una solucion EDR/XDR para estaciones de trabajo, "
        "servidores, consola centralizada, respuesta ante amenazas y soporte."
    )

    result = classify_requirement(text)

    assert result.category == "Antivirus / EDR / XDR"
    assert result.confidence >= 0.5


def test_recommender_returns_matching_solution_from_knowledge_base():
    kb = KnowledgeBase.from_records(
        [
            {
                "id": "solucion_001",
                "nombre": "Backup cloud",
                "categoria": "Backup y recuperación",
                "descripcion": "Servicio de respaldo en la nube.",
                "caracteristicas_principales": ["copias automaticas", "retencion"],
                "requisitos_que_cubre": ["backup", "recuperacion", "nube"],
                "restricciones": [],
                "modalidad": "servicio",
                "observaciones": "",
            }
        ]
    )
    recommender = RecommendationEngine(kb)

    result = recommender.recommend(
        category="Backup y recuperación",
        requirements=["respaldo automatico en nube", "recuperacion ante desastres"],
    )

    assert result.recommended is not None
    assert result.recommended.nombre == "Backup cloud"
    assert result.confidence > 0


def test_recommend_by_vector_uses_knowledge_base_search(monkeypatch):
    kb = KnowledgeBase([])
    best = Solution(
        id="solucion_001",
        nombre="Backup cloud",
        categoria="Backup y recuperación",
        descripcion="Servicio de respaldo en la nube.",
        caracteristicas_principales=[],
        requisitos_que_cubre=[],
        restricciones=[],
        modalidad="servicio",
    )
    alternative = Solution(
        id="solucion_002",
        nombre="VPS administrado",
        categoria="Hosting / cPanel / VPS",
        descripcion="Servidor virtual administrado.",
        caracteristicas_principales=[],
        requisitos_que_cubre=[],
        restricciones=[],
        modalidad="servicio",
    )

    def fake_search_by_vector(query_embedding, settings, top_k=5):
        return [(best, 0.91), (alternative, 0.42)]

    monkeypatch.setattr(kb, "search_by_vector", fake_search_by_vector)
    recommender = RecommendationEngine(kb)

    result = recommender.recommend_by_vector([0.1, 0.2], get_settings())

    assert result.recommended is best
    assert result.alternatives == [alternative]
    assert result.confidence == 0.91
    assert "Backup cloud" in result.rationale
