from app.classifier import classify_requirement
from app.knowledge_base import KnowledgeBase
from app.recommender import RecommendationEngine


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
