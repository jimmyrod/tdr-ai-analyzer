from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TextChunk:
    index: int
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Requirement:
    id: str
    descripcion: str
    tipo: str
    prioridad: str
    fragmento_fuente: str


@dataclass
class RecommendedSolution:
    nombre: str
    categoria: str
    justificacion: str
    nivel_confianza: str


@dataclass
class Solution:
    id: str
    nombre: str
    categoria: str
    descripcion: str
    caracteristicas_principales: list[str]
    requisitos_que_cubre: list[str]
    restricciones: list[str]
    modalidad: str
    observaciones: str = ""


@dataclass
class ClassificationResult:
    category: str
    confidence: float
    evidence: list[str] = field(default_factory=list)


@dataclass
class RecommendationResult:
    recommended: Solution | None
    alternatives: list[Solution]
    confidence: float
    rationale: str


@dataclass
class EvaluationMetrics:
    precision_requisitos: float
    recall_requisitos: float
    f1_score: float
    exactitud_clasificacion: float
    coincidencia_con_experto: float
    tiempo_analisis_segundos: float
    reduccion_tiempo_estimada: float


@dataclass
class ExpertEvaluation:
    solucion_recomendada_experto: str
    categoria_correcta: str
    requisitos_correctamente_extraidos: list[str]
    requisitos_omitidos: list[str]
    recomendacion_ia: str
    observaciones: str


@dataclass
class AnalysisResult:
    nombre_documento: str
    resumen_general: str
    objeto_requerimiento: str
    categoria_tecnologica: str
    requisitos_tecnicos: list[Requirement]
    productos_o_servicios_esperados: list[str]
    solucion_recomendada: RecommendedSolution
    alternativas: list[str]
    datos_faltantes_o_ambiguos: list[str]
    observaciones: str
    fragmentos_recuperados: list[str] = field(default_factory=list)
    tiempo_analisis_segundos: float = 0.0
    modo_demo: bool = True
    proveedor_ia: str = "Demo local"
    error_openai: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data
