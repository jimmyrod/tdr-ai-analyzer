from __future__ import annotations

import json
import re
import time
from pathlib import Path

from app.chunker import TextChunker
from app.classifier import classify_requirement
from app.config import Settings, ensure_directories, get_settings
from app.embeddings import EmbeddingProvider
from app.knowledge_base import KnowledgeBase
from app.recommender import RecommendationEngine
from app.schemas import AnalysisResult, RecommendedSolution, Requirement
from app.text_cleaner import clean_text, normalize_for_matching
from app.vector_store import VectorStore


class RAGEngine:
    def __init__(
        self,
        settings: Settings | None = None,
        vector_store_path: Path | None = None,
        knowledge_base: KnowledgeBase | None = None,
    ):
        self.settings = settings or get_settings()
        ensure_directories(self.settings)
        self.chunker = TextChunker(self.settings.chunk_size, self.settings.chunk_overlap)
        self.embedding_provider = EmbeddingProvider(self.settings)
        self.vector_store = VectorStore(
            vector_store_path or self.settings.vector_store_path, settings=self.settings
        )
        self.knowledge_base = knowledge_base or KnowledgeBase.load(
            self.settings.knowledge_base_path, settings=self.settings
        )
        self.recommender = RecommendationEngine(self.knowledge_base)

    def analyze_document(self, document_name: str, raw_text: str) -> AnalysisResult:
        start = time.perf_counter()
        text = clean_text(raw_text)
        chunks = self.chunker.split(text, document_name)
        embeddings = self.embedding_provider.embed_texts([chunk.text for chunk in chunks])
        self.vector_store.add_chunks(document_name, chunks, embeddings)

        openai_error = ""
        try:
            openai_result = self._try_openai_analysis(document_name, text, chunks)
        except Exception as exc:
            openai_result = None
            openai_error = self._safe_error_message(exc)

        if openai_result:
            openai_result.tiempo_analisis_segundos = round(time.perf_counter() - start, 2)
            openai_result.modo_demo = False
            openai_result.proveedor_ia = "OpenAI Responses API"
            openai_result.error_openai = ""
            self._feed_solutions_from_analysis(document_name, openai_result)
            return openai_result

        classification = classify_requirement(text)
        requirements = self._extract_requirements(text)
        products = self._extract_products(text)
        object_text = self._extract_object(text)
        summary = self._summarize(text, object_text)

        query = " ".join([classification.category, object_text] + [req.descripcion for req in requirements])
        query_embedding = self.embedding_provider.embed_text(query)
        # No document_name filter: the vector store accumulates every analyzed TDR,
        # so retrieval draws on the whole history, not just this document's own chunks.
        retrieved = self.vector_store.similarity_search(query_embedding, top_k=4)
        retrieved_texts = [record.text[:700] for record in retrieved] or [chunk.text[:700] for chunk in chunks[:2]]

        recommendation = self._recommend_solution(
            classification.category, requirements, products, retrieved_texts, query_embedding
        )
        recommended_solution = self._build_recommended_solution(recommendation, classification.category, retrieved_texts)
        missing = self._detect_missing_data(text, classification.category)
        observations = self._build_observations(
            classification.confidence,
            bool(self.settings.has_openai_key),
            openai_error,
        )

        result = AnalysisResult(
            nombre_documento=document_name,
            resumen_general=summary,
            objeto_requerimiento=object_text,
            categoria_tecnologica=classification.category,
            requisitos_tecnicos=requirements,
            productos_o_servicios_esperados=products,
            solucion_recomendada=recommended_solution,
            alternativas=[solution.nombre for solution in recommendation.alternatives],
            datos_faltantes_o_ambiguos=missing,
            observaciones=observations,
            fragmentos_recuperados=retrieved_texts,
            tiempo_analisis_segundos=round(time.perf_counter() - start, 2),
            modo_demo=True,
            proveedor_ia="Demo local",
            error_openai=openai_error,
        )
        self._feed_solutions_from_analysis(document_name, result)
        return result

    def _feed_solutions_from_analysis(self, document_name: str, result: AnalysisResult) -> None:
        """Store what this TDR asked for as an origen='analisis' row in 'solutions'."""
        if not self.settings.has_supabase:
            return
        try:
            text = " ".join(
                [result.categoria_tecnologica, result.objeto_requerimiento]
                + [req.descripcion for req in result.requisitos_tecnicos]
            )
            embedding = self.embedding_provider.embed_text(text)
            row = {
                "id": f"analisis:{document_name}",
                "nombre": (result.objeto_requerimiento or document_name)[:150],
                "categoria": result.categoria_tecnologica,
                "descripcion": result.resumen_general[:2000],
                "caracteristicas_principales": result.productos_o_servicios_esperados,
                "requisitos_que_cubre": [req.descripcion for req in result.requisitos_tecnicos],
                "restricciones": result.datos_faltantes_o_ambiguos,
                "modalidad": self._infer_modalidad(result),
                "observaciones": result.observaciones[:2000],
                "origen": "analisis",
                "embedding": embedding,
            }
            self.knowledge_base.add_solution(row, self.settings)
        except Exception:
            # Feeding the catalog is best-effort; never break the analysis over it.
            pass

    def _infer_modalidad(self, result: AnalysisResult) -> str:
        """Copy the modalidad of the recommended solution when it matches a known one."""
        target = normalize_for_matching(result.solucion_recomendada.nombre)
        for solution in self.knowledge_base.all():
            if normalize_for_matching(solution.nombre) == target:
                return solution.modalidad
        return ""

    def _recommend_solution(
        self,
        category: str,
        requirements: list[Requirement],
        products: list[str],
        retrieved_texts: list[str],
        query_embedding: list[float],
    ):
        if self.settings.has_supabase:
            try:
                recommendation = self.recommender.recommend_by_vector(query_embedding, self.settings)
                if recommendation.recommended:
                    return recommendation
            except Exception:
                # Keep the app usable even if Supabase is unreachable or misconfigured.
                pass
        return self.recommender.recommend(
            category,
            [req.descripcion for req in requirements] + products + retrieved_texts,
        )

    def _try_openai_analysis(self, document_name: str, text: str, chunks) -> AnalysisResult | None:
        if not self.settings.has_openai_key:
            return None
        from openai import OpenAI

        prompt = self._build_openai_prompt(document_name, text[:12000])
        client = OpenAI(api_key=self.settings.openai_api_key)
        response = client.responses.create(model=self.settings.model_name, input=prompt)
        content = getattr(response, "output_text", "") or ""
        payload = json.loads(self._extract_json(content))
        return self._analysis_from_payload(document_name, payload)

    def _build_openai_prompt(self, document_name: str, text: str) -> str:
        kb_summary = "\n".join(
            f"- {solution.nombre} | {solution.categoria}: {solution.descripcion}"
            for solution in self.knowledge_base.all()
        )
        return f"""
Analiza el siguiente termino de referencia tecnologico y responde SOLO JSON valido.
Documento: {document_name}

Base de conocimiento:
{kb_summary}

Texto:
{text}

Estructura JSON requerida:
{{
  "resumen_general": "",
  "objeto_requerimiento": "",
  "categoria_tecnologica": "",
  "requisitos_tecnicos": [
    {{"id":"REQ-001","descripcion":"","tipo":"","prioridad":"alta/media/baja","fragmento_fuente":""}}
  ],
  "productos_o_servicios_esperados": [],
  "solucion_recomendada": {{"nombre":"","categoria":"","justificacion":"","nivel_confianza":""}},
  "alternativas": [],
  "datos_faltantes_o_ambiguos": [],
  "observaciones": ""
}}
No inventes soluciones si la evidencia es insuficiente.
"""

    def _extract_json(self, content: str) -> str:
        match = re.search(r"\{.*\}", content, flags=re.S)
        return match.group(0) if match else content

    def _analysis_from_payload(self, document_name: str, payload: dict) -> AnalysisResult:
        requirements = [
            Requirement(
                id=item.get("id", f"REQ-{index:03d}"),
                descripcion=item.get("descripcion", ""),
                tipo=item.get("tipo", "tecnico"),
                prioridad=item.get("prioridad", "media"),
                fragmento_fuente=item.get("fragmento_fuente", ""),
            )
            for index, item in enumerate(payload.get("requisitos_tecnicos", []), start=1)
        ]
        rec = payload.get("solucion_recomendada", {})
        return AnalysisResult(
            nombre_documento=document_name,
            resumen_general=payload.get("resumen_general", ""),
            objeto_requerimiento=payload.get("objeto_requerimiento", ""),
            categoria_tecnologica=payload.get("categoria_tecnologica", "Otro / no identificado"),
            requisitos_tecnicos=requirements,
            productos_o_servicios_esperados=payload.get("productos_o_servicios_esperados", []),
            solucion_recomendada=RecommendedSolution(
                nombre=rec.get("nombre", "No determinado"),
                categoria=rec.get("categoria", payload.get("categoria_tecnologica", "")),
                justificacion=rec.get("justificacion", ""),
                nivel_confianza=rec.get("nivel_confianza", "media"),
            ),
            alternativas=payload.get("alternativas", []),
            datos_faltantes_o_ambiguos=payload.get("datos_faltantes_o_ambiguos", []),
            observaciones=payload.get("observaciones", ""),
            fragmentos_recuperados=[],
            modo_demo=False,
            proveedor_ia="OpenAI Responses API",
            error_openai="",
        )

    def _extract_requirements(self, text: str) -> list[Requirement]:
        sentences = self._sentences(text)
        requirement_markers = [
            "debe",
            "debera",
            "deberá",
            "requiere",
            "requisito",
            "incluye",
            "soporte",
            "garantia",
            "garantía",
            "licencia",
            "seguridad",
            "disponibilidad",
            "backup",
            "correo",
            "usuarios",
            "capacitacion",
            "capacitación",
            "entregable",
        ]
        selected = [
            sentence
            for sentence in sentences
            if any(marker in normalize_for_matching(sentence) for marker in requirement_markers)
        ]
        if not selected:
            selected = sentences[:5]

        requirements: list[Requirement] = []
        seen: set[str] = set()
        for sentence in selected:
            description = sentence.strip(" -•")
            normalized = normalize_for_matching(description)
            if len(description) < 18 or normalized in seen:
                continue
            seen.add(normalized)
            requirements.append(
                Requirement(
                    id=f"REQ-{len(requirements) + 1:03d}",
                    descripcion=description[:260],
                    tipo=self._requirement_type(description),
                    prioridad=self._priority(description),
                    fragmento_fuente=description[:350],
                )
            )
            if len(requirements) >= 12:
                break
        return requirements

    def _extract_products(self, text: str) -> list[str]:
        products: list[str] = []
        markers = ["producto", "servicio", "entregable", "licencia", "capacitacion", "soporte"]
        for sentence in self._sentences(text):
            normalized = normalize_for_matching(sentence)
            if any(marker in normalized for marker in markers):
                products.append(sentence[:180])
            if len(products) >= 6:
                break
        return products or ["Análisis técnico preliminar del requerimiento"]

    def _extract_object(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            normalized = normalize_for_matching(line)
            if normalized.startswith("objeto") or "objeto de la contratacion" in normalized:
                if ":" in line:
                    return line.split(":", 1)[1].strip()[:500]
                if index + 1 < len(lines):
                    return lines[index + 1][:500]
        return self._sentences(text)[0][:500] if self._sentences(text) else "No identificado."

    def _summarize(self, text: str, object_text: str) -> str:
        sentences = self._sentences(text)
        if not sentences:
            return "No fue posible generar un resumen por falta de texto extraido."
        body = " ".join(sentences[:3])
        return f"{object_text} {body}".strip()[:900]

    def _detect_missing_data(self, text: str, category: str) -> list[str]:
        normalized = normalize_for_matching(text)
        checks = {
            "cantidad de usuarios/licencias": ["usuario", "usuarios", "licencia", "licencias"],
            "plazo de contratación o vigencia": ["plazo", "vigencia", "meses", "años", "anos"],
            "nivel de soporte o SLA": ["sla", "soporte", "tiempo de respuesta"],
            "presupuesto referencial": ["presupuesto", "valor referencial", "monto"],
        }
        if category == "Backup y recuperación":
            checks["periodo de retención de respaldos"] = ["retencion", "retención"]
        if category in {"Antivirus / EDR / XDR", "Ciberseguridad"}:
            checks["cantidad de endpoints o servidores"] = ["endpoint", "estaciones", "servidores"]

        missing = [
            f"No se especifica {label}."
            for label, terms in checks.items()
            if not any(term in normalized for term in terms)
        ]
        return missing[:6] or ["No se identifican datos faltantes críticos en el texto analizado."]

    def _build_recommended_solution(self, recommendation, category: str, fragments: list[str]) -> RecommendedSolution:
        if not recommendation.recommended:
            return RecommendedSolution(
                nombre="No determinado",
                categoria=category,
                justificacion=(
                    "La base de conocimiento no contiene una solucion con evidencia suficiente. "
                    "Se recomienda revisar manualmente el TDR y ampliar la base de conocimiento."
                ),
                nivel_confianza="baja",
            )

        confidence_label = "alta" if recommendation.confidence >= 0.75 else "media" if recommendation.confidence >= 0.45 else "baja"
        evidence = " ".join(fragment[:180] for fragment in fragments[:2])
        justification = (
            f"{recommendation.rationale} La recomendacion se basa en los fragmentos recuperados "
            f"del TDR y en la base de conocimiento. Evidencia documental: {evidence}"
        )
        return RecommendedSolution(
            nombre=recommendation.recommended.nombre,
            categoria=recommendation.recommended.categoria,
            justificacion=justification[:1200],
            nivel_confianza=confidence_label,
        )

    def _build_observations(self, confidence: float, has_key: bool, openai_error: str = "") -> str:
        if openai_error:
            mode = f"modo demo por fallback de OpenAI ({openai_error})"
        else:
            mode = "modo demo con OPENAI_API_KEY detectada" if has_key else "modo demo sin OPENAI_API_KEY"
        return (
            f"Análisis generado en {mode}. La recomendación es preliminar, no reemplaza la "
            f"revisión de un especialista técnico y no constituye evaluación legal, contractual "
            f"ni financiera. Confianza de clasificación: {confidence:.2f}."
        )

    def _safe_error_message(self, exc: Exception) -> str:
        message = str(exc).replace(self.settings.openai_api_key, "[REDACTED]")
        message = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-[REDACTED]", message)
        message = re.sub(r"\s+", " ", message).strip()
        if len(message) > 220:
            message = message[:217] + "..."
        return f"{exc.__class__.__name__}: {message}" if message else exc.__class__.__name__

    def _requirement_type(self, sentence: str) -> str:
        normalized = normalize_for_matching(sentence)
        if any(term in normalized for term in ["soporte", "capacitacion", "garantia", "implementacion"]):
            return "servicio"
        if any(term in normalized for term in ["disponibilidad", "seguridad", "sla", "tiempo de respuesta"]):
            return "no funcional"
        return "tecnico"

    def _priority(self, sentence: str) -> str:
        normalized = normalize_for_matching(sentence)
        if any(term in normalized for term in ["debe", "obligatorio", "critico", "seguridad", "xdr", "backup"]):
            return "alta"
        if any(term in normalized for term in ["opcional", "deseable"]):
            return "baja"
        return "media"

    def _sentences(self, text: str) -> list[str]:
        compact = re.sub(r"\s+", " ", text)
        sentences = re.split(r"(?<=[.!?])\s+|(?:\n)+", compact)
        return [sentence.strip() for sentence in sentences if sentence.strip()]
