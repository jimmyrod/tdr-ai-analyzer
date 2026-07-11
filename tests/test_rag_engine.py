from app.config import get_settings
from app.rag_engine import RAGEngine
from app.schemas import AnalysisResult, RecommendationResult, RecommendedSolution


def _offline_settings(openai_api_key: str = ""):
    settings = get_settings()
    object.__setattr__(settings, "openai_api_key", openai_api_key)
    object.__setattr__(settings, "supabase_url", "")
    return settings


def test_rag_engine_demo_generates_structured_analysis(tmp_path):
    engine = RAGEngine(settings=_offline_settings(), vector_store_path=tmp_path / "vectors")
    text = """
    OBJETO: Contratar licencias de Microsoft 365 para correo corporativo,
    almacenamiento en OneDrive y videoconferencia. El proveedor debe incluir
    soporte tecnico, activacion de usuarios y capacitacion.
    REQUISITOS: correo seguro, Teams, administracion de usuarios, garantia de soporte.
    """

    result = engine.analyze_document("tdr_m365.txt", text)

    assert result.nombre_documento == "tdr_m365.txt"
    assert result.resumen_general
    assert result.objeto_requerimiento
    assert result.categoria_tecnologica in {
        "Licenciamiento de software",
        "Colaboración y videoconferencia",
        "Cloud / SaaS",
    }
    assert result.requisitos_tecnicos
    assert result.solucion_recomendada.nombre
    assert result.solucion_recomendada.justificacion
    assert result.proveedor_ia == "Demo local"


def test_rag_engine_marks_openai_result_when_generation_succeeds(tmp_path, monkeypatch):
    engine = RAGEngine(
        settings=_offline_settings("sk-test-redacted"), vector_store_path=tmp_path / "vectors"
    )
    monkeypatch.setattr(engine.embedding_provider, "embed_texts", lambda texts: [[0.1] * 96 for _ in texts])
    monkeypatch.setattr(engine.embedding_provider, "embed_text", lambda text: [0.1] * 96)

    def fake_openai_analysis(document_name, text, chunks):
        return AnalysisResult(
            nombre_documento=document_name,
            resumen_general="Resumen OpenAI.",
            objeto_requerimiento="Objeto OpenAI.",
            categoria_tecnologica="Cloud / SaaS",
            requisitos_tecnicos=[],
            productos_o_servicios_esperados=[],
            solucion_recomendada=RecommendedSolution(
                nombre="Google Workspace Business Standard",
                categoria="Cloud / SaaS",
                justificacion="Justificación generada.",
                nivel_confianza="alta",
            ),
            alternativas=[],
            datos_faltantes_o_ambiguos=[],
            observaciones="",
            modo_demo=False,
            proveedor_ia="OpenAI Responses API",
        )

    monkeypatch.setattr(engine, "_try_openai_analysis", fake_openai_analysis)

    result = engine.analyze_document("tdr_openai.txt", "OBJETO: contratar SaaS.")

    assert result.modo_demo is False
    assert result.proveedor_ia == "OpenAI Responses API"
    assert result.error_openai == ""


def test_rag_engine_reports_openai_fallback_reason(tmp_path, monkeypatch):
    engine = RAGEngine(
        settings=_offline_settings("sk-test-redacted"), vector_store_path=tmp_path / "vectors"
    )
    monkeypatch.setattr(engine.embedding_provider, "embed_texts", lambda texts: [[0.1] * 96 for _ in texts])
    monkeypatch.setattr(engine.embedding_provider, "embed_text", lambda text: [0.1] * 96)

    def fake_openai_analysis(document_name, text, chunks):
        raise RuntimeError("simulated invalid model")

    monkeypatch.setattr(engine, "_try_openai_analysis", fake_openai_analysis)

    result = engine.analyze_document("tdr_fallback.txt", "OBJETO: requiere backup cloud.")

    assert result.modo_demo is True
    assert result.proveedor_ia == "Demo local"
    assert "RuntimeError" in result.error_openai


def test_rag_engine_feeds_solutions_table_when_supabase_enabled(tmp_path, monkeypatch):
    engine = RAGEngine(settings=_offline_settings(), vector_store_path=tmp_path / "vectors")
    monkeypatch.setattr(engine.vector_store, "add_chunks", lambda *a, **k: None)
    monkeypatch.setattr(engine.vector_store, "similarity_search", lambda *a, **k: [])
    monkeypatch.setattr(
        engine.recommender,
        "recommend_by_vector",
        lambda *a, **k: RecommendationResult(
            recommended=None, alternatives=[], confidence=0.0, rationale=""
        ),
    )
    fed_rows = []
    monkeypatch.setattr(
        engine.knowledge_base, "add_solution", lambda row, settings: fed_rows.append(row)
    )
    object.__setattr__(engine.settings, "supabase_url", "http://fake-supabase")
    object.__setattr__(engine.settings, "supabase_secret_key", "fake-key")

    result = engine.analyze_document(
        "tdr_backup.txt", "OBJETO: Contratar backup cloud para servidores institucionales."
    )

    assert len(fed_rows) == 1
    row = fed_rows[0]
    assert row["id"] == "analisis:tdr_backup.txt"
    assert row["origen"] == "analisis"
    assert row["categoria"] == result.categoria_tecnologica
    assert row["embedding"]


def _analysis_result_recommending(nombre: str, categoria: str) -> AnalysisResult:
    return AnalysisResult(
        nombre_documento="doc.txt",
        resumen_general="",
        objeto_requerimiento="",
        categoria_tecnologica=categoria,
        requisitos_tecnicos=[],
        productos_o_servicios_esperados=[],
        solucion_recomendada=RecommendedSolution(
            nombre=nombre, categoria=categoria, justificacion="", nivel_confianza="alta"
        ),
        alternativas=[],
        datos_faltantes_o_ambiguos=[],
        observaciones="",
    )


def test_infer_modalidad_copies_from_matching_catalog_solution(tmp_path):
    engine = RAGEngine(settings=_offline_settings(), vector_store_path=tmp_path / "vectors")
    result = _analysis_result_recommending("Backup cloud", "Backup y recuperación")

    assert engine._infer_modalidad(result) == "servicio"


def test_infer_modalidad_returns_empty_when_no_catalog_match(tmp_path):
    engine = RAGEngine(settings=_offline_settings(), vector_store_path=tmp_path / "vectors")
    result = _analysis_result_recommending("Producto inventado que no existe", "Otro")

    assert engine._infer_modalidad(result) == ""
