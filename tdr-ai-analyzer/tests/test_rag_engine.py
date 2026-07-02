from app.rag_engine import RAGEngine
from app.schemas import AnalysisResult, RecommendedSolution


def test_rag_engine_demo_generates_structured_analysis(tmp_path):
    engine = RAGEngine(vector_store_path=tmp_path / "vectors")
    object.__setattr__(engine.settings, "openai_api_key", "")
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
    engine = RAGEngine(vector_store_path=tmp_path / "vectors")
    object.__setattr__(engine.settings, "openai_api_key", "sk-test-redacted")
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
    engine = RAGEngine(vector_store_path=tmp_path / "vectors")
    object.__setattr__(engine.settings, "openai_api_key", "sk-test-redacted")
    monkeypatch.setattr(engine.embedding_provider, "embed_texts", lambda texts: [[0.1] * 96 for _ in texts])
    monkeypatch.setattr(engine.embedding_provider, "embed_text", lambda text: [0.1] * 96)

    def fake_openai_analysis(document_name, text, chunks):
        raise RuntimeError("simulated invalid model")

    monkeypatch.setattr(engine, "_try_openai_analysis", fake_openai_analysis)

    result = engine.analyze_document("tdr_fallback.txt", "OBJETO: requiere backup cloud.")

    assert result.modo_demo is True
    assert result.proveedor_ia == "Demo local"
    assert "RuntimeError" in result.error_openai
