from __future__ import annotations

from pathlib import Path

from app.classifier import CATEGORIES
from app.config import ensure_directories, get_settings
from app.document_loader import load_document, save_uploaded_file
from app.evaluator import calculate_metrics
from app.exporter import export_analysis_json, export_analysis_markdown, export_analysis_pdf
from app.rag_engine import RAGEngine
from app.schemas import ExpertEvaluation
from app.storage import AnalysisStorage


def run() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit no esta instalado. Ejecute: pip install -r requirements.txt"
        ) from exc

    settings = get_settings()
    ensure_directories(settings)
    storage = AnalysisStorage(settings.database_path)

    st.set_page_config(
        page_title="TDR AI Analyzer",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles(st)

    st.sidebar.title("TDR AI Analyzer")
    st.sidebar.caption("Análisis automático de términos de referencia tecnológicos")
    section = st.sidebar.radio(
        "Navegación del proyecto",
        ["Nuevo análisis", "Mis análisis", "Base de conocimiento", "Ayuda"],
    )
    st.sidebar.warning(
        "La recomendación generada por IA es preliminar. No reemplaza la revisión "
        "de un especialista técnico ni constituye evaluación legal, contractual o financiera."
    )
    st.sidebar.markdown("### Estado IA")
    if settings.has_openai_key:
        st.sidebar.success("Clave detectada")
        st.sidebar.caption(f"Modelo configurado: `{settings.model_name}`")
    else:
        st.sidebar.warning("Demo sin OPENAI_API_KEY")

    if section == "Nuevo análisis":
        _render_new_analysis(st, settings, storage)
    elif section == "Mis análisis":
        _render_recent_analyses(st, storage)
    elif section == "Base de conocimiento":
        _render_knowledge_base(st, settings)
    else:
        _render_help(st)


def _render_new_analysis(st, settings, storage: AnalysisStorage) -> None:
    st.title("Análisis de Términos de Referencia Tecnológicos")

    upload_col, text_col = st.columns([0.9, 1.1])
    with upload_col:
        st.subheader("1. Cargar documento TDR")
        uploaded = st.file_uploader(
            "Seleccione un archivo PDF, DOCX o TXT",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=False,
        )
        analyze = st.button("Analizar documento", type="primary", disabled=uploaded is None)

    with text_col:
        st.subheader("2. Texto extraído")
        extracted_text = st.session_state.get("extracted_text", "")
        st.text_area("Vista previa", extracted_text, height=260, label_visibility="collapsed")

    if uploaded is not None and analyze:
        with st.spinner("Extrayendo texto y generando análisis..."):
            path = save_uploaded_file(uploaded, settings)
            text = load_document(path)
            st.session_state["extracted_text"] = text
            engine = RAGEngine(settings=settings)
            result = engine.analyze_document(uploaded.name, text)
            analysis_id = storage.save_analysis(result)
            st.session_state["analysis_result"] = result
            st.session_state["analysis_id"] = analysis_id
        st.rerun()

    result = st.session_state.get("analysis_result")
    if not result:
        st.info("Cargue un documento para iniciar el análisis automático.")
        return

    st.divider()
    tab_summary, tab_requirements, tab_recommendation, tab_export, tab_eval = st.tabs(
        [
            "Análisis estructurado",
            "Requisitos técnicos",
            "Solución recomendada",
            "Exportación",
            "Evaluación experta",
        ]
    )

    with tab_summary:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Categoría", result.categoria_tecnologica)
        metric_cols[1].metric("Requisitos", len(result.requisitos_tecnicos))
        metric_cols[2].metric("Tiempo", f"{result.tiempo_analisis_segundos:.2f}s")
        metric_cols[3].metric("Proveedor IA", result.proveedor_ia)
        if result.modo_demo and result.error_openai:
            st.error(f"No se usó OpenAI en este análisis. Fallback activado: {result.error_openai}")
        elif result.modo_demo:
            st.warning("Este análisis se generó en modo demo local.")
        else:
            st.success(f"Este análisis usó OpenAI correctamente con el modelo {settings.model_name}.")
        st.subheader("Resumen general")
        st.write(result.resumen_general)
        st.subheader("Objeto o necesidad principal")
        st.write(result.objeto_requerimiento)
        st.subheader("Datos faltantes o ambiguos")
        for item in result.datos_faltantes_o_ambiguos:
            st.warning(item)

    with tab_requirements:
        st.subheader("Requisitos técnicos identificados")
        st.dataframe(
            [req.__dict__ for req in result.requisitos_tecnicos],
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Fragmentos recuperados")
        for fragment in result.fragmentos_recuperados:
            st.code(fragment[:900])

    with tab_recommendation:
        st.subheader("Clasificación tecnológica")
        st.info(result.categoria_tecnologica)
        st.subheader("Solución recomendada")
        st.success(result.solucion_recomendada.nombre)
        st.write(f"**Categoría:** {result.solucion_recomendada.categoria}")
        st.write(f"**Nivel de confianza:** {result.solucion_recomendada.nivel_confianza}")
        st.write(result.solucion_recomendada.justificacion)
        st.subheader("Alternativas")
        st.write(result.alternativas or ["Sin alternativas identificadas."])
        st.caption(result.observaciones)

    with tab_export:
        st.subheader("Exportar análisis")
        json_path = export_analysis_json(result, settings.outputs_json_dir)
        md_path = export_analysis_markdown(result, settings.outputs_markdown_dir)
        pdf_path = export_analysis_pdf(result, settings.outputs_pdf_dir)

        col_json, col_md, col_pdf = st.columns(3)
        col_json.download_button(
            "Descargar JSON",
            json_path.read_bytes(),
            file_name=json_path.name,
            mime="application/json",
        )
        col_md.download_button(
            "Descargar Markdown",
            md_path.read_bytes(),
            file_name=md_path.name,
            mime="text/markdown",
        )
        col_pdf.download_button(
            "Descargar PDF",
            pdf_path.read_bytes(),
            file_name=pdf_path.name,
            mime="application/pdf",
        )

    with tab_eval:
        _render_evaluation_form(st, storage, result)


def _render_evaluation_form(st, storage: AnalysisStorage, result) -> None:
    st.subheader("Validación por experto")
    with st.form("expert_validation"):
        expert_solution = st.text_input(
            "Solución recomendada por experto",
            value=result.solucion_recomendada.nombre,
        )
        expert_category = st.selectbox(
            "Categoría correcta",
            options=list(CATEGORIES.keys()) + ["Otro / no identificado"],
            index=(list(CATEGORIES.keys()) + ["Otro / no identificado"]).index(result.categoria_tecnologica)
            if result.categoria_tecnologica in list(CATEGORIES.keys()) + ["Otro / no identificado"]
            else 0,
        )
        correct_requirements = st.text_area(
            "Requisitos correctamente extraídos (uno por línea)",
            value="\n".join(req.descripcion for req in result.requisitos_tecnicos),
        )
        omitted_requirements = st.text_area("Requisitos omitidos (uno por línea)")
        recommendation_status = st.selectbox(
            "Recomendación de la IA",
            ["correcta", "parcialmente correcta", "incorrecta"],
        )
        observations = st.text_area("Observaciones")
        submitted = st.form_submit_button("Guardar validación", type="primary")

    if submitted:
        correct = [line.strip() for line in correct_requirements.splitlines() if line.strip()]
        omitted = [line.strip() for line in omitted_requirements.splitlines() if line.strip()]
        metrics = calculate_metrics(
            extracted_requirements=[req.descripcion for req in result.requisitos_tecnicos],
            correct_requirements=correct + omitted,
            predicted_category=result.categoria_tecnologica,
            expert_category=expert_category,
            predicted_solution=result.solucion_recomendada.nombre,
            expert_solution=expert_solution,
            analysis_seconds=result.tiempo_analisis_segundos,
        )
        evaluation = ExpertEvaluation(
            solucion_recomendada_experto=expert_solution,
            categoria_correcta=expert_category,
            requisitos_correctamente_extraidos=correct,
            requisitos_omitidos=omitted,
            recomendacion_ia=recommendation_status,
            observaciones=observations,
        )
        storage.save_evaluation(st.session_state.get("analysis_id"), evaluation, metrics)
        st.success("Validación experta registrada.")
        st.json(metrics.__dict__)


def _render_recent_analyses(st, storage: AnalysisStorage) -> None:
    st.title("Mis análisis")
    rows = storage.list_recent_analyses()
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Todavía no existen análisis registrados.")


def _render_knowledge_base(st, settings) -> None:
    st.title("Base de conocimiento tecnológica")
    st.caption("Archivo editable: data/knowledge_base/solutions.json")
    if settings.knowledge_base_path.exists():
        st.json(settings.knowledge_base_path.read_text(encoding="utf-8"))
    else:
        st.warning("No se encontró la base de conocimiento.")


def _render_help(st) -> None:
    st.title("Ayuda")
    st.markdown(
        """
        1. Cargue un documento TDR en PDF, DOCX o TXT.
        2. Revise el texto extraído.
        3. Ejecute el análisis para obtener requisitos, categoría y recomendación.
        4. Exporte el resultado en Markdown, JSON o PDF.
        5. Registre la validación experta para calcular métricas.

        El sistema puede operar en modo demo si no existe `OPENAI_API_KEY`.
        """
    )


def _inject_styles(st) -> None:
    st.markdown(
        """
        <style>
        :root {
          --tdr-bg: #f5f8f7;
          --tdr-sidebar: #ffffff;
          --tdr-text: #14242b;
          --tdr-muted: #52646d;
          --tdr-accent: #0b6b6f;
          --tdr-accent-dark: #064e52;
          --tdr-border: #d7e2df;
          --tdr-panel: #ffffff;
        }
        .stApp {
          background: var(--tdr-bg);
          color: var(--tdr-text);
        }
        .block-container {
          padding-top: 3rem;
          max-width: 1380px;
        }
        h1, h2, h3, h4, h5, h6, p, label, span, div {
          color: var(--tdr-text);
        }
        [data-testid="stSidebar"] {
          background: var(--tdr-sidebar);
          border-right: 1px solid var(--tdr-border);
        }
        [data-testid="stSidebar"] * {
          color: var(--tdr-text) !important;
        }
        [data-testid="stSidebar"] h1 {
          color: var(--tdr-accent-dark) !important;
          font-weight: 800;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
          color: var(--tdr-muted) !important;
        }
        .stButton>button, .stDownloadButton>button {
          border-radius: 8px;
          border-color: var(--tdr-border);
        }
        .stButton>button[kind="primary"] {
          background: var(--tdr-accent);
          border-color: var(--tdr-accent);
          color: white;
        }
        div[data-testid="stMetric"] {
          background: var(--tdr-panel);
          border: 1px solid var(--tdr-border);
          border-radius: 8px;
          padding: 12px;
        }
        [data-testid="stFileUploaderDropzone"],
        textarea,
        .stTextArea textarea {
          background: #ffffff !important;
          border: 1px solid var(--tdr-border) !important;
          color: var(--tdr-text) !important;
        }
        div[data-testid="stAlert"] {
          border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    run()
