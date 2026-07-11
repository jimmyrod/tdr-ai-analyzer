from __future__ import annotations

import json
from pathlib import Path

from app.classifier import CATEGORIES
from app.config import ensure_directories, get_settings
from app.document_loader import load_document, save_uploaded_file
from app.evaluator import calculate_metrics
from app.exporter import export_analysis_json, export_analysis_markdown, export_analysis_pdf
from app.knowledge_base import KnowledgeBase
from app.offline_evaluation import (
    load_evaluation_cases,
    load_evaluation_cases_payload,
    run_evaluation,
    summarize_evaluation,
    write_evaluation_outputs,
)
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
        [
            "Nuevo análisis",
            "Evaluación del modelo",
            "Mis análisis",
            "Base de conocimiento",
            "Ayuda",
        ],
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
    elif section == "Evaluación del modelo":
        _render_model_evaluation(st, settings)
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


def _render_model_evaluation(st, settings) -> None:
    st.title("Evaluación y optimización del modelo IA")
    default_cases_path = settings.evaluations_dir / "evaluation_cases.example.json"
    output_dir = settings.project_root / "outputs" / "evaluation"

    source_col, action_col = st.columns([1.2, 0.8])
    with source_col:
        source = st.radio(
            "Fuente de casos etiquetados",
            ["Casos de ejemplo", "Subir JSON"],
            horizontal=True,
        )
        uploaded = None
        if source == "Subir JSON":
            uploaded = st.file_uploader(
                "Archivo JSON de casos etiquetados",
                type=["json"],
                accept_multiple_files=False,
                key="evaluation_cases_upload",
            )
        elif default_cases_path.exists():
            st.caption(f"Dataset activo: {default_cases_path}")
        else:
            st.warning("No se encontró el dataset de ejemplo.")

    with action_col:
        st.write("")
        st.write("")
        run_clicked = st.button(
            "Ejecutar evaluación",
            type="primary",
            use_container_width=True,
            disabled=source == "Subir JSON" and uploaded is None,
        )

    if run_clicked:
        try:
            with st.spinner("Ejecutando análisis sobre casos etiquetados..."):
                cases = _load_cases_for_portal(source, uploaded, default_cases_path, settings)
                engine = RAGEngine(settings=settings)
                rows = run_evaluation(
                    cases,
                    analyze=lambda case: engine.analyze_document(case.document_name, case.text),
                    manual_minutes=settings.manual_analysis_minutes,
                )
                summary = summarize_evaluation(rows)
                paths = write_evaluation_outputs(rows, summary, output_dir)
                st.session_state["model_evaluation_result"] = {
                    "rows": rows,
                    "summary": summary,
                    "paths": paths,
                }
        except Exception as exc:
            st.error(f"No fue posible ejecutar la evaluación: {exc}")

    evaluation = st.session_state.get("model_evaluation_result")
    if not evaluation:
        st.info("Ejecute la evaluación para generar el informe del modelo.")
        return

    _render_evaluation_report(st, evaluation["rows"], evaluation["summary"], evaluation["paths"])


def _load_cases_for_portal(source: str, uploaded, default_cases_path: Path, settings):
    if source == "Subir JSON":
        payload = json.loads(uploaded.getvalue().decode("utf-8-sig"))
        return load_evaluation_cases_payload(payload, base_dir=settings.project_root)
    return load_evaluation_cases(default_cases_path)


def _render_evaluation_report(st, rows, summary: dict, paths: dict[str, Path]) -> None:
    overall = summary.get("overall", {})
    st.divider()
    metric_cols = st.columns(5)
    metric_cols[0].metric("Casos", overall.get("case_count", 0))
    metric_cols[1].metric("F1 requisitos", f"{overall.get('f1_score', 0):.4f}")
    metric_cols[2].metric("Clasificación", f"{overall.get('exactitud_clasificacion', 0):.4f}")
    metric_cols[3].metric("Solución", f"{overall.get('coincidencia_con_experto', 0):.4f}")
    metric_cols[4].metric("Tiempo prom.", f"{overall.get('tiempo_analisis_segundos', 0):.2f}s")

    tab_report, tab_benchmark, tab_fairness, tab_generalization, tab_cases, tab_downloads = st.tabs(
        [
            "Informe",
            "Benchmarking",
            "Sesgo",
            "Generalización",
            "Casos",
            "Descargas",
        ]
    )

    with tab_report:
        report_path = paths.get("report_markdown")
        if report_path and report_path.exists():
            st.markdown(report_path.read_text(encoding="utf-8"))
        else:
            st.warning("No se encontró el informe Markdown generado.")

    with tab_benchmark:
        st.subheader("Benchmarking comparativo")
        st.dataframe(_benchmark_table(summary), use_container_width=True, hide_index=True)

    with tab_fairness:
        st.subheader("Análisis de sesgo por grupos")
        fairness = summary.get("fairness", {})
        if fairness:
            group_key = st.selectbox("Grupo", sorted(fairness.keys()))
            group_payload = fairness[group_key]
            cols = st.columns(3)
            cols[0].metric("Brecha F1", f"{group_payload.get('f1_score_gap', 0):.4f}")
            cols[1].metric(
                "Brecha clasificación",
                f"{group_payload.get('exactitud_clasificacion_gap', 0):.4f}",
            )
            cols[2].metric(
                "Brecha solución",
                f"{group_payload.get('coincidencia_con_experto_gap', 0):.4f}",
            )
            st.dataframe(_group_metrics_table(group_payload), use_container_width=True, hide_index=True)
        else:
            st.info("No existen grupos suficientes para analizar sesgo.")

    with tab_generalization:
        st.subheader("Diagnóstico train vs validation")
        st.dataframe(_split_table(summary), use_container_width=True, hide_index=True)
        st.json(summary.get("generalization", {}))

    with tab_cases:
        st.subheader("Resultados por caso")
        st.dataframe(_case_rows_table(rows), use_container_width=True, hide_index=True)

    with tab_downloads:
        st.subheader("Archivos generados")
        for label, path in paths.items():
            if not path.exists():
                continue
            mime = "text/markdown" if path.suffix == ".md" else "application/json"
            if path.suffix == ".csv":
                mime = "text/csv"
            elif path.suffix == ".pdf":
                mime = "application/pdf"
            st.download_button(
                f"Descargar {label}",
                path.read_bytes(),
                file_name=path.name,
                mime=mime,
                use_container_width=True,
            )


def _benchmark_table(summary: dict) -> list[dict]:
    rows = []
    for name, metrics in summary.get("benchmarking", {}).items():
        rows.append(
            {
                "modelo": name,
                "casos": metrics.get("case_count", 0),
                "f1_requisitos": metrics.get("f1_score", ""),
                "exactitud_categoria": metrics.get("exactitud_clasificacion", ""),
                "coincidencia_solucion": metrics.get("coincidencia_con_experto", ""),
                "nota": metrics.get("nota", ""),
            }
        )
    return rows


def _group_metrics_table(group_payload: dict) -> list[dict]:
    rows = []
    for group, metrics in group_payload.get("groups", {}).items():
        rows.append(
            {
                "grupo": group,
                "casos": metrics.get("case_count", 0),
                "f1_requisitos": metrics.get("f1_score", 0),
                "exactitud_categoria": metrics.get("exactitud_clasificacion", 0),
                "coincidencia_solucion": metrics.get("coincidencia_con_experto", 0),
            }
        )
    return rows


def _split_table(summary: dict) -> list[dict]:
    rows = []
    for split, metrics in summary.get("by_split", {}).items():
        rows.append(
            {
                "split": split,
                "casos": metrics.get("case_count", 0),
                "precision_requisitos": metrics.get("precision_requisitos", 0),
                "recall_requisitos": metrics.get("recall_requisitos", 0),
                "f1_requisitos": metrics.get("f1_score", 0),
                "exactitud_categoria": metrics.get("exactitud_clasificacion", 0),
                "coincidencia_solucion": metrics.get("coincidencia_con_experto", 0),
            }
        )
    return rows


def _case_rows_table(rows) -> list[dict]:
    return [
        {
            "caso": row.case_id,
            "documento": row.document_name,
            "split": row.split,
            "categoria_esperada": row.expected_category,
            "categoria_predicha": row.predicted_category,
            "solucion_esperada": row.expected_solution,
            "solucion_predicha": row.predicted_solution,
            "precision": row.precision_requisitos,
            "recall": row.recall_requisitos,
            "f1": row.f1_score,
            "tiempo_s": row.tiempo_analisis_segundos,
        }
        for row in rows
    ]


def _render_recent_analyses(st, storage: AnalysisStorage) -> None:
    st.title("Mis análisis")
    rows = storage.list_recent_analyses()
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Todavía no existen análisis registrados.")


def _render_knowledge_base(st, settings) -> None:
    st.title("Base de conocimiento tecnológica")
    kb = KnowledgeBase.load(settings.knowledge_base_path, settings=settings)
    solutions = kb.all()

    if settings.has_supabase:
        st.caption(f"Fuente: Supabase (tabla `solutions`) — {len(solutions)} soluciones.")
    else:
        st.caption(
            f"Fuente: archivo local `data/knowledge_base/solutions.json` — {len(solutions)} soluciones."
        )

    if not solutions:
        st.warning("No se encontraron soluciones en la base de conocimiento.")
        return

    st.dataframe(
        [
            {
                "Nombre": solution.nombre,
                "Categoría": solution.categoria,
                "Modalidad": solution.modalidad,
                "Descripción": solution.descripcion,
            }
            for solution in solutions
        ],
        use_container_width=True,
        hide_index=True,
    )

    for solution in solutions:
        with st.expander(f"{solution.nombre} — {solution.categoria}"):
            st.write(solution.descripcion)
            st.write("**Características principales:**")
            st.write(solution.caracteristicas_principales or ["-"])
            st.write("**Requisitos que cubre:**")
            st.write(solution.requisitos_que_cubre or ["-"])
            st.write("**Restricciones:**")
            st.write(solution.restricciones or ["-"])
            if solution.observaciones:
                st.caption(solution.observaciones)


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
          --tdr-code-bg: #eef3f2;
          --tdr-input-bg: #ffffff;
        }
        @media (prefers-color-scheme: dark) {
          :root {
            --tdr-bg: #0f1a1c;
            --tdr-sidebar: #142325;
            --tdr-text: #e7f1ef;
            --tdr-muted: #9fb4b0;
            --tdr-accent-dark: #7fe3da;
            --tdr-border: #24393c;
            --tdr-panel: #142325;
            --tdr-code-bg: #1b2c2e;
            --tdr-input-bg: #142325;
          }
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
        code {
          background: var(--tdr-code-bg) !important;
          color: var(--tdr-text) !important;
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
          background: var(--tdr-input-bg) !important;
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
