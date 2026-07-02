# TDR AI Analyzer

Prototipo académico para analizar términos de referencia tecnológicos mediante procesamiento de lenguaje natural, recuperación aumentada con generación (RAG) y recomendación preliminar de soluciones tecnológicas.

## Problema que resuelve..

El análisis de TDR tecnológicos suele hacerse manualmente sobre documentos PDF, Word o texto. Este prototipo permite cargar un documento, extraer requisitos técnicos, clasificar el requerimiento y recomendar una solución compatible desde una base de conocimiento editable.

## Arquitectura del sistema

```text
tdr-ai-analyzer/
├── app/
│   ├── main.py              # Interfaz Streamlit
│   ├── document_loader.py   # Extracción PDF, DOCX y TXT
│   ├── text_cleaner.py      # Limpieza y normalización
│   ├── chunker.py           # Fragmentación del texto
│   ├── embeddings.py        # OpenAI embeddings o fallback hash local
│   ├── vector_store.py      # Base vectorial local JSON
│   ├── knowledge_base.py    # Lectura de soluciones tecnológicas
│   ├── rag_engine.py        # Orquestación RAG y análisis estructurado
│   ├── classifier.py        # Clasificación tecnológica heurística
│   ├── recommender.py       # Recomendación por coincidencia semántica/simple
│   ├── evaluator.py         # Métricas de validación experta
│   ├── exporter.py          # Exportación Markdown, JSON y PDF
│   └── storage.py           # Registro SQLite
├── data/
│   ├── examples/            # TDR simulados
│   └── knowledge_base/      # solutions.json editable
├── outputs/                 # Reportes generados
├── tests/                   # Pruebas básicas
└── run.py                   # Lanzador de Streamlit
```

## Instalación

Requisitos:

- Python 3.11 o superior.
- Windows o Linux.

Crear y activar un entorno virtual:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Configuración

Copiar el archivo de ejemplo:

```bash
copy .env.example .env
```

En Linux/macOS:

```bash
cp .env.example .env
```

Variables:

```text
OPENAI_API_KEY=
MODEL_NAME=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small
VECTOR_STORE_PATH=data/processed/chroma
```

Si `OPENAI_API_KEY` está vacío, el sistema funciona en modo demo con embeddings hash locales y análisis heurístico estructurado.

## Ejecución local

```bash
python run.py
```

También puede ejecutarse directamente:

```bash
streamlit run app/main.py
```

## Uso de la aplicación

1. Abrir la app en Streamlit.
2. Cargar un documento PDF, DOCX o TXT.
3. Revisar el texto extraído.
4. Ejecutar el análisis automático.
5. Revisar resumen, objeto, requisitos, clasificación y recomendación.
6. Exportar el resultado en Markdown, JSON o PDF.
7. Registrar la validación experta para calcular métricas.

## Base de conocimiento

La base editable está en:

```text
data/knowledge_base/solutions.json
```

Incluye soluciones de referencia como Microsoft 365, Power BI, Google Workspace, Adobe, Sophos, Kaspersky, ESET, Fortinet, Zoom, hosting, VPS, backup cloud y certificados SSL.

## Ejemplo de análisis

Documentos simulados:

- `data/examples/tdr_microsoft_365.txt`
- `data/examples/tdr_edr_xdr.txt`
- `data/examples/tdr_power_bi.txt`
- `data/examples/tdr_backup_cloud.txt`

Generar salidas de ejemplo:

```bash
python scripts/generate_sample_outputs.py
```

Esto crea reportes en:

- `outputs/json/`
- `outputs/markdown/`
- `outputs/pdf/`

## Métricas de evaluación

El módulo de evaluación calcula:

- Precisión en extracción de requisitos.
- Recall de requisitos.
- F1-score.
- Exactitud de clasificación.
- Coincidencia con experto.
- Tiempo de análisis.
- Reducción estimada de tiempo frente a análisis manual.

## Limitaciones

- La recomendación es preliminar y requiere revisión técnica especializada.
- No realiza evaluación legal, contractual ni financiera.
- El modo demo usa heurísticas y no reemplaza un modelo generativo real.
- La calidad del análisis depende de la claridad del TDR y de la base de conocimiento.
- La base vectorial incluida es local y simple; puede sustituirse por ChromaDB o FAISS en una versión posterior.

## Próximas mejoras

- Integrar ChromaDB como backend vectorial principal.
- Añadir esquemas JSON estrictos para respuestas generativas.
- Mejorar extracción de tablas desde PDF.
- Añadir autenticación y gestión de usuarios.
- Incorporar retroalimentación experta para ajustar pesos de recomendación.
- Crear un conjunto de evaluación más amplio con TDR reales anonimizados.

## Seguridad y advertencias

El sistema debe usarse como apoyo al análisis técnico. No debe inventar soluciones si no existe evidencia suficiente; por ello muestra datos faltantes o ambiguos y justifica recomendaciones con fragmentos recuperados del documento y la base de conocimiento.
