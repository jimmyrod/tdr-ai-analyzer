# TDR AI Analyzer

Prototipo académico para analizar términos de referencia tecnológicos mediante procesamiento de lenguaje natural, recuperación aumentada con generación (RAG) y recomendación preliminar de soluciones tecnológicas.

## Problema que resuelve

El análisis de TDR tecnológicos suele hacerse manualmente sobre documentos PDF, Word o texto. Este prototipo permite cargar un documento, extraer requisitos técnicos, clasificar el requerimiento y recomendar una solución compatible desde una base de conocimiento que vive en Supabase y crece con cada análisis.

## Arquitectura del sistema

```text
tdr-ai-analyzer/
├── app/
│   ├── main.py              # Interfaz Streamlit
│   ├── document_loader.py   # Extracción PDF, DOCX y TXT
│   ├── text_cleaner.py      # Limpieza y normalización
│   ├── chunker.py           # Fragmentación del texto
│   ├── embeddings.py        # OpenAI embeddings o fallback hash local
│   ├── vector_store.py      # Fragmentos de TDR en Supabase (pgvector), con fallback JSON local
│   ├── knowledge_base.py    # Catálogo de soluciones en Supabase, con fallback a solutions.json
│   ├── rag_engine.py        # Orquestación RAG, análisis estructurado y alimentación del catálogo
│   ├── classifier.py        # Clasificación tecnológica heurística
│   ├── recommender.py       # Recomendación por búsqueda vectorial o por coincidencia de palabras clave
│   ├── evaluator.py         # Métricas de validación experta
│   ├── exporter.py          # Exportación Markdown, JSON y PDF
│   └── storage.py           # Registro SQLite
├── supabase/
│   └── schema.sql           # Extensión vector, tablas, índices y funciones RPC (correr en Supabase Studio)
├── scripts/
│   ├── migrate_knowledge_base.py   # Sube/actualiza data/knowledge_base/solutions.json en Supabase
│   └── generate_sample_outputs.py  # Genera reportes de ejemplo
├── data/
│   ├── examples/            # TDR simulados
│   └── knowledge_base/      # solutions.json (catálogo original / fallback local)
├── outputs/                 # Reportes generados
├── tests/                   # Pruebas (pytest)
├── Dockerfile / docker-compose.yml
└── run.py                   # Lanzador de Streamlit
```

## Instalación

Requisitos:

- Python 3.11 o superior.
- Windows o Linux.
- Opcional: Docker, si prefieres correr todo en contenedor.

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

### Con Docker

```bash
docker compose up --build -d
```

Levanta la app en `http://localhost:8501` usando las variables de `.env` (ver siguiente sección) y monta `data/` y `outputs/` como volúmenes. Cada vez que cambie el código o `requirements.txt` hay que reconstruir la imagen (`docker compose up --build -d`); solo reiniciar el contenedor no recoge los cambios.

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
SUPABASE_URL=
SUPABASE_SECRET_KEY=
```

- Si `OPENAI_API_KEY` está vacío (o la key falla), el sistema funciona en modo demo con embeddings hash locales y análisis heurístico estructurado, marcando `proveedor_ia: "Demo local"` en el resultado.
- Si `SUPABASE_URL` o `SUPABASE_SECRET_KEY` están vacíos, o Supabase no responde, todo lo relacionado a Supabase cae automáticamente a un almacenamiento local (JSON o `solutions.json`) sin interrumpir el análisis.

## Supabase

El proyecto usa Postgres + [pgvector](https://github.com/pgvector/pgvector) en Supabase (self-hosted o cloud) como base vectorial para dos cosas independientes: los fragmentos de cada TDR analizado y el catálogo de soluciones recomendables.

### 1. Crear el esquema (una sola vez)

En Supabase Studio → **SQL Editor**, ejecutar el contenido completo de [`supabase/schema.sql`](supabase/schema.sql). Es idempotente (usa `if not exists` / `create or replace` donde aplica), así que volver a correrlo no rompe nada. Crea:

- La extensión `vector`.
- `document_chunks`: fragmentos embebidos de cada TDR analizado, con la función RPC `match_document_chunks` para búsqueda por similitud.
- `solutions`: catálogo de soluciones tecnológicas, con la función RPC `match_solutions`.
- Row Level Security habilitado en ambas tablas, sin políticas para `anon`/`publishable` — solo el backend, usando la `service_role` (`SUPABASE_SECRET_KEY`, que hace bypass de RLS), puede leer o escribir.

### 2. Completar `.env`

- `SUPABASE_URL`: URL de tu API (ej. `http://tu-servidor:8000`).
- `SUPABASE_SECRET_KEY`: la **secret key** (`service_role`), no la publishable/anon. Se usa solo del lado del servidor; nunca debe exponerse en el navegador ni commitearse (`.env` está en `.gitignore`).

### 3. `document_chunks`: memoria de los TDR analizados

Cada vez que se analiza un documento, `RAGEngine` lo trocea, genera embeddings (`app/embeddings.py`) y los guarda en `document_chunks` vía `VectorStore` (`app/vector_store.py`). La búsqueda de similitud para armar la evidencia ("fragmentos recuperados") consulta **toda** la tabla, no solo el documento actual — así el histórico de TDR ya analizados se usa como contexto de análisis futuros.

### 4. `solutions`: catálogo de soluciones

`KnowledgeBase` (`app/knowledge_base.py`) lee y escribe la tabla `solutions`. Cada fila tiene una columna `origen`:

- `origen = 'catalogo'`: las 18 soluciones curadas originales (Microsoft 365, Power BI, Sophos, FortiGate, etc.), definidas en `data/knowledge_base/solutions.json`.
- `origen = 'analisis'`: una fila por cada TDR analizado, con lo que ese documento pedía (objeto, categoría, requisitos). `RAGEngine._feed_solutions_from_analysis` la inserta automáticamente al final de cada análisis (`upsert` por `id = "analisis:<nombre_documento>"`, así reanalizar el mismo archivo actualiza su fila en vez de duplicarla).

Ambos orígenes se buscan y recomiendan juntos (la función `match_solutions` acepta un parámetro `filter_origen`, pero la app lo llama con `null` para no separarlos) — la idea es que el catálogo crezca con casos reales, no solo con el listado curado inicial.

Para cargar/actualizar el catálogo original en Supabase (con sus embeddings) desde `data/knowledge_base/solutions.json`:

```bash
python scripts/migrate_knowledge_base.py
```

Solo hace falta correrlo una vez, o cuando se edite `solutions.json` a mano. Requiere `SUPABASE_URL`/`SUPABASE_SECRET_KEY` configurados; usa `OPENAI_API_KEY` para generar embeddings reales (si no hay key válida, cae al embedding hash local — más débil semánticamente).

### 5. Recomendación por búsqueda vectorial

`RecommendationEngine.recommend_by_vector` (`app/recommender.py`) construye un embedding a partir de la categoría, el objeto y los requisitos del TDR, y llama a `match_solutions` para traer las soluciones más cercanas semánticamente. Si Supabase no está configurado, falla, o no encuentra nada, `RAGEngine` cae automáticamente al método anterior por coincidencia de palabras clave (`RecommendationEngine.recommend`) sobre el catálogo local.

**Importante:** este camino de búsqueda vectorial solo se ejecuta en el modo de análisis heurístico (cuando no hay `OPENAI_API_KEY` válida o la llamada a OpenAI falla). Cuando el análisis se resuelve con éxito vía OpenAI Responses API, es el modelo el que elige la solución directamente (recibe el catálogo como texto en el prompt), sin pasar por `match_solutions`.

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
8. En la pestaña **Base de conocimiento** se puede ver el catálogo completo (fuente Supabase o `solutions.json` local, según configuración), con una tabla resumen y el detalle de cada solución.

## Base de conocimiento

El catálogo vive en Supabase (tabla `solutions`) cuando está configurado; si no, cae a:

```text
data/knowledge_base/solutions.json
```

Incluye 18 soluciones de referencia (Microsoft 365, Power BI, Google Workspace, Adobe, Sophos, Kaspersky, ESET, Fortinet, Zoom, hosting, VPS, backup cloud, certificados SSL) más una fila por cada TDR que se haya analizado con Supabase configurado. Para agregar una solución nueva al catálogo curado hoy: editar `solutions.json` y correr `scripts/migrate_knowledge_base.py`; no hay todavía una forma de hacerlo desde el frontend.

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
- La base vectorial usa Supabase/pgvector; sin configurarlo, cae a almacenamiento local simple (JSON).
- La búsqueda vectorial de soluciones solo corre en el modo heurístico (ver sección Supabase); con OpenAI activo, la recomendación la decide el modelo directamente.
- Si el catálogo se migró a Supabase sin una `OPENAI_API_KEY` válida, sus embeddings quedaron generados con el fallback hash local — semánticamente más débiles que embeddings reales. Volver a correr `scripts/migrate_knowledge_base.py` con una key válida los regenera.

## Próximas mejoras

- Hacer que el camino de análisis con OpenAI también consulte `match_solutions`, en vez de que el modelo recomiende solo a partir del texto del prompt.
- Permitir agregar/editar soluciones del catálogo desde el frontend, sin pasar por el script de migración.
- Añadir esquemas JSON estrictos para respuestas generativas.
- Mejorar extracción de tablas desde PDF.
- Añadir autenticación y gestión de usuarios.
- Incorporar retroalimentación experta para ajustar pesos de recomendación.
- Crear un conjunto de evaluación más amplio con TDR reales anonimizados.

## Seguridad y advertencias

El sistema debe usarse como apoyo al análisis técnico. No debe inventar soluciones si no existe evidencia suficiente; por ello muestra datos faltantes o ambiguos y justifica recomendaciones con fragmentos recuperados del documento y la base de conocimiento.

La `SUPABASE_SECRET_KEY` (`service_role`) tiene acceso total a la base de datos, sin restricciones de Row Level Security. Debe tratarse como cualquier otro secreto de servidor: solo en `.env` (nunca commiteado), nunca expuesta al navegador ni a un cliente.
