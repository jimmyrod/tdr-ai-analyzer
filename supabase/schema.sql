-- TDR AI Analyzer: almacen vectorial de fragmentos analizados.
-- Ejecutar una sola vez en Supabase Studio -> SQL Editor.

create extension if not exists vector;

create table if not exists document_chunks (
  id text primary key,
  document_name text not null,
  chunk_index int not null,
  text text not null,
  embedding vector(1536),
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists document_chunks_embedding_idx
  on document_chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);

create index if not exists document_chunks_document_name_idx
  on document_chunks (document_name);

create or replace function match_document_chunks(
  query_embedding vector(1536),
  match_count int default 5,
  filter_document_name text default null
)
returns table (
  id text,
  document_name text,
  chunk_index int,
  text text,
  metadata jsonb,
  similarity float
)
language sql stable as $$
  select
    id, document_name, chunk_index, text, metadata,
    1 - (embedding <=> query_embedding) as similarity
  from document_chunks
  where filter_document_name is null or document_name = filter_document_name
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- RLS: solo el backend (service_role, que hace bypass de RLS) puede leer/escribir.
-- No se agregan politicas para anon/publishable, asi el endpoint publico
-- no puede leer ni insertar en esta tabla.
alter table document_chunks enable row level security;

-- Catalogo de soluciones tecnologicas (antes data/knowledge_base/solutions.json).
-- origen='catalogo': las 18 soluciones curadas originales.
-- origen='analisis': una fila por cada TDR analizado (lo que ese TDR pedia),
-- para que la base crezca con casos reales. La app llama a match_solutions
-- con filter_origen=null, asi que ambos origenes se buscan/recomiendan juntos;
-- el parametro sigue disponible para filtrar por uno solo si hace falta.
create table if not exists solutions (
  id text primary key,
  nombre text not null,
  categoria text not null,
  descripcion text not null,
  caracteristicas_principales jsonb not null default '[]'::jsonb,
  requisitos_que_cubre jsonb not null default '[]'::jsonb,
  restricciones jsonb not null default '[]'::jsonb,
  modalidad text not null default '',
  observaciones text not null default '',
  origen text not null default 'catalogo',
  embedding vector(1536),
  created_at timestamptz not null default now()
);

alter table solutions add column if not exists origen text not null default 'catalogo';

create index if not exists solutions_embedding_idx
  on solutions using ivfflat (embedding vector_cosine_ops) with (lists = 100);

drop function if exists match_solutions(vector, int);

create function match_solutions(
  query_embedding vector(1536),
  match_count int default 5,
  filter_origen text default 'catalogo'
)
returns table (
  id text,
  nombre text,
  categoria text,
  descripcion text,
  caracteristicas_principales jsonb,
  requisitos_que_cubre jsonb,
  restricciones jsonb,
  modalidad text,
  observaciones text,
  origen text,
  similarity float
)
language sql stable as $$
  select
    id, nombre, categoria, descripcion, caracteristicas_principales,
    requisitos_que_cubre, restricciones, modalidad, observaciones, origen,
    1 - (embedding <=> query_embedding) as similarity
  from solutions
  where filter_origen is null or origen = filter_origen
  order by embedding <=> query_embedding
  limit match_count;
$$;

alter table solutions enable row level security;
