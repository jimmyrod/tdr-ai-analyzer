import json
from types import SimpleNamespace

from app.config import get_settings
from app.knowledge_base import KnowledgeBase


class FakeSelectQuery:
    def __init__(self, data):
        self._data = data

    def eq(self, column, value):
        return self

    def execute(self):
        return SimpleNamespace(data=self._data)


class FakeUpsertQuery:
    def __init__(self, table, row):
        self.table = table
        self.row = row

    def execute(self):
        self.table.upserted.append(self.row)
        return SimpleNamespace(data=[self.row])


class FakeTable:
    def __init__(self, data):
        self._data = data
        self.upserted = []

    def select(self, columns):
        return FakeSelectQuery(self._data)

    def upsert(self, row):
        return FakeUpsertQuery(self, row)


class FakeRpcQuery:
    def __init__(self, data):
        self._data = data

    def execute(self):
        return SimpleNamespace(data=self._data)


class FakeSupabaseClient:
    def __init__(self, data=None, raise_error=False, rpc_data=None):
        self._data = data or []
        self._raise_error = raise_error
        self._rpc_data = rpc_data or []
        self.rpc_calls = []
        self.tables: dict[str, FakeTable] = {}

    def table(self, name):
        if self._raise_error:
            raise RuntimeError("supabase unreachable")
        return self.tables.setdefault(name, FakeTable(self._data))

    def rpc(self, name, params):
        if self._raise_error:
            raise RuntimeError("supabase unreachable")
        self.rpc_calls.append((name, params))
        return FakeRpcQuery(self._rpc_data)


def _supabase_settings():
    settings = get_settings()
    object.__setattr__(settings, "supabase_url", "http://fake-supabase")
    object.__setattr__(settings, "supabase_secret_key", "fake-key")
    return settings


def test_knowledge_base_loads_from_supabase(monkeypatch, tmp_path):
    settings = _supabase_settings()
    row = {
        "id": "solucion_001",
        "nombre": "Backup cloud",
        "categoria": "Backup y recuperación",
        "descripcion": "Servicio de respaldo.",
        "caracteristicas_principales": ["copias automaticas"],
        "requisitos_que_cubre": ["backup"],
        "restricciones": [],
        "modalidad": "servicio",
        "observaciones": "",
    }
    fake_client = FakeSupabaseClient(data=[row])
    monkeypatch.setattr("supabase.create_client", lambda url, key: fake_client)

    kb = KnowledgeBase.load(tmp_path / "solutions.json", settings=settings)

    assert len(kb.all()) == 1
    assert kb.all()[0].nombre == "Backup cloud"


def test_knowledge_base_falls_back_to_local_json_on_supabase_error(monkeypatch, tmp_path):
    settings = _supabase_settings()
    fake_client = FakeSupabaseClient(raise_error=True)
    monkeypatch.setattr("supabase.create_client", lambda url, key: fake_client)

    local_path = tmp_path / "solutions.json"
    local_path.write_text(
        json.dumps(
            [
                {
                    "id": "solucion_local",
                    "nombre": "Solucion local",
                    "categoria": "Otro",
                    "descripcion": "desc",
                    "caracteristicas_principales": [],
                    "requisitos_que_cubre": [],
                    "restricciones": [],
                    "modalidad": "servicio",
                    "observaciones": "",
                }
            ]
        ),
        encoding="utf-8",
    )

    kb = KnowledgeBase.load(local_path, settings=settings)

    assert len(kb.all()) == 1
    assert kb.all()[0].nombre == "Solucion local"


def test_knowledge_base_search_by_vector_maps_rows_with_similarity(monkeypatch):
    settings = _supabase_settings()
    row = {
        "id": "solucion_001",
        "nombre": "Backup cloud",
        "categoria": "Backup y recuperación",
        "descripcion": "Servicio de respaldo.",
        "caracteristicas_principales": ["copias automaticas"],
        "requisitos_que_cubre": ["backup"],
        "restricciones": [],
        "modalidad": "servicio",
        "observaciones": "",
        "similarity": 0.87,
    }
    fake_client = FakeSupabaseClient(rpc_data=[row])
    monkeypatch.setattr("supabase.create_client", lambda url, key: fake_client)

    kb = KnowledgeBase([])
    results = kb.search_by_vector([0.1, 0.2], settings, top_k=3)

    assert fake_client.rpc_calls == [
        (
            "match_solutions",
            {"query_embedding": [0.1, 0.2], "match_count": 3, "filter_origen": None},
        )
    ]
    assert len(results) == 1
    solution, similarity = results[0]
    assert solution.nombre == "Backup cloud"
    assert similarity == 0.87


def test_knowledge_base_add_solution_upserts_row(monkeypatch):
    settings = _supabase_settings()
    fake_client = FakeSupabaseClient()
    monkeypatch.setattr("supabase.create_client", lambda url, key: fake_client)

    kb = KnowledgeBase([])
    row = {
        "id": "analisis:tdr_backup.txt",
        "nombre": "Backup para servidores",
        "categoria": "Backup y recuperación",
        "descripcion": "resumen",
        "caracteristicas_principales": [],
        "requisitos_que_cubre": [],
        "restricciones": [],
        "modalidad": "",
        "observaciones": "",
        "origen": "analisis",
        "embedding": [0.1, 0.2],
    }

    kb.add_solution(row, settings)

    assert fake_client.tables["solutions"].upserted == [row]
