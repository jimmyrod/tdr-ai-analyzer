from types import SimpleNamespace

from app.config import get_settings
from app.schemas import TextChunk
from app.vector_store import SupabaseVectorStore, VectorStore


class FakeTable:
    def __init__(self):
        self.deleted: list[dict] = []
        self.inserted: list[dict] = []

    def delete(self):
        return FakeDeleteQuery(self)

    def insert(self, rows):
        return FakeInsertQuery(self, rows)


class FakeDeleteQuery:
    def __init__(self, table: FakeTable):
        self.table = table
        self.filters: dict = {}

    def eq(self, column, value):
        self.filters[column] = value
        return self

    def execute(self):
        self.table.deleted.append(dict(self.filters))
        return SimpleNamespace(data=[])


class FakeInsertQuery:
    def __init__(self, table: FakeTable, rows: list[dict]):
        self.table = table
        self.rows = rows

    def execute(self):
        self.table.inserted.extend(self.rows)
        return SimpleNamespace(data=self.rows)


class FakeRpcQuery:
    def __init__(self, data: list[dict]):
        self._data = data

    def execute(self):
        return SimpleNamespace(data=self._data)


class FakeSupabaseClient:
    def __init__(self, rpc_data: list[dict] | None = None):
        self.tables: dict[str, FakeTable] = {}
        self.rpc_calls: list[tuple[str, dict]] = []
        self._rpc_data = rpc_data or []

    def table(self, name):
        return self.tables.setdefault(name, FakeTable())

    def rpc(self, name, params):
        self.rpc_calls.append((name, params))
        return FakeRpcQuery(self._rpc_data)


def _supabase_store(client: FakeSupabaseClient) -> SupabaseVectorStore:
    store = SupabaseVectorStore.__new__(SupabaseVectorStore)
    store.client = client
    return store


def test_supabase_vector_store_add_chunks_resets_and_inserts():
    client = FakeSupabaseClient()
    store = _supabase_store(client)
    chunks = [TextChunk(index=0, text="hola", start_char=0, end_char=4, metadata={"foo": "bar"})]

    store.add_chunks("doc.txt", chunks, [[0.1, 0.2]])

    table = client.tables[SupabaseVectorStore.TABLE]
    assert table.deleted == [{"document_name": "doc.txt"}]
    assert table.inserted == [
        {
            "id": "doc.txt:0",
            "document_name": "doc.txt",
            "chunk_index": 0,
            "text": "hola",
            "embedding": [0.1, 0.2],
            "metadata": {"foo": "bar"},
        }
    ]


def test_supabase_vector_store_similarity_search_maps_rows():
    client = FakeSupabaseClient(
        rpc_data=[
            {
                "id": "doc.txt:0",
                "document_name": "doc.txt",
                "chunk_index": 0,
                "text": "hola",
                "metadata": {"foo": "bar"},
                "similarity": 0.9,
            }
        ]
    )
    store = _supabase_store(client)

    results = store.similarity_search([0.1, 0.2], top_k=3, document_name="doc.txt")

    assert client.rpc_calls == [
        (
            SupabaseVectorStore.MATCH_FUNCTION,
            {"query_embedding": [0.1, 0.2], "match_count": 3, "filter_document_name": "doc.txt"},
        )
    ]
    assert len(results) == 1
    assert results[0].id == "doc.txt:0"
    assert results[0].text == "hola"
    assert results[0].metadata["document_name"] == "doc.txt"


def test_vector_store_falls_back_to_local_when_supabase_fails(tmp_path, monkeypatch):
    settings = get_settings()
    object.__setattr__(settings, "supabase_url", "http://fake-supabase")
    object.__setattr__(settings, "supabase_secret_key", "fake-key")

    store = VectorStore(tmp_path / "vectors", settings=settings)

    class BrokenRemote:
        def add_chunks(self, *args, **kwargs):
            raise RuntimeError("supabase unreachable")

        def similarity_search(self, *args, **kwargs):
            raise RuntimeError("supabase unreachable")

    monkeypatch.setattr(store, "_remote", lambda: BrokenRemote())

    chunks = [TextChunk(index=0, text="hola mundo", start_char=0, end_char=10, metadata={})]
    store.add_chunks("doc.txt", chunks, [[0.1] * 1536])

    results = store.similarity_search([0.1] * 1536, top_k=3, document_name="doc.txt")

    assert len(results) == 1
    assert results[0].text == "hola mundo"
