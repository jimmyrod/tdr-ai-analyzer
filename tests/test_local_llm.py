from app.local_llm import LocalLLMClient


def test_local_llm_client_parses_ollama_json_response(monkeypatch):
    calls = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": '{"resumen_general":"Resumen local","categoria_tecnologica":"Cloud / SaaS"}'
            }

    def fake_post(url, json, timeout, headers=None):
        calls["url"] = url
        calls["json"] = json
        calls["timeout"] = timeout
        calls["headers"] = headers or {}
        return FakeResponse()

    client = LocalLLMClient(model_name="qwen2.5:7b", base_url="http://localhost:11434")
    monkeypatch.setattr(client, "_post", fake_post)

    payload = client.generate_json("Responde JSON.")

    assert payload["resumen_general"] == "Resumen local"
    assert payload["categoria_tecnologica"] == "Cloud / SaaS"
    assert calls["url"] == "http://localhost:11434/api/generate"
    assert calls["json"]["model"] == "qwen2.5:7b"
    assert calls["json"]["format"] == "json"
    assert calls["json"]["stream"] is False
    assert "Authorization" not in calls["headers"]


def test_local_llm_client_accepts_json_schema_for_structured_generation(monkeypatch):
    calls = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"response": '{"resumen_general":"Resumen con esquema"}'}

    def fake_post(url, json, timeout, headers=None):
        calls["json"] = json
        return FakeResponse()

    schema = {
        "type": "object",
        "properties": {"resumen_general": {"type": "string"}},
        "required": ["resumen_general"],
    }
    client = LocalLLMClient(model_name="qwen2.5:7b", base_url="http://localhost:11434")
    monkeypatch.setattr(client, "_post", fake_post)

    payload = client.generate_json("Responde JSON.", json_schema=schema)

    assert payload["resumen_general"] == "Resumen con esquema"
    assert calls["json"]["format"] == schema


def test_local_llm_client_sends_bearer_token_for_ollama_cloud(monkeypatch):
    calls = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"response": '{"resumen_general":"Resumen cloud"}'}

    def fake_post(url, json, timeout, headers=None):
        calls["url"] = url
        calls["headers"] = headers or {}
        calls["json"] = json
        return FakeResponse()

    client = LocalLLMClient(
        model_name="gpt-oss:120b",
        base_url="https://ollama.com",
        api_key="ollama-test-key",
        service_name="Ollama Cloud",
    )
    monkeypatch.setattr(client, "_post", fake_post)

    payload = client.generate_json("Responde JSON.")

    assert payload["resumen_general"] == "Resumen cloud"
    assert calls["url"] == "https://ollama.com/api/generate"
    assert calls["headers"]["Authorization"] == "Bearer ollama-test-key"
    assert calls["json"]["model"] == "gpt-oss:120b"


def test_local_llm_client_reports_connection_error(monkeypatch):
    client = LocalLLMClient(model_name="qwen2.5:7b", base_url="http://localhost:11434")

    def fake_post(url, json, timeout, headers=None):
        raise OSError("connection refused")

    monkeypatch.setattr(client, "_post", fake_post)

    try:
        client.generate_json("Responde JSON.")
    except RuntimeError as exc:
        assert "Ollama no respondió" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")
