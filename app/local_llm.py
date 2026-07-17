from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class LocalLLMClient:
    """Ollama client for local structured generation."""

    model_name: str = "qwen2.5:7b"
    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 180
    api_key: str = ""
    service_name: str = "Ollama"

    def generate_json(self, prompt: str, json_schema: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "format": json_schema or "json",
            "stream": False,
            "options": {
                "temperature": 0,
                "top_p": 0.2,
                "num_ctx": 8192,
            },
        }

        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = self._post(
                f"{self.base_url.rstrip('/')}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
                headers=headers,
            )
            response.raise_for_status()
            raw_payload = response.json()
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                raise RuntimeError(
                    f"{self.service_name} rechazó la autenticación. "
                    "Verifique la API key configurada."
                ) from exc
            if exc.code == 404:
                raise RuntimeError(
                    f"El modelo '{self.model_name}' no está disponible en {self.service_name}."
                ) from exc
            raise RuntimeError(f"{self.service_name} devolvió HTTP {exc.code}.") from exc
        except Exception as exc:
            raise RuntimeError(
                f"{self.service_name} no respondió. Verifique la URL configurada: "
                f"{self.base_url.rstrip('/')}."
            ) from exc

        content = raw_payload.get("response", "")
        if not content:
            raise RuntimeError(f"{self.service_name} respondió sin contenido generado.")

        try:
            return json.loads(self._extract_json(content))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"El modelo en {self.service_name} no devolvió JSON válido.") from exc

    def _post(
        self,
        url: str,
        json: dict[str, Any],
        timeout: int,
        headers: dict[str, str] | None = None,
    ):
        request = urllib.request.Request(
            url,
            data=json_dumps(json).encode("utf-8"),
            headers=headers or {"Content-Type": "application/json"},
            method="POST",
        )
        return _UrllibResponse(urllib.request.urlopen(request, timeout=timeout))

    def _extract_json(self, content: str) -> str:
        match = re.search(r"\{.*\}", content, flags=re.S)
        return match.group(0) if match else content


class _UrllibResponse:
    def __init__(self, response):
        self._response = response
        self.status_code = getattr(response, "status", 200)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise urllib.error.HTTPError(
                url="",
                code=self.status_code,
                msg="Ollama HTTP error",
                hdrs=None,
                fp=None,
            )

    def json(self) -> dict[str, Any]:
        return json.loads(self._response.read().decode("utf-8"))


def json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)
