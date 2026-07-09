from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.text_cleaner import normalize_for_matching


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if not norm_left or not norm_right:
        return 0.0
    return dot / (norm_left * norm_right)


@dataclass
class EmbeddingProvider:
    settings: Settings | None = None
    dimensions: int = 1536

    def __post_init__(self) -> None:
        self.settings = self.settings or get_settings()

    @property
    def using_openai(self) -> bool:
        return bool(self.settings and self.settings.has_openai_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.settings and self.settings.has_openai_key:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=self.settings.openai_api_key)
                response = client.embeddings.create(
                    model=self.settings.embedding_model,
                    input=texts,
                )
                return [item.embedding for item in response.data]
            except Exception:
                # Keep the academic demo usable even when API, network, or package setup fails.
                pass
        return [self._hash_embedding(text) for text in texts]

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _hash_embedding(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = [token for token in normalize_for_matching(text).split() if token]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [round(value / norm, 6) for value in vector]
        return vector
