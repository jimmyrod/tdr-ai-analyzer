from __future__ import annotations

import re
import unicodedata


def clean_text(text: str) -> str:
    """Normalize extracted document text while preserving readable line breaks."""
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", normalized)

    cleaned_lines: list[str] = []
    for line in normalized.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def normalize_for_matching(text: str) -> str:
    """Lowercase and remove accents for robust keyword matching."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9ñáéíóúü\s/.-]", " ", text)
