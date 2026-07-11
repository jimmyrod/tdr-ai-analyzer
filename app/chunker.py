from __future__ import annotations

from dataclasses import dataclass

from app.schemas import TextChunk


@dataclass
class TextChunker:
    chunk_size: int = 1200
    overlap: int = 180

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size debe ser mayor que cero.")
        if self.overlap < 0:
            raise ValueError("overlap no puede ser negativo.")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap debe ser menor que chunk_size.")

    def split(self, text: str, source_name: str | None = None) -> list[TextChunk]:
        if not text:
            return []

        chunks: list[TextChunk] = []
        start = 0
        index = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            if end < text_length:
                boundary = text.rfind(" ", start + int(self.chunk_size * 0.6), end)
                if boundary > start:
                    end = boundary

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    TextChunk(
                        index=index,
                        text=chunk_text,
                        start_char=start,
                        end_char=end,
                        metadata={"source": source_name or "", "chunk_index": index},
                    )
                )
                index += 1

            if end >= text_length:
                break
            start = max(0, end - self.overlap)

        return chunks
