from __future__ import annotations

import shutil
from pathlib import Path

from app.config import Settings, get_settings
from app.text_cleaner import clean_text


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def save_uploaded_file(uploaded_file, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.uploads_dir / uploaded_file.name
    with destination.open("wb") as output:
        output.write(uploaded_file.getbuffer())
    return destination


def load_document(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el documento: {path}")
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Formato no soportado: {extension}. Use PDF, DOCX o TXT.")

    if extension == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
    elif extension == ".pdf":
        text = _load_pdf(path)
    else:
        text = _load_docx(path)
    return clean_text(text)


def copy_example_to_uploads(path: Path, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.uploads_dir / path.name
    shutil.copy2(path, destination)
    return destination


def _load_pdf(path: Path) -> str:
    try:
        import fitz

        document = fitz.open(path)
        return "\n".join(page.get_text("text") for page in document)
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF no esta instalado. Instale dependencias con: pip install -r requirements.txt"
        ) from exc


def _load_docx(path: Path) -> str:
    try:
        from docx import Document

        document = Document(path)
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    except ImportError as exc:
        raise RuntimeError(
            "python-docx no esta instalado. Instale dependencias con: pip install -r requirements.txt"
        ) from exc
