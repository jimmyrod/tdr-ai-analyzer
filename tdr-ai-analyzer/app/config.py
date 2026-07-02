from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs without requiring python-dotenv."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Settings:
    project_root: Path
    uploads_dir: Path
    processed_dir: Path
    knowledge_base_path: Path
    evaluations_dir: Path
    outputs_markdown_dir: Path
    outputs_json_dir: Path
    outputs_pdf_dir: Path
    database_path: Path
    vector_store_path: Path
    openai_api_key: str
    model_name: str
    embedding_model: str
    chunk_size: int = 1200
    chunk_overlap: int = 180
    manual_analysis_minutes: float = 60.0

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key.strip())


def get_settings() -> Settings:
    root = Path(__file__).resolve().parents[1]
    _load_env_file(root / ".env")
    _load_env_file(root / ".env.local")

    vector_path = Path(os.getenv("VECTOR_STORE_PATH", "data/processed/chroma"))
    if not vector_path.is_absolute():
        vector_path = root / vector_path

    return Settings(
        project_root=root,
        uploads_dir=root / "data" / "uploads",
        processed_dir=root / "data" / "processed",
        knowledge_base_path=root / "data" / "knowledge_base" / "solutions.json",
        evaluations_dir=root / "data" / "evaluations",
        outputs_markdown_dir=root / "outputs" / "markdown",
        outputs_json_dir=root / "outputs" / "json",
        outputs_pdf_dir=root / "outputs" / "pdf",
        database_path=root / "data" / "evaluations" / "tdr_ai_analyzer.db",
        vector_store_path=vector_path,
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        model_name=os.getenv("MODEL_NAME", "gpt-4.1-mini").strip() or "gpt-4.1-mini",
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
        or "text-embedding-3-small",
    )


def ensure_directories(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    for path in [
        settings.uploads_dir,
        settings.processed_dir,
        settings.knowledge_base_path.parent,
        settings.evaluations_dir,
        settings.outputs_markdown_dir,
        settings.outputs_json_dir,
        settings.outputs_pdf_dir,
        settings.vector_store_path,
    ]:
        path.mkdir(parents=True, exist_ok=True)
