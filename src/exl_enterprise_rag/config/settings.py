"""Application configuration.

Two sources of truth:
  - config/departments.yaml  → the ingestion allowlist (business config)
  - .env                     → secrets and environment (runtime config)

Department config is validated at load time, so a typo in the YAML fails
fast rather than producing chunks that no partial index covers.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------

# settings.py is at src/exl_enterprise_rag/config/settings.py
# parents[0] = config/
# parents[1] = exl_enterprise_rag/
# parents[2] = src/
# parents[3] = repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = Path(__file__).resolve().parent
DEPARTMENTS_YAML = CONFIG_DIR / "departments.yaml"


# ---------------------------------------------------------------
# Department configuration (YAML-driven)
# ---------------------------------------------------------------

class DepartmentConfig(BaseModel):
    """The department allowlist and folder→department mapping."""

    departments: list[str] = Field(
        ..., description="Canonical department names (one per partial index)."
    )
    folder_to_department: dict[str, str] = Field(
        ..., description="Filesystem folder → canonical department label."
    )
    excluded_folders: list[str] = Field(
        default_factory=list,
        description="Folders explicitly excluded (documentation only).",
    )

    @field_validator("departments")
    @classmethod
    def _no_duplicates(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            dupes = sorted({d for d in v if v.count(d) > 1})
            raise ValueError(f"duplicate departments in allowlist: {dupes}")
        return v

    def canonical(self) -> frozenset[str]:
        return frozenset(self.departments)

    def is_allowed_folder(self, folder_name: str) -> bool:
        return folder_name in self.folder_to_department

    def department_for(self, folder_name: str) -> str | None:
        return self.folder_to_department.get(folder_name)

    def allowed_folders(self) -> frozenset[str]:
        return frozenset(self.folder_to_department.keys())


def load_department_config(path: Path | None = None) -> DepartmentConfig:
    """Load and validate config/departments.yaml.

    Raises:
        FileNotFoundError: if the YAML file is missing.
        pydantic.ValidationError: if the YAML structure is wrong.
        ValueError: if folder_to_department maps to an unknown department.
    """
    yaml_path = path or DEPARTMENTS_YAML
    with yaml_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = DepartmentConfig(**raw)

    bad_targets = {
        folder: target
        for folder, target in cfg.folder_to_department.items()
        if target not in cfg.canonical()
    }
    if bad_targets:
        raise ValueError(
            f"folder_to_department maps to unknown departments: {bad_targets}. "
            f"Valid departments: {sorted(cfg.canonical())}"
        )

    return cfg


# ---------------------------------------------------------------
# Runtime settings (env-driven)
# ---------------------------------------------------------------

class Settings(BaseSettings):
    """Environment-driven settings. Values come from .env / process env."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Database ---
    database_url: str = Field(
        ..., description="postgresql://user:pass@host:port/db"
    )

    # --- Corpus location ---
    handbook_root: Path = Field(
        default=Path("../handbook"),
        description="Path to the cloned GitLab Handbook repo.",
    )

    # --- Embeddings ---
    embedding_provider: str = Field(default="stub")
    openai_api_key: str | None = None
    openai_embedding_model: str = Field(default="text-embedding-3-small")
    local_embedding_model: str = Field(default="BAAI/bge-large-en-v1.5")

    # --- Logging ---
    log_level: str = Field(default="INFO")

    # --- Chunking ---
    chunk_target_tokens: int = Field(default=512, ge=64, le=4096)
    chunk_overlap_tokens: int = Field(default=64, ge=0, le=512)
    min_chunk_tokens: int = Field(default=20, ge=0)

    @property
    def handbook_content_dir(self) -> Path:
        """Directory containing department folders inside the clone."""
        return self.handbook_root / "content" / "handbook"


# ---------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------

_settings: Settings | None = None
_departments: DepartmentConfig | None = None


def get_settings() -> Settings:
    """Return singleton Settings (loads .env on first call)."""
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings


def get_departments() -> DepartmentConfig:
    """Return singleton DepartmentConfig (loads YAML on first call)."""
    global _departments
    if _departments is None:
        _departments = load_department_config()
    return _departments


def reset_caches() -> None:
    """Clear singletons. Tests only."""
    global _settings, _departments
    _settings = None
    _departments = None