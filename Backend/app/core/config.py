"""Typed application configuration.

Every tunable lives here so that no value is hardcoded in a service or model.
Values are read from the environment (optionally seeded by a ``.env`` file).

Security note: this module must never log or echo secret values. It only
exposes operational limits.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from typing_extensions import Annotated

# Project root = .../CS_Project (the directory that contains backend/ and frontend/)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"


class Settings(BaseSettings):
    """Application settings resolved from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    model_dir: Path = Field(default=BACKEND_ROOT / "checkpoints")
    data_dir: Path = Field(default=BACKEND_ROOT / "data")
    reports_dir: Path = Field(default=BACKEND_ROOT / "reports")
    logs_dir: Path = Field(default=BACKEND_ROOT / "logs")
    phiusiil_csv: Path = Field(default=PROJECT_ROOT / "Dataset" / "Phishing_URL_Dataset.csv")

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    device: str = Field(default="auto")
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    # ------------------------------------------------------------------
    # URL input limits
    # ------------------------------------------------------------------
    max_url_length: int = Field(default=2048, ge=1, le=65536)
    # NoDecode stops pydantic-settings from trying to JSON-parse these complex
    # fields out of the environment; the ``_split_csv`` validator below turns
    # "http,https" into a tuple instead.
    allowed_schemes: Annotated[tuple[str, ...], NoDecode] = ("http", "https")

    # ------------------------------------------------------------------
    # Secure fetcher
    # ------------------------------------------------------------------
    request_timeout: float = Field(default=10.0, gt=0)
    connect_timeout: float = Field(default=5.0, gt=0)
    max_response_size_mb: float = Field(default=5.0, gt=0)
    max_redirects: int = Field(default=5, ge=0, le=20)
    allowed_ports: Annotated[tuple[int, ...], NoDecode] = (80, 443)
    allowed_content_types: Annotated[tuple[str, ...], NoDecode] = ("text/html", "application/xhtml+xml")
    html_max_chars: int = Field(default=200_000, ge=1_000)
    fetch_delay_seconds: float = Field(default=1.0, ge=0)
    fetch_concurrency: int = Field(default=8, ge=1, le=64)
    fetch_max_per_domain: int = Field(default=1, ge=1)

    # ------------------------------------------------------------------
    # Screenshot service
    # ------------------------------------------------------------------
    screenshot_timeout: int = Field(default=20_000, ge=1_000)
    screenshot_width: int = Field(default=1280, ge=320, le=3840)
    screenshot_height: int = Field(default=800, ge=240, le=2160)
    screenshot_nav_timeout_ms: int = Field(default=15_000, ge=1_000)
    screenshot_service_url: str = Field(default="http://screenshot-service:8000")

    # ------------------------------------------------------------------
    # Model input limits
    # ------------------------------------------------------------------
    url_max_length: int = Field(default=256, ge=32, le=1024)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    rate_limit_per_minute: int = Field(default=10, ge=1)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    database_url: str = Field(default="sqlite:///./data/app.db")
    allow_store_artifacts: bool = Field(default=False)
    return_artifacts_inline: bool = Field(default=True)

    # ------------------------------------------------------------------
    # Collection policy
    # ------------------------------------------------------------------
    respect_robots: bool = Field(default=True)
    collection_user_agent: str = Field(
        default="MMRESearchBot/1.0 (+academic research; contact: set-your-email@example.com)"
    )

    # ------------------------------------------------------------------
    # Validators / coercion
    # ------------------------------------------------------------------
    @field_validator("allowed_schemes", "allowed_content_types", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        """Accept ``a,b,c`` from the environment as a tuple."""
        if isinstance(v, str):
            return tuple(part.strip().lower() for part in v.split(",") if part.strip())
        return v

    @field_validator("allowed_ports", mode="before")
    @classmethod
    def _parse_ports(cls, v: object) -> object:
        if isinstance(v, str):
            return tuple(int(part.strip()) for part in v.split(",") if part.strip())
        if isinstance(v, (list, tuple)):
            return tuple(int(p) for p in v)
        return v

    @field_validator("device")
    @classmethod
    def _check_device(cls, v: str) -> str:
        allowed = {"auto", "cpu", "cuda"}
        value = v.strip().lower()
        if value not in allowed:
            raise ValueError(f"DEVICE must be one of {sorted(allowed)}, got {v!r}")
        return value

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    @property
    def splits_dir(self) -> Path:
        """Where grouped split index files are written."""
        return self.data_dir / "splits"

    @property
    def manifest_path(self) -> Path:
        """Multimodal acquisition manifest."""
        return self.data_dir / "manifest.csv"

    def ensure_dirs(self) -> None:
        """Create the writable directories the app needs."""
        for d in (self.model_dir, self.data_dir, self.reports_dir, self.logs_dir, self.splits_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton.

    Cached so that the whole process observes one consistent configuration.
    Tests may call ``get_settings.cache_clear()`` after patching the env.
    """
    return Settings()


def resolve_device(requested: str | None = None) -> str:
    """Resolve the torch device string.

    Returns ``"cuda"`` only when torch actually reports CUDA availability, so a
    misconfigured ``DEVICE=cuda`` degrades to CPU with a warning instead of
    crashing at the first tensor allocation.
    """
    import torch

    choice = (requested or os.getenv("DEVICE") or get_settings().device).strip().lower()
    cuda_ok = torch.cuda.is_available()
    if choice == "cuda" and not cuda_ok:
        return "cpu"
    if choice == "auto":
        return "cuda" if cuda_ok else "cpu"
    return choice
