from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Academic AutoML Platform"
    app_version: str = "0.1.0"
    api_v1_prefix: str = "/api/v1"
    debug: bool = False

    data_dir: Path = Path("data")
    max_upload_mb: int = 200

    redis_url: str = "redis://localhost:6379/0"
    job_store_backend: Literal["redis", "memory"] = "redis"
    celery_eager: bool = False

    default_random_state: int = 42
    figures_dpi: int = 300
    shap_background_samples: int = 100
    shap_max_eval_rows: int = 500
    pdf_engine: Literal["auto", "tectonic", "none"] = "auto"

    cleanup_enabled: bool = True
    cleanup_interval_minutes: int = 360
    job_retention_hours: int = 168
    upload_retention_hours: int = 168

    auth_enabled: bool = False
    api_keys: str = ""
    rate_limit_enabled: bool = False
    rate_limit_max_requests: int = 60
    rate_limit_window_seconds: int = 60

    cors_origins: str = ""
    hsts_enabled: bool = False

    ingestion_service_url: str = "http://ingestion:8000"
    mlcore_service_url: str = "http://mlcore-api:8000"
    inference_service_url: str = "http://inference:8000"

    max_dataset_rows: int = 200_000
    max_dataset_columns: int = 2_000
    predict_max_records: int = 50_000
    predict_max_body_mb: int = 64

    artifact_integrity: Literal["off", "if_present", "strict"] = "if_present"
    artifact_signing_key: str = ""

    # Mango managed dataset (F2). Unprefixed env names: MANGO_SOURCE_DIR, ...
    mango_dataset_id: str = "ds_mango_malambo"
    mango_source_dir: Path = Path("data/mango")
    mango_enable_nasa_api: bool = False
    mango_cache_ttl_hours: int = 24
    mango_date_start: date = date(2018, 3, 1)
    mango_date_end: date = date(2026, 8, 30)
    mango_latitude: float = 10.85
    mango_longitude: float = -74.78
    mango_min_positive_days: int = 30
    mango_strict_coverage: bool = False
    mango_allow_edge_fill: bool = False
    mango_include_uncovered_rows: bool = False
    mango_max_source_mb: int = 100
    mango_source_scan_limit: int = 500
    mango_sync_lock_timeout_seconds: int = 600
    mango_sync_timeout_seconds: int = 180
    mango_nasa_retries: int = 2
    mango_nasa_timeout_seconds: float = 20.0
    mango_keep_versions: int = 10

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    def ensure_dirs(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.mango_source_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
