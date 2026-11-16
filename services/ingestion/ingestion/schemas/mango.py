"""Request/response and manifest contracts for the mango ingestion layer."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

MANIFEST_SCHEMA_VERSION = 1


class TargetKind(StrEnum):
    auto = "auto"
    collected = "collected"
    lost = "lost"


class SourceRoot(StrEnum):
    source_dir = "source_dir"
    managed = "managed"
    nasa_api = "nasa_api"


class RegistrationOrigin(StrEnum):
    upload = "upload"
    source_dir = "source_dir"


class NasaSourceRequest(BaseModel):
    path: str | None = None
    sha256: str | None = None


class LocalSourceRequest(BaseModel):
    path: str | None = None
    target: TargetKind = TargetKind.auto
    sha256: str | None = None


class SourceInspection(BaseModel):
    rows: int
    span: tuple[date, date] | None = None
    parameters: list[str] | None = None
    invalid_rows: int = 0
    duplicate_rows: int = 0
    invalid_dates: int = 0


class SourceRegistration(BaseModel):
    root: SourceRoot
    path: str
    origin: RegistrationOrigin
    sha256: str
    filename: str
    registered_at: datetime
    target: str | None = None


class NasaSourceResponse(BaseModel):
    source: Literal["nasa"] = "nasa"
    registered_at: datetime
    registration: SourceRegistration
    inspection: SourceInspection
    warnings: list[str] = Field(default_factory=list)
    next: str = "POST /api/v1/datasets/mango/sync"


class LocalRegistration(BaseModel):
    target: str
    root: SourceRoot
    path: str
    origin: RegistrationOrigin
    sha256: str
    filename: str
    rows: int
    span: tuple[date, date]
    duplicate_rows: int = 0
    invalid_dates: int = 0


class LocalSourceResponse(BaseModel):
    source: Literal["local"] = "local"
    registered_at: datetime
    registrations: list[LocalRegistration]
    unmatched_targets: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    next: str = "POST /api/v1/datasets/mango/sync"


class MangoSyncRequest(BaseModel):
    requested_start: date | None = None
    requested_end: date | None = None
    refresh_nasa: bool | None = None
    include_uncovered_rows: bool | None = None
    force: bool = False
    dry_run: bool = False

    @model_validator(mode="after")
    def _check_range(self) -> MangoSyncRequest:
        if (
            self.requested_start is not None
            and self.requested_end is not None
            and self.requested_start > self.requested_end
        ):
            raise ValueError("requested_start must be on or before requested_end")
        return self


class MangoSpecPayload(BaseModel):
    dataset_id: str
    canonical_filename: str
    requested_start: date
    requested_end: date
    latitude: float
    longitude: float
    strict_coverage: bool
    include_uncovered_rows: bool
    allow_edge_fill: bool
    min_positive_days: int
    require_targets: list[str]
    collected_target: str
    lost_target: str


class ClimateSourcePayload(BaseModel):
    origin: str
    root: SourceRoot
    path: str
    sha256: str
    fetched_at: datetime | None = None
    params: dict[str, str] | None = None


class LocalSourcePayload(BaseModel):
    root: SourceRoot
    path: str
    sha256: str
    rows: int | None = None
    span: tuple[date, date] | None = None
    duplicate_rows: int = 0
    invalid_dates: int = 0


class MangoSourcesPayload(BaseModel):
    climate: ClimateSourcePayload
    local: dict[str, LocalSourcePayload | None]


class CoverageRange(BaseModel):
    start: date
    end: date
    days: int


class MissingRangePayload(BaseModel):
    start: date
    end: date
    days: int
    reason: str


class CoveragePayload(BaseModel):
    status: str
    requested: CoverageRange
    climate: CoverageRange | None = None
    local: CoverageRange | None = None
    effective: CoverageRange | None = None
    missing_days: int = 0
    missing_ranges: list[MissingRangePayload] = Field(default_factory=list)


class TargetPayload(BaseModel):
    available: bool
    source: str | None = None
    rows: int = 0
    positive_days: int = 0
    sum_kg: float | None = None
    span: tuple[date, date] | None = None
    reason: str | None = None


class CanonicalPayload(BaseModel):
    filename: str
    version: str
    path: str
    sha256: str
    rows: int
    columns: int
    bytes: int
    float_format: str = "%.4f"


class MangoManifest(BaseModel):
    schema_version: int = MANIFEST_SCHEMA_VERSION
    dataset_id: str
    domain: str
    managed: bool = True
    state: Literal["ready", "partial"]
    generated_at: datetime
    run_id: str
    generator: str = "ingestion.mango.sync"
    spec: MangoSpecPayload
    sources: MangoSourcesPayload
    coverage: CoveragePayload
    targets: dict[str, TargetPayload]
    available_targets: list[str] = Field(default_factory=list)
    unavailable_targets: dict[str, str] = Field(default_factory=dict)
    canonical: CanonicalPayload
    imputation: dict[str, Any] = Field(default_factory=dict)
    audit: dict[str, Any] = Field(default_factory=dict)
    input_fingerprint: str
    warnings: list[str] = Field(default_factory=list)


class MangoSyncResponse(BaseModel):
    dataset_id: str
    status: Literal["ready", "partial", "unchanged", "dry_run"]
    changed: bool
    dry_run: bool = False
    run_id: str
    version: str | None = None
    canonical: dict[str, Any]
    previous_sha256: str | None = None
    coverage: dict[str, Any]
    targets: dict[str, Any]
    sources: dict[str, Any]
    caching: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    elapsed_ms: int = 0
