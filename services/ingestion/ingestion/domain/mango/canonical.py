"""Build the canonical daily mango dataset from climate + local records.

The canonical frame is the only artefact downstream services see. It carries a
complete daily grid over the effective window, internally interpolated climate,
calendar/climate features and the available regression targets. Targets without
usable data are omitted (never serialised as an all-NaN column).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date

import pandas as pd
from automl_shared.core.exceptions import ValidationFailedError

from .coverage import CoverageReport, compute_coverage
from .features import (
    CALENDAR_FEATURE_COLUMNS,
    CLIMATE_FEATURE_COLUMNS,
    RAIN_THRESHOLD_MM,
    add_features,
)
from .local_records import LocalRecords
from .nasa_power import ParsedClimate
from .spec import CLIMATE_COLUMNS, DATE_COLUMN, MangoDatasetSpec

CSV_FLOAT_FORMAT = "%.4f"


@dataclass(frozen=True, slots=True)
class TargetStatus:
    """Availability + audit facts for one regression target."""

    name: str
    available: bool
    source: str | None
    rows: int
    positive_days: int
    sum_kg: float | None
    span: tuple[date, date] | None
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "source": self.source,
            "rows": self.rows,
            "positive_days": self.positive_days,
            "sum_kg": round(self.sum_kg, 4) if self.sum_kg is not None else None,
            "span": [self.span[0].isoformat(), self.span[1].isoformat()] if self.span else None,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CanonicalResult:
    frame: pd.DataFrame
    coverage: CoverageReport
    targets: dict[str, TargetStatus]
    warnings: tuple[str, ...]
    audit: dict[str, object]
    imputation: dict[str, object]
    sha256: str

    @property
    def available_targets(self) -> tuple[str, ...]:
        return tuple(name for name, status in self.targets.items() if status.available)

    @property
    def effective_start(self) -> date | None:
        return self.coverage.effective.start if self.coverage.effective else None

    @property
    def effective_end(self) -> date | None:
        return self.coverage.effective.end if self.coverage.effective else None

    def as_metadata(self) -> dict[str, object]:
        """Manifest fragment for the managed dataset (F2 merges the ids)."""
        return {
            "coverage": self.coverage.as_dict(),
            "targets": {name: status.as_dict() for name, status in self.targets.items()},
            "imputation": self.imputation,
            "audit": self.audit,
            "canonical_sha256": self.sha256,
            "warnings": list(self.warnings),
        }


def frame_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialise with the exact settings used to hash the canonical file."""
    out = frame.copy()
    if DATE_COLUMN in out.columns:
        out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN]).dt.strftime("%Y-%m-%d")
    text = out.to_csv(index=False, float_format=CSV_FLOAT_FORMAT, lineterminator="\n")
    return text.encode("utf-8")


def _target_status(
    name: str,
    series: pd.Series,
    *,
    source: str | None,
    span: tuple[date, date] | None,
    min_positive_days: int,
    reason: str | None = None,
) -> TargetStatus:
    values = series.fillna(0.0)
    positive_days = int((series > 0).sum())
    if reason is None and span is None:
        reason = "no local record span"
    if reason is None and positive_days < min_positive_days:
        reason = f"only {positive_days} positive days (< {min_positive_days})"
    available = reason is None
    return TargetStatus(
        name=name,
        available=available,
        source=source,
        rows=int(series.notna().sum()),
        positive_days=positive_days,
        sum_kg=float(values.sum()) if available else None,
        span=span,
        reason=reason,
    )


def build_canonical(
    climate: ParsedClimate,
    records: LocalRecords,
    *,
    spec: MangoDatasetSpec | None = None,
) -> CanonicalResult:
    """Build the canonical frame, its audit block and its content hash."""
    spec = spec or MangoDatasetSpec()
    coverage = compute_coverage(spec, climate=climate.span, local=records.span)
    if coverage.effective is None:
        raise ValidationFailedError(
            "cannot build canonical dataset: " + (" ".join(coverage.warnings) or "no overlapping coverage")
        )

    warnings: list[str] = [*coverage.warnings, *records.warnings]

    grid = pd.DataFrame(
        {DATE_COLUMN: pd.date_range(coverage.effective.start, coverage.effective.end, freq="D")}
    )
    merged = grid.merge(climate.frame, on=DATE_COLUMN, how="left")

    raw_climate = merged.loc[:, list(CLIMATE_COLUMNS)]
    filled = raw_climate.interpolate(method="linear", limit_area="inside")
    interpolated = int((raw_climate.isna() & filled.notna()).to_numpy().sum())

    edge_filled = 0
    if spec.allow_edge_fill:
        with_edges = filled.ffill().bfill()
        edge_filled = int((filled.isna() & with_edges.notna()).to_numpy().sum())
        filled = with_edges

    missing_climate = int(filled.isna().to_numpy().sum())
    merged.loc[:, list(CLIMATE_COLUMNS)] = filled
    if edge_filled:
        warnings.append(f"edge-filled {edge_filled} climate cells outside the observed climate span")
    if missing_climate:
        warnings.append(
            f"{missing_climate} climate cells could not be interpolated internally; rows keep NaN values"
        )

    temperature_inversions = int((merged["temp_max_c"] < merged["temp_min_c"]).sum())
    negative_precip = int((merged["precipitacion_mm"] < 0).sum())
    if temperature_inversions:
        warnings.append(f"{temperature_inversions} days have temp_max_c < temp_min_c")
    if negative_precip:
        warnings.append(f"{negative_precip} days have negative precipitation")

    merged = merged.merge(records.frame, on=DATE_COLUMN, how="left")

    targets: dict[str, TargetStatus] = {}
    zero_filled: dict[str, int] = {}
    for name in spec.target_names:
        local = records.targets.get(name)
        if local is None or name not in merged.columns:
            targets[name] = TargetStatus(
                name=name,
                available=False,
                source=None if local is None else local.path,
                rows=0,
                positive_days=0,
                sum_kg=None,
                span=None if local is None else local.span,
                reason="local source not available",
            )
            continue

        within_span = (merged[DATE_COLUMN] >= pd.Timestamp(local.span[0])) & (
            merged[DATE_COLUMN] <= pd.Timestamp(local.span[1])
        )
        missing_target = merged[name].isna() & within_span
        zero_filled[name] = int(missing_target.sum())
        merged.loc[missing_target, name] = 0.0

        status = _target_status(
            name,
            merged[name],
            source=local.path,
            span=local.span,
            min_positive_days=spec.min_positive_days,
        )
        if not status.available and name in spec.require_targets:
            raise ValidationFailedError(f"required target '{name}' is not usable: {status.reason}")
        if not status.available:
            warnings.append(f"target '{name}' marked unavailable: {status.reason}")
        targets[name] = status

    targets_available = [name for name, status in targets.items() if status.available]
    overlap_days = 0
    if len(targets_available) == len(spec.target_names):
        collected, lost = spec.target_names
        overlap = (merged[collected] > 0) & (merged[lost] > 0)
        overlap_days = int(overlap.sum())
        if overlap_days:
            warnings.append(
                f"{overlap_days} days have both '{collected}' and '{lost}' above zero; "
                "the targets are expected to be mutually exclusive"
            )

    featured = add_features(merged)
    columns = [
        DATE_COLUMN,
        *CLIMATE_COLUMNS,
        *CLIMATE_FEATURE_COLUMNS,
        *CALENDAR_FEATURE_COLUMNS,
        *targets_available,
    ]
    frame = featured.loc[:, columns].sort_values(DATE_COLUMN).reset_index(drop=True)
    payload = frame_to_csv_bytes(frame)

    collected_status = targets.get(spec.collected_target)
    lost_status = targets.get(spec.lost_target)
    collected_sum = collected_status.sum_kg if collected_status and collected_status.available else None
    lost_sum = lost_status.sum_kg if lost_status and lost_status.available else None
    total_fallen = (
        (collected_sum or 0.0) + (lost_sum or 0.0)
        if len(targets_available) == len(spec.target_names)
        else None
    )

    dry_days = int((merged["precipitacion_mm"] < RAIN_THRESHOLD_MM).sum())
    audit: dict[str, object] = {
        "rows": int(len(frame)),
        "requested_days": coverage.requested.days,
        "effective_days": coverage.effective_days,
        "missing_days": coverage.missing_days,
        "climated_days": int(merged["temp_media_c"].notna().sum()),
        "dry_days": dry_days,
        "temperature_inversions": temperature_inversions,
        "negative_precip_days": negative_precip,
        "collected_sum_kg": collected_sum,
        "lost_sum_kg": lost_sum,
        "total_fallen_kg": total_fallen,
        "overlap_days": overlap_days,
    }

    imputation: dict[str, object] = {
        "climate": {
            "method": "linear",
            "limit_area": "inside",
            "cells": interpolated,
            "edge_cells": edge_filled,
            "allow_edge_fill": spec.allow_edge_fill,
            "missing_cells": missing_climate,
        },
        "targets": {
            name: {"method": "zero_within_local_span", "days": days} for name, days in zero_filled.items()
        },
    }

    return CanonicalResult(
        frame=frame,
        coverage=coverage,
        targets=targets,
        warnings=tuple(warnings),
        audit=audit,
        imputation=imputation,
        sha256=hashlib.sha256(payload).hexdigest(),
    )
