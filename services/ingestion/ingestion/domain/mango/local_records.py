"""Reader for the farm's local harvest/loss records.

The local CSV is the runtime source of truth for the regression targets. It is
read as text (headers are matched through aliases), dates are parsed day-first
and repeated dates are consolidated with ``sum(min_count=1)`` so a day with no
record stays distinguishable from a day recorded as zero.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from automl_shared.core.exceptions import ValidationFailedError
from automl_shared.tabular import read_tabular

from .spec import (
    COLLECTED_ALIASES,
    DATE_ALIASES,
    DATE_FORMATS,
    LOST_ALIASES,
    MangoDatasetSpec,
    normalize_header,
)


@dataclass(frozen=True, slots=True)
class LocalTarget:
    """One local target file after consolidation."""

    name: str
    path: str
    rows: int
    span: tuple[date, date]
    duplicate_rows: int = 0
    invalid_dates: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.path,
            "rows": self.rows,
            "span": [self.span[0].isoformat(), self.span[1].isoformat()],
            "duplicate_rows": self.duplicate_rows,
            "invalid_dates": self.invalid_dates,
        }


@dataclass(frozen=True, slots=True)
class LocalRecords:
    """Consolidated local targets plus what the manifest needs to know."""

    frame: pd.DataFrame
    targets: dict[str, LocalTarget | None]
    rows: int
    span: tuple[date, date] | None
    warnings: tuple[str, ...]

    @property
    def available_targets(self) -> dict[str, bool]:
        return {name: target is not None for name, target in self.targets.items()}

    @property
    def source_by_target(self) -> dict[str, str]:
        return {name: target.path for name, target in self.targets.items() if target is not None}

    @property
    def target_spans(self) -> dict[str, tuple[date, date]]:
        return {name: target.span for name, target in self.targets.items() if target is not None}

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(target.path for target in self.targets.values() if target is not None)

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "span": ([self.span[0].isoformat(), self.span[1].isoformat()] if self.span else None),
            "targets": {
                name: target.as_dict() if target is not None else None
                for name, target in self.targets.items()
            },
        }


@dataclass(frozen=True, slots=True)
class LocalSourceInspection:
    """Summary of one local file before it is registered as a source."""

    target: str
    rows: int
    span: tuple[date, date]
    duplicate_rows: int
    invalid_dates: int

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "span": [self.span[0].isoformat(), self.span[1].isoformat()],
            "duplicate_rows": self.duplicate_rows,
            "invalid_dates": self.invalid_dates,
        }


def find_column(frame: pd.DataFrame, aliases: frozenset[str]) -> str | None:
    """Return the original header whose normalised form is in ``aliases``."""
    for column in frame.columns:
        if normalize_header(column) in aliases:
            return str(column)
    return None


def parse_local_dates(values: pd.Series) -> pd.Series:
    """Parse day-first dates, tolerating timestamps and blanks."""
    text = values.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    remaining = text.notna() & (text != "")

    for date_format in DATE_FORMATS:
        if not remaining.any():
            break
        candidate = pd.to_datetime(text, format=date_format, errors="coerce")
        hit = candidate.notna() & remaining
        parsed = parsed.mask(hit, candidate)
        remaining = remaining & ~hit

    if remaining.any():
        candidate = pd.to_datetime(text, format="mixed", dayfirst=True, errors="coerce")
        hit = candidate.notna() & remaining
        parsed = parsed.mask(hit, candidate)

    return parsed


def coerce_numeric(values: pd.Series) -> pd.Series:
    """Convert a spreadsheet column to floats, accepting ``1.234,5``."""

    def to_number(value: object) -> float:
        if value is None or value is pd.NA:
            return np.nan
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        text = str(value).strip().replace("\u00a0", "")
        if not text:
            return np.nan
        if "," in text and "." in text:
            if text.rfind(",") > text.rfind("."):
                text = text.replace(".", "").replace(",", ".")
            else:
                text = text.replace(",", "")
        elif "," in text:
            text = text.replace(",", ".")
        try:
            return float(text)
        except ValueError:
            return np.nan

    if pd.api.types.is_numeric_dtype(values):
        return values.astype(float)
    return values.map(to_number).astype(float)


def aliases_for(target: str, spec: MangoDatasetSpec) -> frozenset[str]:
    if target == spec.collected_target:
        return COLLECTED_ALIASES
    if target == spec.lost_target:
        return LOST_ALIASES
    raise ValueError(f"unknown target: {target}")


def _aliases_for(target: str, spec: MangoDatasetSpec) -> frozenset[str]:
    return aliases_for(target, spec)


def _read_target(
    target: str,
    path: Path,
    *,
    spec: MangoDatasetSpec,
    required: bool,
    warnings: list[str],
) -> tuple[pd.Series | None, int, int]:
    """Return ``(series indexed by fecha, duplicate rows, invalid dates)``."""
    frame = read_tabular(path)
    date_column = find_column(frame, DATE_ALIASES)
    if date_column is None:
        message = f"{path.name}: no date column matching {sorted(DATE_ALIASES)}"
        if required:
            raise ValidationFailedError(message)
        warnings.append(message)
        return None, 0, 0

    target_column = find_column(frame, _aliases_for(target, spec))
    if target_column is None:
        message = f"{path.name}: no column matching target '{target}'"
        if required:
            raise ValidationFailedError(message)
        warnings.append(message)
        return None, 0, 0

    dates = parse_local_dates(frame[date_column])
    invalid_dates = int(dates.isna().sum())
    values = coerce_numeric(frame[target_column])
    unparsed = int(values.isna().sum() - frame[target_column].isna().sum())
    if unparsed > 0:
        warnings.append(f"{path.name}: {unparsed} non-numeric '{target}' cells treated as missing")

    negative = values.lt(0).fillna(False)
    if negative.any():
        raise ValidationFailedError(f"{path.name}: '{target}' has {int(negative.sum())} negative values")

    rows = pd.DataFrame({"fecha": dates, target: values}).loc[dates.notna()]
    if rows.empty:
        warnings.append(f"{path.name}: no valid rows for '{target}'")
        return None, 0, invalid_dates

    consolidated = rows.groupby("fecha", as_index=True, sort=True)[target].sum(min_count=1)
    duplicates = int(len(rows) - len(consolidated))
    return consolidated.astype(float), duplicates, invalid_dates


def inspect_local_source(
    path: str | Path,
    *,
    target: str,
    spec: MangoDatasetSpec | None = None,
) -> tuple[LocalSourceInspection, tuple[str, ...]]:
    """Validate one local file for a single target without consolidating it.

    Used by ``PUT /datasets/mango/sources/local`` to reject unusable files
    before they are registered. Raises ``ValidationFailedError`` when the date
    or target column is missing, dates are unusable, or values are negative.
    """
    spec = spec or MangoDatasetSpec()
    warnings: list[str] = []
    source = Path(path)
    series, duplicates, invalid_dates = _read_target(
        target, source, spec=spec, required=True, warnings=warnings
    )
    if series is None or series.dropna().empty:
        raise ValidationFailedError(f"{source.name}: no usable values for '{target}'")
    dates = series.index
    inspection = LocalSourceInspection(
        target=target,
        rows=int(len(series)),
        span=(dates.min().date(), dates.max().date()),
        duplicate_rows=duplicates,
        invalid_dates=invalid_dates,
    )
    return inspection, tuple(warnings)


def build_local_records(
    sources: Mapping[str, str | Path | None],
    *,
    spec: MangoDatasetSpec | None = None,
) -> LocalRecords:
    """Read every configured local file and consolidate it into one frame.

    ``sources`` maps target name -> CSV path. A missing optional path marks the
    target unavailable instead of failing, so the loss file can land later
    without changing any code.
    """
    spec = spec or MangoDatasetSpec()
    warnings: list[str] = []
    series: list[pd.Series] = []
    targets: dict[str, LocalTarget | None] = {}

    for target in spec.target_names:
        required = target in spec.require_targets
        raw_path = sources.get(target)
        path = Path(raw_path) if raw_path is not None else None
        targets[target] = None

        if path is None or not path.is_file():
            message = f"local source for '{target}' is not available"
            if required:
                raise ValidationFailedError(message)
            warnings.append(message)
            continue

        target_series, duplicates, invalid_dates = _read_target(
            target, path, spec=spec, required=required, warnings=warnings
        )
        if target_series is None or target_series.dropna().empty:
            message = f"local source for '{target}' has no usable values"
            if required:
                raise ValidationFailedError(message)
            warnings.append(message)
            continue

        dates = target_series.index
        targets[target] = LocalTarget(
            name=target,
            path=path.name,
            rows=int(len(target_series)),
            span=(dates.min().date(), dates.max().date()),
            duplicate_rows=duplicates,
            invalid_dates=invalid_dates,
        )
        series.append(target_series.rename(target))

    if not series:
        return LocalRecords(
            frame=pd.DataFrame({"fecha": pd.Series(dtype="datetime64[ns]")}),
            targets=targets,
            rows=0,
            span=None,
            warnings=tuple(warnings),
        )

    frame = pd.concat(series, axis=1, join="outer").sort_index()
    frame.index.name = "fecha"
    frame = frame.reset_index()

    dates = frame["fecha"]
    return LocalRecords(
        frame=frame.loc[:, ["fecha", *[name for name in spec.target_names if name in frame]]],
        targets=targets,
        rows=int(len(frame)),
        span=(dates.min().date(), dates.max().date()),
        warnings=tuple(warnings),
    )
