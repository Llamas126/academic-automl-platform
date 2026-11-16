"""Parser for the NASA POWER daily CSV export.

NASA POWER prepends a free-text preamble to the CSV and identifies days with
``YEAR`` + ``DOY`` instead of a date column. Some exports (and hand-edited
copies) also carry an extra ``PARAMETER`` column, so both shapes are accepted.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from automl_shared.core.exceptions import ValidationFailedError

from .spec import (
    CLIMATE_COLUMNS,
    NASA_MISSING_VALUE,
    NASA_PARAM_MAP,
    MangoDatasetSpec,
    normalize_header,
)


@dataclass(frozen=True, slots=True)
class ParsedClimate:
    """Daily climate frame plus the facts the coverage/report layer needs."""

    frame: pd.DataFrame
    start: date | None
    end: date | None
    days: int
    parameters: tuple[str, ...]
    invalid_rows: int

    @property
    def span(self) -> tuple[date, date] | None:
        if self.start is None or self.end is None:
            return None
        return (self.start, self.end)


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValidationFailedError("could not decode NASA POWER CSV")


def _read_text(source: str | bytes | Path) -> str:
    if isinstance(source, Path):
        try:
            return _decode(source.read_bytes())
        except OSError as exc:
            raise ValidationFailedError(f"could not read NASA POWER CSV: {exc}") from exc
    if isinstance(source, bytes):
        return _decode(source)
    return source.removeprefix("\ufeff")


def _find_header(lines: list[str]) -> int:
    for index, line in enumerate(lines):
        fields = {normalize_header(part) for part in line.split(",")}
        if {"year", "doy"} <= fields:
            return index
    raise ValidationFailedError("NASA POWER CSV has no YEAR,DOY header")


def _read_frame(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    header = _find_header(lines)
    body = "\n".join(lines[header:]).strip()
    if not body:
        raise ValidationFailedError("NASA POWER CSV is empty")
    frame = pd.read_csv(io.StringIO(body))
    frame.columns = [normalize_header(column) for column in frame.columns]
    if frame.empty:
        raise ValidationFailedError("NASA POWER CSV has no data rows")
    return frame.loc[:, ~frame.columns.duplicated()]


def _dates(year: pd.Series, doy: pd.Series) -> pd.Series:
    numeric_year = pd.to_numeric(year, errors="coerce")
    numeric_doy = pd.to_numeric(doy, errors="coerce")
    valid = (
        numeric_year.notna()
        & numeric_doy.notna()
        & numeric_year.between(1900, 2200)
        & numeric_doy.between(1, 366)
    )
    origin = pd.to_datetime(numeric_year.fillna(2000).astype(int).astype(str) + "-01-01", format="%Y-%m-%d")
    dates = origin + pd.to_timedelta(numeric_doy.fillna(1).astype(int) - 1, unit="D")
    return dates.where(valid)


def parse_nasa_power_csv(
    source: str | bytes | Path,
    *,
    spec: MangoDatasetSpec | None = None,
) -> ParsedClimate:
    """Parse a NASA POWER daily CSV into a canonical climate frame.

    The returned frame is sorted by ``fecha``, has one row per day and always
    carries every canonical climate column (missing parameters become NaN).
    """
    spec = spec or MangoDatasetSpec()
    frame = _read_frame(_read_text(source))

    if "parameter" in frame.columns:
        # Some exports repeat the parameter name on every row; it is not data.
        frame = frame.drop(columns=["parameter"])

    missing_grid = [column for column in ("year", "doy") if column not in frame.columns]
    if missing_grid:
        raise ValidationFailedError(f"NASA POWER CSV is missing columns: {missing_grid}")

    parameters: list[str] = []
    for column in frame.columns:
        canonical = NASA_PARAM_MAP.get(column)
        if canonical is not None:
            parameters.append(column)
            if canonical in frame.columns and canonical != column:
                raise ValidationFailedError(
                    f"NASA POWER CSV has two columns for '{canonical}': '{column}' and '{canonical}'"
                )

    if not parameters:
        raise ValidationFailedError(
            f"NASA POWER CSV has no recognised parameter columns; expected any of {sorted(NASA_PARAM_MAP)}"
        )

    frame = frame.rename(columns={name: NASA_PARAM_MAP[name] for name in parameters})
    frame = frame.loc[:, ~frame.columns.duplicated()]

    frame["fecha"] = _dates(frame["year"], frame["doy"])
    invalid_rows = int(frame["fecha"].isna().sum())
    frame = frame.loc[frame["fecha"].notna()]

    for column in CLIMATE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
        values = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = values.mask(values <= NASA_MISSING_VALUE)

    frame = (
        frame.loc[:, ["fecha", *CLIMATE_COLUMNS]]
        .sort_values("fecha")
        .drop_duplicates(subset="fecha", keep="last")
        .reset_index(drop=True)
    )

    if frame.empty:
        raise ValidationFailedError("NASA POWER CSV has no valid dated rows")

    start = frame["fecha"].min().date()
    end = frame["fecha"].max().date()

    return ParsedClimate(
        frame=frame,
        start=start,
        end=end,
        days=int(len(frame)),
        parameters=tuple(parameters),
        invalid_rows=invalid_rows,
    )
