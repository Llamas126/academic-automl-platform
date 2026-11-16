"""Coverage maths for the requested mango date range.

Coverage never raises: it reports which slices of the requested range cannot be
built from the available sources (NASA climate span and local record span) so
callers can warn and keep going.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from .spec import MangoDatasetSpec

REASON_NO_CLIMATE = "no_climate"
REASON_NO_LOCAL_RECORDS = "no_local_records"


@dataclass(frozen=True, slots=True)
class DateRange:
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"invalid date range: {self.start} > {self.end}")

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end

    def intersection(self, other: DateRange) -> DateRange | None:
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        if start > end:
            return None
        return DateRange(start, end)

    def as_dict(self) -> dict[str, object]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat(), "days": self.days}


@dataclass(frozen=True, slots=True)
class MissingRange:
    start: date
    end: date
    reason: str

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    @property
    def span(self) -> DateRange:
        return DateRange(self.start, self.end)

    def as_dict(self) -> dict[str, object]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "days": self.days,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CoverageReport:
    requested: DateRange
    climate: DateRange | None
    local: DateRange | None
    effective: DateRange | None
    missing_ranges: tuple[MissingRange, ...]
    status: str
    warnings: tuple[str, ...]

    @property
    def missing_days(self) -> int:
        return sum(item.days for item in self.missing_ranges)

    @property
    def effective_days(self) -> int:
        return self.effective.days if self.effective else 0

    @property
    def is_complete(self) -> bool:
        return self.status == "complete"

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "requested": self.requested.as_dict(),
            "climate": self.climate.as_dict() if self.climate else None,
            "local": self.local.as_dict() if self.local else None,
            "effective": self.effective.as_dict() if self.effective else None,
            "missing_days": self.missing_days,
            "missing_ranges": [item.as_dict() for item in self.missing_ranges],
        }


def _as_range(value: DateRange | tuple[date, date] | None) -> DateRange | None:
    if value is None or isinstance(value, DateRange):
        return value
    start, end = value
    return DateRange(start, end)


def _missing_reason(day: date, climate: DateRange | None, local: DateRange | None) -> str | None:
    if climate is None:
        return REASON_NO_CLIMATE
    if not climate.contains(day):
        return REASON_NO_CLIMATE
    if local is None:
        return REASON_NO_LOCAL_RECORDS
    if not local.contains(day):
        return REASON_NO_LOCAL_RECORDS
    return None


def compute_coverage(
    spec: MangoDatasetSpec,
    *,
    climate: DateRange | tuple[date, date] | None,
    local: DateRange | tuple[date, date] | None,
) -> CoverageReport:
    """Report which days of the requested range can actually be built."""
    requested = DateRange(spec.date_start, spec.date_end)
    climate_range = _as_range(climate)
    local_range = _as_range(local)

    missing: list[MissingRange] = []
    current: MissingRange | None = None
    day = requested.start

    while day <= requested.end:
        reason = _missing_reason(day, climate_range, local_range)
        if reason is None:
            current = None
        elif current is not None and current.reason == reason:
            current = MissingRange(current.start, day, reason)
            missing[-1] = current
        else:
            current = MissingRange(day, day, reason)
            missing.append(current)
        day += timedelta(days=1)

    effective: DateRange | None = requested
    if not spec.include_uncovered_rows:
        for span in (climate_range, local_range):
            if span is None:
                effective = None
                break
            effective = effective.intersection(span) if effective else None
            if effective is None:
                break

    if effective is not None and effective.days == 0:
        effective = None

    if effective is None:
        status = "empty"
    elif not missing and effective == requested:
        status = "complete"
    else:
        status = "partial"

    warnings = _coverage_warnings(requested, effective, missing, status)
    return CoverageReport(
        requested=requested,
        climate=climate_range,
        local=local_range,
        effective=effective,
        missing_ranges=tuple(missing),
        status=status,
        warnings=warnings,
    )


def _coverage_warnings(
    requested: DateRange,
    effective: DateRange | None,
    missing: list[MissingRange],
    status: str,
) -> tuple[str, ...]:
    if status == "empty":
        return (
            f"no overlap between the requested range {requested.start}..{requested.end} "
            "and the available climate/local sources",
        )

    warnings: list[str] = []
    if status == "partial":
        described = "; ".join(f"{item.start}..{item.end} ({item.reason}, {item.days}d)" for item in missing)
        warnings.append(
            f"partial coverage: {sum(item.days for item in missing)} of "
            f"{requested.days} requested days are missing ({described})"
        )
        if effective is not None:
            warnings.append(
                f"training window limited to {effective.start}..{effective.end} ({effective.days} days)"
            )
    return tuple(warnings)
