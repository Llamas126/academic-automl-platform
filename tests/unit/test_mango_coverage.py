from __future__ import annotations

from datetime import date, timedelta

import pytest
from ingestion.domain.mango.coverage import (
    REASON_NO_CLIMATE,
    REASON_NO_LOCAL_RECORDS,
    CoverageReport,
    DateRange,
    compute_coverage,
)
from ingestion.domain.mango.spec import MangoDatasetSpec

START = date(2018, 3, 1)
END = date(2026, 8, 30)


def spec(**overrides) -> MangoDatasetSpec:
    values = {"date_start": START, "date_end": END}
    values.update(overrides)
    return MangoDatasetSpec(**values)


def test_complete_coverage() -> None:
    report = compute_coverage(spec(), climate=(START, END), local=(START, END))

    assert report.status == "complete"
    assert report.is_complete
    assert report.effective == DateRange(START, END)
    assert report.missing_ranges == ()
    assert report.warnings == ()
    assert report.missing_days == 0


def test_partial_coverage_reports_before_and_after() -> None:
    climate = (date(2024, 1, 11), date(2025, 8, 23))
    report = compute_coverage(spec(), climate=climate, local=(START, END))

    assert report.status == "partial"
    assert report.effective == DateRange(*climate)
    assert len(report.missing_ranges) == 2
    before, after = report.missing_ranges
    assert (before.start, before.end, before.reason) == (
        START,
        date(2024, 1, 10),
        REASON_NO_CLIMATE,
    )
    assert (after.start, after.end, after.reason) == (
        date(2025, 8, 24),
        END,
        REASON_NO_CLIMATE,
    )
    assert before.days == (date(2024, 1, 11) - START).days
    assert after.days == (END - date(2025, 8, 23)).days
    assert report.missing_days == report.requested.days - report.effective_days
    assert any("partial coverage" in warning for warning in report.warnings)
    assert any("2024-01-11..2025-08-23" in warning for warning in report.warnings)


def test_local_span_is_reported_with_its_own_reason() -> None:
    local = (date(2024, 1, 11), date(2025, 8, 23))
    report = compute_coverage(spec(), climate=(START, END), local=local)

    assert report.status == "partial"
    assert report.effective == DateRange(*local)
    assert {item.reason for item in report.missing_ranges} == {REASON_NO_LOCAL_RECORDS}


def test_missing_climate_wins_over_missing_local_records() -> None:
    report = compute_coverage(
        spec(),
        climate=(date(2024, 1, 11), date(2025, 8, 23)),
        local=(date(2024, 6, 1), date(2025, 6, 30)),
    )

    assert report.effective == DateRange(date(2024, 6, 1), date(2025, 6, 30))
    reasons = [(item.reason, item.start, item.end) for item in report.missing_ranges]
    assert reasons[0] == (REASON_NO_CLIMATE, START, date(2024, 1, 10))
    assert reasons[-1] == (REASON_NO_CLIMATE, date(2025, 8, 24), END)


def test_no_overlap_is_empty() -> None:
    report = compute_coverage(
        spec(),
        climate=(date(2024, 1, 1), date(2024, 6, 1)),
        local=(date(2018, 3, 1), date(2019, 3, 1)),
    )

    assert report.status == "empty"
    assert report.effective is None
    assert report.effective_days == 0
    assert any("no overlap" in warning for warning in report.warnings)


def test_missing_source_marks_the_range_uncoverable() -> None:
    report = compute_coverage(spec(), climate=None, local=(START, END))

    assert report.status == "empty"
    assert report.effective is None
    assert single_range(report).reason == REASON_NO_CLIMATE


def test_include_uncovered_rows_keeps_the_requested_grid() -> None:
    report = compute_coverage(
        spec(include_uncovered_rows=True),
        climate=(date(2024, 1, 11), date(2025, 8, 23)),
        local=(START, END),
    )

    assert report.effective == DateRange(START, END)
    assert report.status == "partial"
    assert report.missing_days > 0


def test_internal_gap_does_not_split_the_effective_range() -> None:
    report = compute_coverage(spec(), climate=(START, END), local=(START, END))

    assert report.missing_ranges == ()
    assert report.effective == DateRange(START, END)


def test_coverage_as_dict_is_json_ready() -> None:
    report = compute_coverage(
        spec(date_start=date(2024, 1, 1), date_end=date(2024, 1, 31)),
        climate=(date(2024, 1, 11), date(2024, 1, 20)),
        local=(date(2024, 1, 1), date(2024, 1, 31)),
    )

    payload = report.as_dict()

    assert payload["status"] == "partial"
    assert payload["effective"] == {"start": "2024-01-11", "end": "2024-01-20", "days": 10}
    assert payload["missing_ranges"][0]["reason"] == REASON_NO_CLIMATE
    assert isinstance(payload["missing_ranges"][0]["start"], str)


def test_date_range_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="invalid date range"):
        DateRange(END, START)


def test_date_range_intersection() -> None:
    left = DateRange(date(2024, 1, 1), date(2024, 1, 31))
    right = DateRange(date(2024, 1, 15), date(2024, 2, 10))

    assert left.intersection(right) == DateRange(date(2024, 1, 15), date(2024, 1, 31))
    assert left.intersection(DateRange(date(2024, 3, 1), date(2024, 3, 2))) is None


def test_date_range_days_are_inclusive() -> None:
    assert DateRange(date(2024, 1, 1), date(2024, 1, 1)).days == 1
    assert DateRange(date(2024, 1, 1), date(2024, 1, 1) + timedelta(days=9)).days == 10


def single_range(report: CoverageReport):
    assert len(report.missing_ranges) == 1
    return report.missing_ranges[0]
