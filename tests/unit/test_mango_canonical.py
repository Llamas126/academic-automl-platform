from __future__ import annotations

import hashlib
from datetime import date, timedelta

import pandas as pd
import pytest
from automl_shared.core.exceptions import ValidationFailedError
from ingestion.domain.mango.canonical import build_canonical, frame_to_csv_bytes
from ingestion.domain.mango.features import (
    CALENDAR_FEATURE_COLUMNS,
    CLIMATE_FEATURE_COLUMNS,
)
from ingestion.domain.mango.local_records import build_local_records
from ingestion.domain.mango.nasa_power import parse_nasa_power_csv
from ingestion.domain.mango.spec import (
    CLIMATE_COLUMNS,
    COLLECTED_TARGET,
    LOST_TARGET,
    MangoDatasetSpec,
)

from tests.fixtures.make_fixtures import (
    make_mango_local_records,
    make_mango_loss_records,
    make_nasa_power_csv,
)

START = date(2024, 1, 11)
DAYS = 60


def make_spec(days: int = DAYS, **overrides) -> MangoDatasetSpec:
    values = {
        "date_start": START,
        "date_end": START + timedelta(days=days - 1),
        "min_positive_days": 1,
    }
    values.update(overrides)
    return MangoDatasetSpec(**values)


def write_records(tmp_path, frame: pd.DataFrame, name: str):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def make_inputs(tmp_path, *, days: int = DAYS, include_loss: bool = True):
    spec = make_spec(days)
    climate = parse_nasa_power_csv(make_nasa_power_csv(start=START.isoformat(), n_days=days))
    sources = {
        COLLECTED_TARGET: write_records(
            tmp_path,
            make_mango_local_records(start=START.isoformat(), n_days=days),
            "recoleccion.csv",
        )
    }
    if include_loss:
        sources[LOST_TARGET] = write_records(
            tmp_path,
            make_mango_loss_records(start=START.isoformat(), n_days=days),
            "perdidas.csv",
        )
    records = build_local_records(sources, spec=spec)
    return spec, climate, records


def test_canonical_schema_and_hash(tmp_path) -> None:
    spec, climate, records = make_inputs(tmp_path)

    result = build_canonical(climate, records, spec=spec)

    expected_columns = [
        "fecha",
        *CLIMATE_COLUMNS,
        *CLIMATE_FEATURE_COLUMNS,
        *CALENDAR_FEATURE_COLUMNS,
        COLLECTED_TARGET,
        LOST_TARGET,
    ]
    assert list(result.frame.columns) == expected_columns
    assert len(result.frame) == DAYS
    assert result.frame["fecha"].iloc[0] == pd.Timestamp(START)
    assert result.frame["fecha"].is_monotonic_increasing
    assert result.frame["fecha"].dt.date.iloc[-1] == START + timedelta(days=DAYS - 1)
    assert result.coverage.status == "complete"
    assert result.effective_start == START
    assert result.available_targets == (COLLECTED_TARGET, LOST_TARGET)
    assert result.sha256 == hashlib.sha256(frame_to_csv_bytes(result.frame)).hexdigest()
    assert result.frame["anio"].iloc[0] == 2024
    assert result.frame["sin_doy"].between(-1.0, 1.0).all()
    assert result.frame["cos_doy"].between(-1.0, 1.0).all()


def test_internal_climate_nulls_are_linearly_interpolated(tmp_path) -> None:
    spec, climate, records = make_inputs(tmp_path)
    raw = climate.frame
    expected_precip = (raw.loc[4, "precipitacion_mm"] + raw.loc[6, "precipitacion_mm"]) / 2
    expected_humidity = (raw.loc[16, "humedad_relativa_pct"] + raw.loc[18, "humedad_relativa_pct"]) / 2

    result = build_canonical(climate, records, spec=spec)

    assert result.frame.loc[5, "precipitacion_mm"] == pytest.approx(expected_precip)
    assert result.frame.loc[17, "humedad_relativa_pct"] == pytest.approx(expected_humidity)
    assert result.frame[list(CLIMATE_COLUMNS)].notna().all().all()
    assert result.imputation["climate"]["cells"] >= 2
    assert result.imputation["climate"]["edge_cells"] == 0


def test_missing_loss_target_is_omitted_not_zeroed(tmp_path) -> None:
    spec, climate, records = make_inputs(tmp_path, include_loss=False)

    result = build_canonical(climate, records, spec=spec)

    assert LOST_TARGET not in result.frame.columns
    lost = result.targets[LOST_TARGET]
    assert lost.available is False
    assert lost.sum_kg is None
    assert lost.reason == "local source not available"
    assert result.available_targets == (COLLECTED_TARGET,)
    assert any("local source" in warning for warning in result.warnings)
    assert any("local source" in warning for warning in records.warnings)
    assert result.audit["lost_sum_kg"] is None
    assert result.audit["total_fallen_kg"] is None


def test_zero_imputation_respects_each_target_span(tmp_path) -> None:
    spec = make_spec(30)
    climate = parse_nasa_power_csv(make_nasa_power_csv(start=START.isoformat(), n_days=30))
    collected = write_records(
        tmp_path,
        pd.DataFrame(
            {
                "FECHA": ["11-01-2024", "09-02-2024"],
                "PESO (PRODUCCIÓN)": [4.0, 6.0],
            }
        ),
        "recoleccion.csv",
    )
    lost = write_records(
        tmp_path,
        pd.DataFrame(
            {
                "FECHA": ["15-01-2024", "20-01-2024"],
                "Peso_Perdido_KG": [2.0, 1.0],
            }
        ),
        "perdidas.csv",
    )
    records = build_local_records({COLLECTED_TARGET: collected, LOST_TARGET: lost}, spec=spec)

    result = build_canonical(climate, records, spec=spec)

    collected_values = result.frame[COLLECTED_TARGET]
    lost_values = result.frame[LOST_TARGET]
    assert collected_values.iloc[0] == pytest.approx(4.0)
    assert (collected_values.iloc[1:29] == 0.0).all()
    assert collected_values.iloc[29] == pytest.approx(6.0)

    # Lost records span 2024-01-15 .. 2024-01-20 only.
    assert lost_values.iloc[:4].isna().all()
    assert lost_values.iloc[4] == pytest.approx(2.0)
    assert (lost_values.iloc[5:9] == 0.0).all()
    assert lost_values.iloc[9] == pytest.approx(1.0)
    assert lost_values.iloc[10:].isna().all()
    assert result.imputation["targets"][LOST_TARGET] == {
        "method": "zero_within_local_span",
        "days": 4,
    }


def test_required_target_without_positive_days_raises(tmp_path) -> None:
    spec = make_spec(30)
    climate = parse_nasa_power_csv(make_nasa_power_csv(start=START.isoformat(), n_days=30))
    collected = write_records(
        tmp_path,
        pd.DataFrame(
            {
                "FECHA": ["11-01-2024", "09-02-2024"],
                "PESO (PRODUCCIÓN)": [0.0, 0.0],
            }
        ),
        "recoleccion.csv",
    )
    records = build_local_records({COLLECTED_TARGET: collected}, spec=spec)

    with pytest.raises(ValidationFailedError, match="required target"):
        build_canonical(climate, records, spec=spec)


def test_partial_coverage_warns_but_does_not_crash(tmp_path) -> None:
    spec = MangoDatasetSpec(
        date_start=date(2024, 1, 1),
        date_end=date(2024, 3, 31),
        min_positive_days=1,
    )
    climate = parse_nasa_power_csv(make_nasa_power_csv(start=START.isoformat(), n_days=30))
    collected = write_records(
        tmp_path,
        make_mango_local_records(start=START.isoformat(), n_days=30),
        "recoleccion.csv",
    )
    records = build_local_records({COLLECTED_TARGET: collected}, spec=spec)

    result = build_canonical(climate, records, spec=spec)

    assert result.coverage.status == "partial"
    assert len(result.frame) == 30
    assert result.coverage.missing_days > 0
    assert any("partial coverage" in warning for warning in result.warnings)
    assert result.audit["requested_days"] == 91
    assert result.audit["effective_days"] == 30


def test_include_uncovered_rows_keeps_requested_grid_with_nans(tmp_path) -> None:
    spec = MangoDatasetSpec(
        date_start=date(2024, 1, 1),
        date_end=date(2024, 3, 31),
        min_positive_days=1,
        include_uncovered_rows=True,
    )
    climate = parse_nasa_power_csv(make_nasa_power_csv(start=START.isoformat(), n_days=30))
    collected = write_records(
        tmp_path,
        make_mango_local_records(start=START.isoformat(), n_days=30),
        "recoleccion.csv",
    )
    records = build_local_records({COLLECTED_TARGET: collected}, spec=spec)

    result = build_canonical(climate, records, spec=spec)

    assert len(result.frame) == 91
    assert result.coverage.effective == result.coverage.requested
    assert pd.isna(result.frame.loc[0, "temp_media_c"])
    assert pd.isna(result.frame.loc[90, "temp_media_c"])
    assert pd.isna(result.frame[COLLECTED_TARGET].iloc[0])
    assert result.frame[COLLECTED_TARGET].iloc[10:40].notna().any()


def test_overlap_between_targets_is_audited_and_warned(tmp_path) -> None:
    spec = make_spec(20)
    climate = parse_nasa_power_csv(make_nasa_power_csv(start=START.isoformat(), n_days=20))
    collected = write_records(
        tmp_path,
        pd.DataFrame(
            {
                "FECHA": ["11-01-2024", "20-01-2024"],
                "PESO (PRODUCCIÓN)": [5.0, 5.0],
            }
        ),
        "recoleccion.csv",
    )
    lost = write_records(
        tmp_path,
        pd.DataFrame(
            {
                "FECHA": ["20-01-2024", "30-01-2024"],
                "Peso_Perdido_KG": [2.0, 1.0],
            }
        ),
        "perdidas.csv",
    )
    records = build_local_records({COLLECTED_TARGET: collected, LOST_TARGET: lost}, spec=spec)

    result = build_canonical(climate, records, spec=spec)

    assert result.audit["overlap_days"] == 1
    assert any("mutually exclusive" in warning for warning in result.warnings)
    assert result.audit["collected_sum_kg"] == pytest.approx(10.0)
    assert result.audit["lost_sum_kg"] == pytest.approx(3.0)
    assert result.audit["total_fallen_kg"] == pytest.approx(13.0)


def test_audit_totals_match_local_records(tmp_path) -> None:
    spec, climate, records = make_inputs(tmp_path)

    result = build_canonical(climate, records, spec=spec)

    assert result.audit["collected_sum_kg"] == pytest.approx(records.frame[COLLECTED_TARGET].sum())
    assert result.audit["lost_sum_kg"] == pytest.approx(records.frame[LOST_TARGET].sum())
    assert result.audit["total_fallen_kg"] == pytest.approx(
        records.frame[COLLECTED_TARGET].sum() + records.frame[LOST_TARGET].sum()
    )
    assert result.audit["rows"] == len(result.frame)
    assert result.audit["effective_days"] == DAYS


def test_no_target_derived_features(tmp_path) -> None:
    spec, climate, records = make_inputs(tmp_path)

    result = build_canonical(climate, records, spec=spec)

    forbidden = {"Peso_Total_Caido_KG", "peso_total_caido_kg"}
    assert forbidden.isdisjoint(result.frame.columns)
    assert not any(column.startswith(("lag_", "target_")) for column in result.frame.columns)
    assert not any(name in result.frame.columns for name in (f"{LOST_TARGET}_lag1",))


def test_csv_serialisation_is_stable_and_uses_four_decimals(tmp_path) -> None:
    spec, climate, records = make_inputs(tmp_path)
    result = build_canonical(climate, records, spec=spec)

    first = frame_to_csv_bytes(result.frame)
    second = frame_to_csv_bytes(result.frame.copy())

    assert first == second
    assert first.startswith(b"fecha,temp_media_c")
    assert b"2024-01-11" in first
    assert first.count(b"\n") == len(result.frame) + 1
    assert b"0.0000" in first
    assert hashlib.sha256(first).hexdigest() == result.sha256


def test_metadata_fragment_exposes_availability(tmp_path) -> None:
    spec, climate, records = make_inputs(tmp_path, include_loss=False)

    result = build_canonical(climate, records, spec=spec)
    metadata = result.as_metadata()

    assert set(metadata) == {
        "coverage",
        "targets",
        "imputation",
        "audit",
        "canonical_sha256",
        "warnings",
    }
    assert metadata["targets"][COLLECTED_TARGET]["available"] is True
    assert metadata["targets"][LOST_TARGET]["available"] is False
    assert metadata["canonical_sha256"] == result.sha256
    assert isinstance(metadata["warnings"], list)


def test_empty_overlap_raises(tmp_path) -> None:
    spec = MangoDatasetSpec(
        date_start=date(2030, 1, 1),
        date_end=date(2030, 3, 1),
        min_positive_days=1,
    )
    climate = parse_nasa_power_csv(make_nasa_power_csv(start=START.isoformat(), n_days=30))
    collected = write_records(
        tmp_path,
        make_mango_local_records(start=START.isoformat(), n_days=30),
        "recoleccion.csv",
    )
    records = build_local_records({COLLECTED_TARGET: collected}, spec=spec)

    with pytest.raises(ValidationFailedError, match="cannot build canonical dataset"):
        build_canonical(climate, records, spec=spec)
