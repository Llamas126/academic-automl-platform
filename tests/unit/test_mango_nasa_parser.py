from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from automl_shared.core.exceptions import ValidationFailedError
from ingestion.domain.mango.nasa_power import parse_nasa_power_csv
from ingestion.domain.mango.spec import CLIMATE_COLUMNS

from tests.fixtures.make_fixtures import make_nasa_power_csv


def test_parses_preamble_and_maps_parameters() -> None:
    text = make_nasa_power_csv(n_days=30)
    parsed = parse_nasa_power_csv(text)

    assert parsed.days == 30
    assert parsed.start == date(2024, 1, 11)
    assert parsed.end == date(2024, 2, 9)
    assert parsed.invalid_rows == 0
    assert list(parsed.frame.columns) == ["fecha", *CLIMATE_COLUMNS]
    assert parsed.frame["fecha"].is_monotonic_increasing
    assert "t2m" not in parsed.frame.columns
    assert parsed.frame["temp_media_c"].notna().all()


def test_missing_sentinel_becomes_nan_and_is_interpolatable() -> None:
    parsed = parse_nasa_power_csv(make_nasa_power_csv(n_days=30))

    assert pd.isna(parsed.frame.loc[5, "precipitacion_mm"])
    assert pd.isna(parsed.frame.loc[17, "humedad_relativa_pct"])
    assert parsed.frame.loc[4, "precipitacion_mm"] >= 0


def test_parameter_column_is_dropped() -> None:
    text = "\n".join(
        [
            "-BEGIN HEADER-",
            "pretend preamble",
            "-END HEADER-",
            "PARAMETER,YEAR,DOY,T2M,PRECTOTCORR",
            "T2M,2024,1,25.5,0.0",
            "T2M,2024,2,26.5,1.5",
        ]
    )
    parsed = parse_nasa_power_csv(text)

    assert parsed.days == 2
    assert parsed.frame.loc[0, "temp_media_c"] == pytest.approx(25.5)
    assert parsed.frame.loc[1, "precipitacion_mm"] == pytest.approx(1.5)


def test_missing_parameter_columns_are_kept_as_nan() -> None:
    text = "\n".join(
        [
            "-END HEADER-",
            "YEAR,DOY,T2M",
            "2024,1,25.5",
            "2024,2,26.5",
        ]
    )
    parsed = parse_nasa_power_csv(text)

    assert parsed.parameters == ("t2m",)
    assert parsed.frame["temp_media_c"].notna().all()
    assert parsed.frame["precipitacion_mm"].isna().all()


def test_legacy_prectot_parameter_is_mapped() -> None:
    text = "\n".join(["-END HEADER-", "YEAR,DOY,T2M,PRECTOT", "2024,1,25.0,2.0"])
    parsed = parse_nasa_power_csv(text)

    assert parsed.frame.loc[0, "precipitacion_mm"] == pytest.approx(2.0)


def test_day_of_year_is_resolved_against_the_leap_year() -> None:
    text = "\n".join(["-END HEADER-", "YEAR,DOY,T2M", "2024,60,25.0", "2024,61,26.0"])
    parsed = parse_nasa_power_csv(text)

    assert parsed.frame.loc[0, "fecha"] == pd.Timestamp("2024-02-29")
    assert parsed.frame.loc[1, "fecha"] == pd.Timestamp("2024-03-01")


def test_invalid_dates_are_dropped_and_counted() -> None:
    text = "\n".join(
        [
            "-END HEADER-",
            "YEAR,DOY,T2M",
            "2024,0,25.0",
            "2024,999,25.0",
            "2024,5,26.0",
        ]
    )
    parsed = parse_nasa_power_csv(text)

    assert parsed.days == 1
    assert parsed.invalid_rows == 2
    assert parsed.frame.loc[0, "fecha"] == pd.Timestamp("2024-01-05")


def test_last_duplicate_date_wins() -> None:
    text = "\n".join(
        [
            "-END HEADER-",
            "YEAR,DOY,T2M",
            "2024,5,20.0",
            "2024,5,30.0",
        ]
    )
    parsed = parse_nasa_power_csv(text)

    assert parsed.days == 1
    assert parsed.frame.loc[0, "temp_media_c"] == pytest.approx(30.0)


def test_bytes_and_path_inputs_are_supported(tmp_path) -> None:
    text = make_nasa_power_csv(n_days=5)
    path = tmp_path / "nasa.csv"
    path.write_text(text, encoding="utf-8")

    from_bytes = parse_nasa_power_csv(text.encode("utf-8"))
    from_path = parse_nasa_power_csv(path)

    assert from_bytes.days == from_path.days == 5
    assert from_bytes.start == from_path.start


def test_missing_header_raises() -> None:
    with pytest.raises(ValidationFailedError, match="YEAR,DOY"):
        parse_nasa_power_csv("just,some,rows\n1,2,3\n")


def test_no_recognised_parameters_raises() -> None:
    text = "\n".join(["-END HEADER-", "YEAR,DOY,UNKNOWN", "2024,1,1.0"])
    with pytest.raises(ValidationFailedError, match="no recognised parameter"):
        parse_nasa_power_csv(text)


def test_empty_csv_raises() -> None:
    with pytest.raises(ValidationFailedError, match="no data rows"):
        parse_nasa_power_csv("-END HEADER-\nYEAR,DOY,T2M\n")
