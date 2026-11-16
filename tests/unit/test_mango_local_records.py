from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from automl_shared.core.exceptions import ValidationFailedError
from ingestion.domain.mango.local_records import (
    build_local_records,
    coerce_numeric,
    parse_local_dates,
)
from ingestion.domain.mango.spec import COLLECTED_TARGET, LOST_TARGET, MangoDatasetSpec


def write_csv(tmp_path, name: str, rows: dict[str, list[object]]):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_reads_spanish_headers_and_day_first_dates(tmp_path) -> None:
    path = write_csv(
        tmp_path,
        "recoleccion.csv",
        {
            "FECHA": ["01-03-2018", "02-03-2018"],
            "PESO (PRODUCCIÓN)": [12.5, 0.0],
            "LOTE": ["A", "B"],
        },
    )

    records = build_local_records({COLLECTED_TARGET: path})

    assert records.rows == 2
    assert records.span == (date(2018, 3, 1), date(2018, 3, 2))
    assert records.available_targets == {COLLECTED_TARGET: True, LOST_TARGET: False}
    assert records.frame["fecha"].iloc[0] == pd.Timestamp("2018-03-01")
    assert records.frame[COLLECTED_TARGET].tolist() == [12.5, 0.0]
    assert any(LOST_TARGET in warning for warning in records.warnings)


def test_duplicate_dates_are_summed(tmp_path) -> None:
    path = write_csv(
        tmp_path,
        "recoleccion.csv",
        {
            "FECHA": ["01-03-2018", "01-03-2018", "02-03-2018"],
            "PESO_RECOLECTADO_KG": [1.5, 2.0, 3.0],
        },
    )

    records = build_local_records({COLLECTED_TARGET: path})

    assert records.rows == 2
    assert records.targets[COLLECTED_TARGET].duplicate_rows == 1
    assert records.frame[COLLECTED_TARGET].tolist() == [3.5, 3.0]


def test_missing_value_stays_distinct_from_recorded_zero(tmp_path) -> None:
    path = write_csv(
        tmp_path,
        "recoleccion.csv",
        {
            "FECHA": ["01-03-2018", "02-03-2018"],
            "PESO_RECOLECTADO_KG": ["", 0.0],
        },
    )

    records = build_local_records({COLLECTED_TARGET: path})

    assert pd.isna(records.frame[COLLECTED_TARGET].iloc[0])
    assert records.frame[COLLECTED_TARGET].iloc[1] == 0.0


def test_decimal_comma_is_parsed(tmp_path) -> None:
    path = write_csv(
        tmp_path,
        "recoleccion.csv",
        {"FECHA": ["01-03-2018"], "PESO_RECOLECTADO_KG": ["1.234,5"]},
    )

    records = build_local_records({COLLECTED_TARGET: path})

    assert records.frame[COLLECTED_TARGET].iloc[0] == pytest.approx(1234.5)


def test_loss_file_is_picked_up_by_alias(tmp_path) -> None:
    path = write_csv(
        tmp_path,
        "perdidas.csv",
        {"FECHA": ["05/03/2018"], "Peso_Perdido_KG": [2.5]},
    )
    spec = MangoDatasetSpec(require_targets=())

    records = build_local_records({LOST_TARGET: path}, spec=spec)

    assert records.available_targets == {COLLECTED_TARGET: False, LOST_TARGET: True}
    assert records.frame[LOST_TARGET].iloc[0] == pytest.approx(2.5)
    assert records.source_by_target[LOST_TARGET] == "perdidas.csv"


def test_required_source_missing_raises() -> None:
    with pytest.raises(ValidationFailedError, match="not available"):
        build_local_records({})


def test_negative_values_raise(tmp_path) -> None:
    path = write_csv(
        tmp_path,
        "recoleccion.csv",
        {"FECHA": ["01-03-2018"], "PESO_RECOLECTADO_KG": [-1.0]},
    )

    with pytest.raises(ValidationFailedError, match="negative values"):
        build_local_records({COLLECTED_TARGET: path})


def test_optional_file_without_target_column_is_warned_not_fatal(tmp_path) -> None:
    path = write_csv(
        tmp_path,
        "perdidas.csv",
        {"FECHA": ["01-03-2018"], "OTRA_COSA": [1.0]},
    )
    spec = MangoDatasetSpec(require_targets=())

    records = build_local_records({LOST_TARGET: path}, spec=spec)

    assert records.available_targets[LOST_TARGET] is False
    assert any("no column matching target" in warning for warning in records.warnings)


def test_span_uses_target_min_and_max(tmp_path) -> None:
    path = write_csv(
        tmp_path,
        "recoleccion.csv",
        {
            "FECHA": ["10-01-2020", "2019-12-31", "not-a-date"],
            "PESO_RECOLECTADO_KG": [1.0, 2.0, 3.0],
        },
    )

    records = build_local_records({COLLECTED_TARGET: path})

    assert records.span == (date(2019, 12, 31), date(2020, 1, 10))
    assert records.targets[COLLECTED_TARGET].invalid_dates == 1


def test_parse_local_dates_handles_mixed_formats() -> None:
    values = pd.Series(["01-03-2018", "2018-03-02", "2018-03-03 00:00:00", "", None])

    parsed = parse_local_dates(values)

    assert parsed.iloc[0] == pd.Timestamp("2018-03-01")
    assert parsed.iloc[1] == pd.Timestamp("2018-03-02")
    assert parsed.iloc[2] == pd.Timestamp("2018-03-03")
    assert parsed.iloc[3:].isna().all()


def test_coerce_numeric_handles_thousands_and_commas() -> None:
    values = pd.Series(["1.234,5", "12,5", "7", "", None, "oops"])

    coerced = coerce_numeric(values)

    assert coerced.iloc[0] == pytest.approx(1234.5)
    assert coerced.iloc[1] == pytest.approx(12.5)
    assert coerced.iloc[2] == pytest.approx(7.0)
    assert coerced.iloc[3:].isna().all()
