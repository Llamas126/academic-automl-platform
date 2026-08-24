from __future__ import annotations

from pathlib import Path

import pandas as pd

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def read_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv_robust(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    from automl_shared.core.exceptions import UnsupportedFileError

    raise UnsupportedFileError(f"unsupported file extension: '{suffix}'")


def _read_csv_robust(path: Path) -> pd.DataFrame:
    for kwargs in ({}, {"sep": ";"}, {"sep": "\t"}, {"sep": "|"}, {"engine": "python", "sep": None}):
        try:
            df = pd.read_csv(path, **kwargs)
            if df.shape[1] >= 2 or kwargs.get("sep") is None:
                return df
        except (pd.errors.ParserError, ValueError, UnicodeDecodeError):
            continue
    try:
        return pd.read_csv(path, engine="python", sep=None, encoding="latin-1")
    except Exception as exc:
        from automl_shared.core.exceptions import ValidationFailedError

        raise ValidationFailedError(f"could not parse CSV file: {exc}") from exc
