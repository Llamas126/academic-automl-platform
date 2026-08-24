from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from app.core.config import get_settings
from app.core.exceptions import (
    NotFoundError,
    PayloadTooLargeError,
    UnsupportedFileError,
    ValidationFailedError,
)
from app.ml.profiling.type_detector import profile_dataframe
from app.schemas.dataset import DatasetProfile
from app.storage.local_storage import sanitize_filename, storage

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
UPLOAD_CHUNK_BYTES = 1024 * 1024


async def read_upload_capped(upload, max_bytes: int) -> bytes:
    """Read an UploadFile in chunks, aborting as soon as the cap is exceeded.

    Prevents full-in-memory buffering of oversized bodies before the size
    check happens (memory-exhaustion DoS).
    """
    buffer = bytearray()
    while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            settings = get_settings()
            raise PayloadTooLargeError(f"file exceeds upload limit of {settings.max_upload_mb} MB")
    return bytes(buffer)


def read_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv_robust(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
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
        raise ValidationFailedError(f"could not parse CSV file: {exc}") from exc


def ingest_dataset(filename: str, content: bytes) -> DatasetProfile:
    settings = get_settings()
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileError(
            f"file type '{suffix}' not allowed; supported: {sorted(ALLOWED_EXTENSIONS)}"
        )
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise PayloadTooLargeError(f"file exceeds upload limit of {settings.max_upload_mb} MB")

    safe_name = sanitize_filename(filename)
    dataset_id, path = storage.save_upload(safe_name, content)
    try:
        df = read_tabular(path)
    except (UnsupportedFileError, ValidationFailedError):
        storage.delete_dataset(dataset_id)
        raise
    except Exception as exc:
        storage.delete_dataset(dataset_id)
        raise ValidationFailedError(f"failed to parse '{safe_name}': {exc}") from exc

    if df.empty:
        storage.delete_dataset(dataset_id)
        raise ValidationFailedError("dataset contains no rows")
    if len(df.columns) < 2:
        storage.delete_dataset(dataset_id)
        raise ValidationFailedError("dataset must contain at least two columns")
    if df.shape[0] > settings.max_dataset_rows:
        storage.delete_dataset(dataset_id)
        raise ValidationFailedError(f"dataset has {df.shape[0]} rows; limit is {settings.max_dataset_rows}")
    if df.shape[1] > settings.max_dataset_columns:
        storage.delete_dataset(dataset_id)
        raise ValidationFailedError(
            f"dataset has {df.shape[1]} columns; limit is {settings.max_dataset_columns}"
        )

    profile = profile_dataframe(df, dataset_id=dataset_id, filename=safe_name)
    storage.save_json(storage.profile_path(dataset_id), profile.model_dump(mode="json"))
    logger.info("profiled dataset %s (%d rows x %d cols)", dataset_id, df.shape[0], df.shape[1])
    return profile


def load_dataset_dataframe(dataset_id: str) -> pd.DataFrame:
    profile_path = storage.profile_path(dataset_id)
    if not profile_path.exists():
        raise NotFoundError(f"dataset '{dataset_id}' not found")
    record = storage.load_json(profile_path)
    path = storage.dataset_file(dataset_id, record["filename"])
    if not path.exists():
        raise NotFoundError(f"source file for dataset '{dataset_id}' is missing")
    return read_tabular(path)


def get_profile(dataset_id: str) -> DatasetProfile:
    path = storage.profile_path(dataset_id)
    if not path.exists():
        raise NotFoundError(f"dataset '{dataset_id}' not found")
    return DatasetProfile.model_validate(storage.load_json(path))


def delete_dataset(dataset_id: str) -> None:
    if not storage.profile_path(dataset_id).exists():
        raise NotFoundError(f"dataset '{dataset_id}' not found")
    storage.delete_dataset(dataset_id)
