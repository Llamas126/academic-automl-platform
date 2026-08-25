"""Dataset access over the shared storage volume.

The ML core never calls the ingestion service over HTTP: it reads the
profile + file that ingestion already persisted under ``datasets/<id>/``.
"""
from __future__ import annotations

import pandas as pd
from automl_shared.core.exceptions import NotFoundError
from automl_shared.schemas.dataset import DatasetProfile
from automl_shared.storage.local_storage import storage
from automl_shared.tabular import read_tabular


def load_profile(dataset_id: str) -> DatasetProfile:
    profile_path = storage.profile_path(dataset_id)
    if not profile_path.exists():
        raise NotFoundError(f"dataset {dataset_id} not found")
    return DatasetProfile.model_validate_json(profile_path.read_text(encoding="utf-8"))


def load_dataset_frame(dataset_id: str) -> tuple[pd.DataFrame, DatasetProfile]:
    profile = load_profile(dataset_id)
    path = storage.dataset_file(dataset_id, profile.filename)
    return read_tabular(path), profile
