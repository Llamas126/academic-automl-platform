"""Build the mango domain spec from the service settings."""

from __future__ import annotations

from datetime import date

from automl_shared.core.config import Settings
from automl_shared.core.validation import validate_resource_id

from .domain.mango.spec import COLLECTED_TARGET, MangoDatasetSpec


def build_spec(
    settings: Settings,
    *,
    requested_start: date | None = None,
    requested_end: date | None = None,
    include_uncovered_rows: bool | None = None,
) -> MangoDatasetSpec:
    validate_resource_id(settings.mango_dataset_id, "dataset id")
    return MangoDatasetSpec(
        dataset_id=settings.mango_dataset_id,
        date_start=requested_start or settings.mango_date_start,
        date_end=requested_end or settings.mango_date_end,
        latitude=settings.mango_latitude,
        longitude=settings.mango_longitude,
        strict_coverage=settings.mango_strict_coverage,
        include_uncovered_rows=(
            settings.mango_include_uncovered_rows
            if include_uncovered_rows is None
            else include_uncovered_rows
        ),
        allow_edge_fill=settings.mango_allow_edge_fill,
        min_positive_days=settings.mango_min_positive_days,
        require_targets=(COLLECTED_TARGET,),
    )
