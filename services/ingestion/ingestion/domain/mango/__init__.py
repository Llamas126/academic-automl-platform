"""Domain layer for the Mango de Hilaza (Malambo) dataset."""

from .canonical import CanonicalResult, TargetStatus, build_canonical, frame_to_csv_bytes
from .coverage import (
    CoverageReport,
    DateRange,
    MissingRange,
    compute_coverage,
)
from .features import (
    CALENDAR_FEATURE_COLUMNS,
    CLIMATE_FEATURE_COLUMNS,
    add_features,
)
from .local_records import LocalRecords, LocalTarget, build_local_records
from .nasa_api import (
    ClimateSource,
    NasaFetchError,
    build_nasa_params,
    fetch_nasa_power_csv,
    load_climate_source,
)
from .nasa_power import ParsedClimate, parse_nasa_power_csv
from .spec import (
    CLIMATE_COLUMNS,
    COLLECTED_TARGET,
    DATE_COLUMN,
    LOST_TARGET,
    MANAGED_DOMAIN,
    NASA_PARAMETERS,
    MangoDatasetSpec,
    normalize_header,
)

__all__ = [
    "CALENDAR_FEATURE_COLUMNS",
    "CLIMATE_COLUMNS",
    "CLIMATE_FEATURE_COLUMNS",
    "COLLECTED_TARGET",
    "CanonicalResult",
    "ClimateSource",
    "CoverageReport",
    "DATE_COLUMN",
    "DateRange",
    "LOST_TARGET",
    "LocalRecords",
    "LocalTarget",
    "MANAGED_DOMAIN",
    "MangoDatasetSpec",
    "MissingRange",
    "NASA_PARAMETERS",
    "NasaFetchError",
    "ParsedClimate",
    "TargetStatus",
    "add_features",
    "build_canonical",
    "build_local_records",
    "build_nasa_params",
    "compute_coverage",
    "fetch_nasa_power_csv",
    "frame_to_csv_bytes",
    "load_climate_source",
    "normalize_header",
    "parse_nasa_power_csv",
]
