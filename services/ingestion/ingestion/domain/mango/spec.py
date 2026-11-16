"""Declarative specification of the Mango de Hilaza (Malambo) dataset.

The spec keeps every naming decision in one place: source aliases, NASA POWER
parameter mapping, target names and coverage policy. Domain modules receive a
``MangoDatasetSpec`` instance so they never read settings directly.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date

MANAGED_DOMAIN = "mango_hilaza_malambo"
DEFAULT_DATASET_ID = "ds_mango_malambo"
DEFAULT_CANONICAL_FILENAME = "mango_daily.csv"

DATE_COLUMN = "fecha"
COLLECTED_TARGET = "Peso_Recolectado_KG"
LOST_TARGET = "Peso_Perdido_KG"

NASA_MISSING_VALUE = -999.0

#: Parameters requested from the NASA POWER daily point endpoint, in order.
NASA_PARAMETERS: tuple[str, ...] = (
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "T2MDEW",
    "PRECTOTCORR",
    "RH2M",
    "WS2M",
    "ALLSKY_SFC_SW_DWN",
)

#: NASA POWER parameter (normalised) -> canonical column name.
NASA_PARAM_MAP: dict[str, str] = {
    "t2m": "temp_media_c",
    "t2m_max": "temp_max_c",
    "t2m_min": "temp_min_c",
    "t2mdew": "punto_rocio_c",
    "prectotcorr": "precipitacion_mm",
    "prectot": "precipitacion_mm",
    "rh2m": "humedad_relativa_pct",
    "ws2m": "viento_ms",
    "allsky_sfc_sw_dwn": "radiacion_mj_m2",
}

CLIMATE_COLUMNS: tuple[str, ...] = tuple(dict.fromkeys(NASA_PARAM_MAP.values()))

DATE_ALIASES = frozenset(
    {
        "fecha",
        "fec",
        "date",
        "dia",
        "fecha_corte",
        "fecha_registro",
        "fecha_recoleccion",
    }
)

COLLECTED_ALIASES = frozenset(
    {
        "peso_recolectado_kg",
        "peso_recolectado",
        "recolectado_kg",
        "recolectado",
        "peso_produccion_kg",
        "peso_produccion",
        "produccion_kg",
        "produccion",
        "peso_aprovechado_kg",
        "peso_aprovechado",
        "harvest_kg",
        "collected_kg",
        "yield_kg",
    }
)

LOST_ALIASES = frozenset(
    {
        "peso_perdido_kg",
        "peso_perdido",
        "perdido_kg",
        "perdido",
        "peso_merma_kg",
        "peso_merma",
        "merma_kg",
        "merma",
        "peso_perdida_kg",
        "peso_perdida",
        "perdida_kg",
        "perdida",
        "loss_kg",
        "lost_kg",
        "waste_kg",
    }
)

#: Day-first formats come first: the farm records use ``dd-mm-yyyy`` and
#: ``format="mixed"`` would silently read ``01-03-2018`` as 3 January.
DATE_FORMATS: tuple[str, ...] = (
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%y",
    "%d/%m/%y",
    "%Y%m%d",
)

_SEPARATORS = re.compile(r"[^0-9a-z]+")


def normalize_header(name: object) -> str:
    """Fold a spreadsheet header into a comparable snake_case token.

    ``"PESO (PRODUCCIÓN)"`` -> ``"peso_produccion"``. Accents are stripped
    before normalising so both Spanish spellings resolve to the same alias.
    """
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return _SEPARATORS.sub("_", text.strip().lower()).strip("_")


@dataclass(frozen=True, slots=True)
class MangoDatasetSpec:
    """Naming + policy contract for the managed mango dataset."""

    dataset_id: str = DEFAULT_DATASET_ID
    canonical_filename: str = DEFAULT_CANONICAL_FILENAME
    date_start: date = date(2018, 3, 1)
    date_end: date = date(2026, 8, 30)
    collected_target: str = COLLECTED_TARGET
    lost_target: str = LOST_TARGET
    latitude: float = 10.85
    longitude: float = -74.78
    strict_coverage: bool = False
    include_uncovered_rows: bool = False
    allow_edge_fill: bool = False
    min_positive_days: int = 30
    require_targets: tuple[str, ...] = (COLLECTED_TARGET,)

    def __post_init__(self) -> None:
        if self.date_start > self.date_end:
            raise ValueError("date_start must be on or before date_end")
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError("latitude out of range")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError("longitude out of range")
        allowed = {self.collected_target, self.lost_target}
        unknown = [name for name in self.require_targets if name not in allowed]
        if unknown:
            raise ValueError(f"require_targets contains unknown targets: {unknown}")

    @property
    def target_names(self) -> tuple[str, str]:
        return (self.collected_target, self.lost_target)

    def as_dict(self) -> dict[str, object]:
        """Stable, JSON-ready view used by fingerprints and the manifest."""
        return {
            "dataset_id": self.dataset_id,
            "canonical_filename": self.canonical_filename,
            "requested_start": self.date_start.isoformat(),
            "requested_end": self.date_end.isoformat(),
            "latitude": self.latitude,
            "longitude": self.longitude,
            "strict_coverage": self.strict_coverage,
            "include_uncovered_rows": self.include_uncovered_rows,
            "allow_edge_fill": self.allow_edge_fill,
            "min_positive_days": self.min_positive_days,
            "require_targets": list(self.require_targets),
            "collected_target": self.collected_target,
            "lost_target": self.lost_target,
        }
