"""Optional online refresh of the NASA POWER daily climate series.

The local CSV stays the source of truth: the API is only used when the online
refresh flag is enabled, and any failure falls back to the file with a warning
instead of breaking the ingestion flow.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from automl_shared.core.exceptions import ValidationFailedError

from .nasa_power import ParsedClimate, parse_nasa_power_csv
from .spec import NASA_PARAMETERS, MangoDatasetSpec

if TYPE_CHECKING:
    from .cache import NasaCache

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_COMMUNITY = "AG"


class NasaFetchError(RuntimeError):
    """The NASA POWER endpoint was unreachable or returned unusable data."""


@dataclass(frozen=True, slots=True)
class ClimateSource:
    climate: ParsedClimate
    origin: str
    warnings: tuple[str, ...] = ()
    sha256: str | None = None
    params: dict[str, str] | None = None
    fetched_at: datetime | None = None


def build_nasa_params(spec: MangoDatasetSpec) -> dict[str, str]:
    """Query parameters for the daily point endpoint."""
    return {
        "parameters": ",".join(NASA_PARAMETERS),
        "community": NASA_COMMUNITY,
        "latitude": f"{spec.latitude:.4f}",
        "longitude": f"{spec.longitude:.4f}",
        "start": spec.date_start.strftime("%Y%m%d"),
        "end": spec.date_end.strftime("%Y%m%d"),
        "format": "CSV",
    }


def fetch_nasa_power_csv(
    spec: MangoDatasetSpec,
    *,
    client: httpx.Client | None = None,
    retries: int = 3,
    backoff: float = 1.0,
    timeout: float = 30.0,
) -> str:
    """Download the NASA POWER CSV, retrying transient failures."""
    if retries < 1:
        raise ValueError("retries must be at least 1")

    owns_client = client is None
    http = client or httpx.Client(timeout=timeout, follow_redirects=True)
    last_error: Exception | None = None
    try:
        for attempt in range(1, retries + 1):
            try:
                response = http.get(NASA_POWER_URL, params=build_nasa_params(spec))
            except httpx.TransportError as exc:
                last_error = exc
            else:
                if response.status_code == 200:
                    text = response.text
                    if "YEAR" in text and "DOY" in text:
                        return text
                    last_error = NasaFetchError("NASA POWER response has no YEAR/DOY header")
                elif response.status_code == 429 or response.status_code >= 500:
                    last_error = NasaFetchError(f"NASA POWER responded {response.status_code}")
                else:
                    snippet = response.text[:200].strip()
                    raise NasaFetchError(f"NASA POWER responded {response.status_code}: {snippet}")

            if attempt < retries and backoff > 0:
                time.sleep(backoff * (2 ** (attempt - 1)))

        raise NasaFetchError(
            f"NASA POWER request failed after {retries} attempts: {last_error}"
        ) from last_error
    finally:
        if owns_client:
            http.close()


def load_climate_source(
    spec: MangoDatasetSpec,
    *,
    csv_path: str | Path | None,
    online: bool = False,
    cache: NasaCache | None = None,
    force: bool = False,
    client: httpx.Client | None = None,
    retries: int = 3,
    backoff: float = 1.0,
    timeout: float = 30.0,
) -> ClimateSource:
    """Return climate data from the API/cache (when enabled) or the local CSV.

    The API is a refresh, not a replacement: if it fails, the local file is
    parsed and a warning is attached so the caller can surface it. ``cache``
    deduplicates downloads within ``MANGO_CACHE_TTL_HOURS``; ``force`` bypasses
    the read side of that cache.
    """
    warnings: list[str] = []
    path = Path(csv_path) if csv_path is not None else None

    if online and cache is not None and not force:
        cached = cache.get(spec)
        if cached is not None:
            climate = parse_nasa_power_csv(cached.text, spec=spec)
            return ClimateSource(
                climate=climate,
                origin="nasa_cache",
                warnings=(),
                sha256=cached.sha256,
                params=build_nasa_params(spec),
                fetched_at=cached.fetched_at,
            )

    if online:
        try:
            text = fetch_nasa_power_csv(
                spec, client=client, retries=retries, backoff=backoff, timeout=timeout
            )
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if cache is not None:
                cache.put(spec, text)
            climate = parse_nasa_power_csv(text, spec=spec)
            return ClimateSource(
                climate=climate,
                origin="nasa_api",
                warnings=(),
                sha256=digest,
                params=build_nasa_params(spec),
                fetched_at=datetime.now(tz=UTC),
            )
        except (NasaFetchError, ValidationFailedError) as exc:
            warnings.append(f"NASA API refresh failed, using local CSV: {exc}")

    if path is None or not path.is_file():
        detail = f"NASA POWER CSV not found: {path}" if path else "no NASA POWER CSV configured"
        raise NasaFetchError(detail)

    raw = path.read_bytes()
    climate = parse_nasa_power_csv(raw, spec=spec)
    return ClimateSource(
        climate=climate,
        origin="local_csv",
        warnings=tuple(warnings),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
