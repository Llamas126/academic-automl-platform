"""TTL cache for the optional NASA POWER online refresh.

The cache is content-addressed by the exact query parameters (location,
requested range, parameter list). A cached payload is reused while it is
younger than ``MANGO_CACHE_TTL_HOURS`` and its SHA-256 still matches, so a
restarted service or a repeated sync never re-downloads the same series.

The cache never owns the truth: if the entry is missing, expired or corrupt,
callers fall back to the local CSV (see ``nasa_api.load_climate_source``).
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .nasa_api import NASA_POWER_URL, build_nasa_params
from .spec import MangoDatasetSpec

CACHE_SCHEMA_VERSION = 1
METADATA_FILENAME = "nasa_power.json"
CSV_FILENAME = "nasa_power.csv"


def cache_key(spec: MangoDatasetSpec) -> str:
    """Stable fingerprint of the NASA POWER query for ``spec``."""
    payload = json.dumps(
        {"url": NASA_POWER_URL, "params": build_nasa_params(spec)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("could not decode cached NASA POWER CSV")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with temp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class CachedClimate:
    text: str
    key: str
    fetched_at: datetime
    sha256: str


class NasaCache:
    """File-backed TTL cache rooted at ``.cache/`` of the managed dataset."""

    def __init__(self, root: str | Path, ttl_hours: int):
        self.root = Path(root)
        self.ttl_hours = max(0, int(ttl_hours))

    @property
    def metadata_path(self) -> Path:
        return self.root / METADATA_FILENAME

    @property
    def csv_path(self) -> Path:
        return self.root / CSV_FILENAME

    @property
    def enabled(self) -> bool:
        return self.ttl_hours > 0

    def metadata(self) -> dict | None:
        try:
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def get(
        self,
        spec: MangoDatasetSpec,
        *,
        now: datetime | None = None,
        force: bool = False,
    ) -> CachedClimate | None:
        """Return a fresh, hash-verified entry or ``None``."""
        if force or not self.enabled:
            return None
        entry = self.metadata()
        if not entry or entry.get("schema_version") != CACHE_SCHEMA_VERSION:
            return None
        key = cache_key(spec)
        if entry.get("key") != key:
            return None
        try:
            fetched_at = datetime.fromisoformat(str(entry["fetched_at"]).replace("Z", "+00:00"))
        except (KeyError, ValueError):
            return None
        moment = now or datetime.now(UTC)
        if moment - fetched_at >= timedelta(hours=self.ttl_hours):
            return None
        if not self.csv_path.is_file():
            return None
        raw = self.csv_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != entry.get("csv_sha256"):
            return None
        try:
            text = _decode(raw)
        except ValueError:
            return None
        return CachedClimate(text=text, key=key, fetched_at=fetched_at, sha256=digest)

    def put(
        self,
        spec: MangoDatasetSpec,
        text: str,
        *,
        now: datetime | None = None,
    ) -> CachedClimate:
        """Persist the raw CSV plus its metadata; returns the stored entry."""
        raw = text.encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        moment = now or datetime.now(UTC)
        key = cache_key(spec)
        _atomic_write(self.csv_path, raw)
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "key": key,
            "params": build_nasa_params(spec),
            "fetched_at": moment.isoformat().replace("+00:00", "Z"),
            "ttl_hours": self.ttl_hours,
            "csv_sha256": digest,
            "bytes": len(raw),
        }
        _atomic_write(
            self.metadata_path,
            json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"),
        )
        return CachedClimate(text=text, key=key, fetched_at=moment, sha256=digest)

    def clear(self) -> None:
        self.metadata_path.unlink(missing_ok=True)
        self.csv_path.unlink(missing_ok=True)
