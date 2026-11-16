"""Resolution of the configured mango sources.

Three places can hold a source:

* an explicit registration in ``sources/registry.json`` (``PUT /sources/...``),
* a drop file under ``MANGO_SOURCE_DIR`` (source of truth in runtime),
* an uploaded copy under the managed dataset's ``sources/`` directory.

Candidates are detected by content (NASA ``YEAR``/``DOY`` header, local date +
target aliases), not by filename. The newest modification time wins, with the
explicit registration as tie-breaker, so a new farm export is picked up on the
next sync without touching configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from automl_shared.core.exceptions import ValidationFailedError
from automl_shared.tabular import ALLOWED_EXTENSIONS

from .local_records import aliases_for
from .spec import DATE_ALIASES, NASA_PARAM_MAP, MangoDatasetSpec, normalize_header

REGISTRY_SCHEMA_VERSION = 1
SourceRoot = Literal["source_dir", "managed"]
SOURCE_ORIGINS = ("upload", "source_dir")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_datetime(value: object) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def resolve_source_path(root: Path, relative: str) -> Path:
    """Join ``relative`` to ``root`` and refuse to escape it."""
    base = Path(root).resolve()
    candidate = (base / relative).resolve()
    if not candidate.is_relative_to(base):
        raise ValidationFailedError(f"source path escapes its root: {relative}")
    return candidate


@dataclass(frozen=True, slots=True)
class RegisteredSource:
    root: SourceRoot
    path: str
    origin: str
    sha256: str
    filename: str
    registered_at: datetime
    target: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "root": self.root,
            "path": self.path,
            "origin": self.origin,
            "sha256": self.sha256,
            "filename": self.filename,
            "registered_at": self.registered_at.isoformat().replace("+00:00", "Z"),
        }


@dataclass(slots=True)
class SourceRegistry:
    nasa: RegisteredSource | None = None
    local: dict[str, RegisteredSource | None] = field(default_factory=dict)
    updated_at: datetime | None = None

    @classmethod
    def empty(cls, targets: tuple[str, ...]) -> SourceRegistry:
        return cls(local={target: None for target in targets})

    @classmethod
    def load(cls, path: Path, targets: tuple[str, ...]) -> SourceRegistry:
        if not path.is_file():
            return cls.empty(targets)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls.empty(targets)
        if not isinstance(payload, dict):
            return cls.empty(targets)

        def parse(raw: object, target: str | None) -> RegisteredSource | None:
            if not isinstance(raw, dict):
                return None
            root = raw.get("root")
            registered_at = _parse_datetime(raw.get("registered_at"))
            if root not in ("source_dir", "managed") or registered_at is None:
                return None
            path_value = str(raw.get("path") or "")
            if not path_value:
                return None
            return RegisteredSource(
                root=root,
                path=path_value,
                origin=str(raw.get("origin") or "source_dir"),
                sha256=str(raw.get("sha256") or ""),
                filename=str(raw.get("filename") or Path(path_value).name),
                registered_at=registered_at,
                target=target,
            )

        return cls(
            nasa=parse(payload.get("nasa"), None),
            local={target: parse((payload.get("local") or {}).get(target), target) for target in targets},
            updated_at=_parse_datetime(payload.get("updated_at")),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "updated_at": (datetime.now(UTC)).isoformat().replace("+00:00", "Z"),
            "nasa": self.nasa.as_dict() if self.nasa else None,
            "local": {
                target: entry.as_dict() if entry else None for target, entry in self.local.items()
            },
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.updated_at = datetime.now(UTC)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "updated_at": (
                self.updated_at.isoformat().replace("+00:00", "Z") if self.updated_at else None
            ),
            "nasa": self.nasa.as_dict() if self.nasa else None,
            "local": {
                target: entry.as_dict() if entry else None for target, entry in self.local.items()
            },
        }


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    key: str
    root: SourceRoot
    path: str
    absolute: Path
    filename: str
    origin: str
    mtime: float
    priority: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "path": self.path,
            "filename": self.filename,
            "origin": self.origin,
            "modified_at": datetime.fromtimestamp(self.mtime, tz=UTC)
            .isoformat()
            .replace("+00:00", "Z"),
        }


@dataclass(frozen=True, slots=True)
class ResolvedSource:
    key: str
    root: SourceRoot
    path: str
    absolute: Path
    filename: str
    origin: str
    mtime: float
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "path": self.path,
            "filename": self.filename,
            "origin": self.origin,
            "sha256": self.sha256,
            "modified_at": datetime.fromtimestamp(self.mtime, tz=UTC)
            .isoformat()
            .replace("+00:00", "Z"),
        }


@dataclass(frozen=True, slots=True)
class SourceResolution:
    climate: ResolvedSource | None
    local: dict[str, ResolvedSource | None]
    warnings: tuple[str, ...]
    candidates: dict[str, tuple[SourceCandidate, ...]]

    def as_dict(self) -> dict[str, object]:
        return {
            "climate": self.climate.as_dict() if self.climate else None,
            "local": {
                target: source.as_dict() if source else None
                for target, source in self.local.items()
            },
            "candidates": {
                key: [candidate.as_dict() for candidate in values]
                for key, values in self.candidates.items()
            },
            "warnings": list(self.warnings),
        }


def _read_text_head(path: Path, limit: int = 65536) -> str:
    try:
        with path.open("rb") as handle:
            raw = handle.read(limit)
    except OSError:
        return ""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return ""


def looks_like_nasa_power(path: str | Path) -> bool:
    """True when the file carries a NASA POWER ``YEAR,DOY`` data header."""
    source = Path(path)
    if source.suffix.lower() != ".csv":
        return False
    text = _read_text_head(source)
    for line in text.splitlines():
        fields = {normalize_header(part) for part in line.split(",")}
        if {"year", "doy"} <= fields and fields & set(NASA_PARAM_MAP):
            return True
    return False


def read_header(path: str | Path) -> list[str]:
    """Read only the column names of a tabular source, tolerating separators."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        try:
            return [str(column) for column in pd.read_excel(source, nrows=0).columns]
        except (OSError, ValueError):
            return []
    if suffix != ".csv":
        return []
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        for separator in (None, ";", "\t", "|"):
            try:
                frame = pd.read_csv(
                    source,
                    nrows=0,
                    encoding=encoding,
                    sep=separator,
                    engine="python" if separator is None else None,
                )
            except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError):
                continue
            if len(frame.columns) >= 2 or separator is None:
                return [str(column) for column in frame.columns]
    return []


def _find_header(header: list[str], aliases: frozenset[str]) -> str | None:
    for column in header:
        if normalize_header(column) in aliases:
            return column
    return None


def detect_local_targets(path: str | Path, spec: MangoDatasetSpec) -> list[str]:
    """Targets whose alias column (plus a date column) is present in ``path``."""
    header = read_header(path)
    if not header or _find_header(header, DATE_ALIASES) is None:
        return []
    return [
        target for target in spec.target_names if _find_header(header, aliases_for(target, spec))
    ]


def _iter_files(root: Path, *, limit: int, max_bytes: int) -> list[tuple[Path, float]]:
    if not root.is_dir():
        return []
    found: list[tuple[Path, float]] = []
    for current, directories, filenames in os.walk(root):
        directories[:] = sorted(name for name in directories if not name.startswith("."))
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            path = Path(current) / name
            if path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size == 0 or stat.st_size > max_bytes:
                continue
            found.append((path, stat.st_mtime))
            if len(found) >= limit:
                return found
    return found


def _candidate_from_registration(
    registration: RegisteredSource,
    *,
    key: str,
    dataset_dir: Path,
    source_dir: Path,
    spec: MangoDatasetSpec,
) -> SourceCandidate | None:
    base = source_dir if registration.root == "source_dir" else dataset_dir
    absolute = resolve_source_path(base, registration.path)
    if not absolute.is_file():
        return None
    if key == "climate":
        matches = looks_like_nasa_power(absolute)
    else:
        matches = key in detect_local_targets(absolute, spec)
    if not matches:
        return None
    return SourceCandidate(
        key=key,
        root=registration.root,
        path=registration.path,
        absolute=absolute,
        filename=registration.filename,
        origin=registration.origin,
        mtime=absolute.stat().st_mtime,
        priority=1,
    )


def resolve_sources(
    spec: MangoDatasetSpec,
    *,
    source_dir: str | Path,
    dataset_dir: str | Path,
    registry: SourceRegistry,
    scan_limit: int = 500,
    max_source_mb: int = 100,
) -> SourceResolution:
    """Pick the winning file for the climate series and for each local target."""
    source_root = Path(source_dir)
    managed_root = Path(dataset_dir)
    managed_sources = managed_root / "sources"
    max_bytes = max(1, max_source_mb) * 1024 * 1024

    candidates: dict[str, list[SourceCandidate]] = {
        "climate": [],
        **{target: [] for target in spec.target_names},
    }
    warnings: list[str] = []

    registrations: dict[str, RegisteredSource | None] = {
        "climate": registry.nasa,
        **{target: registry.local.get(target) for target in spec.target_names},
    }
    for key, registration in registrations.items():
        if registration is None:
            continue
        try:
            candidate = _candidate_from_registration(
                registration,
                key=key,
                dataset_dir=managed_root,
                source_dir=source_root,
                spec=spec,
            )
        except ValidationFailedError as exc:
            warnings.append(f"ignoring registration for '{key}': {exc}")
            continue
        if candidate is None:
            warnings.append(
                f"registered source for '{key}' is missing or no longer matches: {registration.path}"
            )
            continue
        candidates[key].append(candidate)

    for root, label in ((source_root, "source_dir"), (managed_sources, "managed")):
        for path, mtime in _iter_files(root, limit=scan_limit, max_bytes=max_bytes):
            base = managed_root if label == "managed" else source_root
            try:
                relative = path.relative_to(base).as_posix()
            except ValueError:
                continue
            common = dict(
                root=label,
                path=relative,
                absolute=path,
                filename=path.name,
                origin=label,
                mtime=mtime,
            )
            if looks_like_nasa_power(path):
                candidates["climate"].append(SourceCandidate(key="climate", **common))
            for target in detect_local_targets(path, spec):
                candidates[target].append(SourceCandidate(key=target, **common))

    resolved_candidates: dict[str, tuple[SourceCandidate, ...]] = {}
    resolved: dict[str, ResolvedSource | None] = {}
    for key, values in candidates.items():
        deduped: dict[Path, SourceCandidate] = {}
        for candidate in values:
            previous = deduped.get(candidate.absolute)
            if previous is None or candidate.priority > previous.priority:
                deduped[candidate.absolute] = candidate
        ordered = sorted(
            deduped.values(), key=lambda item: (item.mtime, item.priority, item.path)
        )
        resolved_candidates[key] = tuple(reversed(ordered))
        if not ordered:
            resolved[key] = None
            continue
        best = ordered[-1]
        if len(ordered) > 1:
            warnings.append(
                f"multiple candidate sources for '{key}'; using '{best.path}' (newest modification)"
            )
        try:
            digest = sha256_file(best.absolute)
        except OSError as exc:
            raise ValidationFailedError(f"could not read source '{best.path}': {exc}") from exc
        resolved[key] = ResolvedSource(
            key=key,
            root=best.root,
            path=best.path,
            absolute=best.absolute,
            filename=best.filename,
            origin=best.origin,
            mtime=best.mtime,
            sha256=digest,
        )

    return SourceResolution(
        climate=resolved["climate"],
        local={target: resolved[target] for target in spec.target_names},
        warnings=tuple(warnings),
        candidates=resolved_candidates,
    )
