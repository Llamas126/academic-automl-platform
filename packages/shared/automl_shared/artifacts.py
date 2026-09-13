from __future__ import annotations

import hashlib
import hmac
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from automl_shared.core.config import get_settings
from automl_shared.schemas.results import RunSummary

logger = logging.getLogger(__name__)

MANIFEST_NAME = "models.sha256"
_HASH_CHUNK_BYTES = 65536


def save_final_models(artifacts: dict, models_dir: Path) -> None:
    import joblib

    for target_name, artifact in artifacts.items():
        if artifact.final_pipeline is None:
            continue
        safe_target = _safe_name(target_name)
        payload = {
            "pipeline": artifact.final_pipeline,
            "target": target_name,
            "task": artifact.task,
            "model_key": artifact.best_entry_key,
            "hyperparameters": artifact.best_params,
            "label_classes": artifact.label_classes,
        }
        joblib.dump(payload, models_dir / f"{safe_target}_best_model.joblib")


def save_all_models(artifacts: dict, models_dir: Path) -> None:
    """Persist a refit pipeline for every evaluated candidate model per target.

    The winner is already stored as ``<target>_best_model.joblib`` by
    ``save_final_models``; this only writes the remaining candidates so every
    evaluated algorithm can be inspected downstream.
    """
    import joblib

    for target_name, artifact in artifacts.items():
        best_key = artifact.best_entry_key
        safe_target = _safe_name(target_name)
        for key, candidate in (getattr(artifact, "candidate_models", None) or {}).items():
            if key == best_key or candidate.get("pipeline") is None:
                continue
            safe_key = _safe_name(key)
            payload = {
                "pipeline": candidate["pipeline"],
                "target": target_name,
                "task": artifact.task,
                "model_key": candidate.get("key") or key,
                "hyperparameters": candidate.get("params"),
                "label_classes": artifact.label_classes,
                "display_name": candidate.get("display_name"),
                "explainer_kind": candidate.get("explainer_kind"),
                "rank": candidate.get("rank"),
            }
            joblib.dump(payload, models_dir / f"{safe_target}_{safe_key}.joblib")


def model_digest(path: Path, signing_key: str | None = None) -> str:
    """Streaming SHA-256 of the artifact; HMAC-SHA256 when a key is configured."""
    if signing_key:
        digest = hmac.new(signing_key.encode("utf-8"), digestmod=hashlib.sha256)
    else:
        digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_model_manifest(models_dir: Path) -> Path:
    settings = get_settings()
    signing_key = settings.artifact_signing_key or None
    lines = [
        f"{model_digest(path, signing_key)}  {path.name}" for path in sorted(models_dir.glob("*.joblib"))
    ]
    manifest_path = models_dir / MANIFEST_NAME
    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return manifest_path


def verify_model_integrity(models_dir: Path) -> None:
    """Verify every .joblib against the manifest before deserialization.

    Modes (settings.artifact_integrity):
      - ``off``: skip verification entirely.
      - ``if_present``: verify when the manifest exists; legacy artifacts
        without one are still accepted.
      - ``strict``: a missing manifest is itself a failure.
    """
    settings = get_settings()
    mode = settings.artifact_integrity
    if mode == "off":
        return
    manifest_path = models_dir / MANIFEST_NAME
    if not manifest_path.exists():
        if mode == "strict":
            from automl_shared.core.exceptions import ExportError

            raise ExportError("model artifact integrity manifest is missing")
        logger.info("no integrity manifest in %s; skipping verification", models_dir.name)
        return

    expected: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, separator, name = line.partition("  ")
        if not separator or not digest or not name:
            from automl_shared.core.exceptions import ExportError

            raise ExportError("model artifact integrity manifest is malformed")
        expected[name] = digest

    actual_files = {path.name for path in models_dir.glob("*.joblib")}
    untracked = sorted(actual_files - set(expected))
    absent = sorted(set(expected) - actual_files)
    if untracked or absent:
        from automl_shared.core.exceptions import ExportError

        if untracked:
            raise ExportError(f"model artifact(s) not covered by integrity manifest: {untracked}")
        raise ExportError(f"integrity manifest references missing artifact(s): {absent}")

    signing_key = settings.artifact_signing_key or None
    for name, digest in expected.items():
        actual = model_digest(models_dir / name, signing_key)
        if not hmac.compare_digest(actual, digest):
            logger.error("artifact integrity mismatch for %s", name)
            from automl_shared.core.exceptions import ExportError

            raise ExportError("model artifact integrity verification failed")


def save_run_summary(summary: RunSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary.model_dump(mode="json"), indent=2), encoding="utf-8")


def load_model(path: Path) -> dict:
    import joblib

    return joblib.load(path)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
