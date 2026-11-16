import json
import os
import re
import shutil
import uuid
from pathlib import Path

from automl_shared.core.config import get_settings


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: str | None) -> str:
    """Reduce an uploaded filename to a safe single path segment.

    Strips any directory components (both separators), replaces unsafe
    characters, and never yields empty, dot-only, or traversal values.
    """
    raw = os.path.basename((filename or "").replace("\\", "/"))
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", raw).lstrip(".")
    cleaned = cleaned[:120].strip()
    if not cleaned:
        return "upload.csv"
    return cleaned


class LocalStorage:
    @property
    def settings(self):
        return get_settings()

    @property
    def uploads_dir(self) -> Path:
        return self.settings.uploads_dir

    @property
    def jobs_dir(self) -> Path:
        return self.settings.jobs_dir

    def dataset_dir(self, dataset_id: str) -> Path:
        return self.uploads_dir / dataset_id

    def dataset_file(self, dataset_id: str, filename: str) -> Path:
        return self.dataset_dir(dataset_id) / filename

    def profile_path(self, dataset_id: str) -> Path:
        return self.dataset_dir(dataset_id) / "profile.json"

    def managed_marker_path(self, dataset_id: str) -> Path:
        return self.dataset_dir(dataset_id) / ".managed"

    def is_managed(self, dataset_id: str) -> bool:
        return self.managed_marker_path(dataset_id).exists()

    def managed_manifest_path(self, dataset_id: str) -> Path:
        return self.dataset_dir(dataset_id) / "dataset.json"

    def managed_versions_dir(self, dataset_id: str) -> Path:
        return self.dataset_dir(dataset_id) / "versions"

    def managed_sources_dir(self, dataset_id: str) -> Path:
        return self.dataset_dir(dataset_id) / "sources"

    def managed_registry_path(self, dataset_id: str) -> Path:
        return self.managed_sources_dir(dataset_id) / "registry.json"

    def managed_cache_dir(self, dataset_id: str) -> Path:
        return self.dataset_dir(dataset_id) / ".cache"

    def save_upload(self, filename: str, content: bytes) -> tuple[str, Path]:
        dataset_id = new_id("ds")
        path = self.dataset_file(dataset_id, sanitize_filename(filename))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return dataset_id, path

    def write_bytes_atomic(self, path: Path, payload: bytes) -> None:
        """Replace ``path`` atomically via a sibling temp file."""
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

    def write_json_atomic(self, path: Path, payload: dict) -> None:
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        self.write_bytes_atomic(path, text.encode("utf-8"))

    def save_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def load_json(self, path: Path) -> dict:
        if not path.exists():
            from automl_shared.core.exceptions import NotFoundError

            raise NotFoundError(f"File not found: {path.name}")
        return json.loads(path.read_text(encoding="utf-8"))

    def job_work_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id / "work"

    def job_out_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id / "output"

    def job_zip_path(self, job_id: str) -> Path:
        return self.jobs_dir / job_id / f"results_{job_id}.zip"

    def delete_dataset(self, dataset_id: str) -> None:
        path = self.dataset_dir(dataset_id)
        if path.exists():
            shutil.rmtree(path)


storage = LocalStorage()
