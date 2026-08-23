import shutil
import uuid
from pathlib import Path

from app.core.config import get_settings


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


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

    def save_upload(self, filename: str, content: bytes) -> tuple[str, Path]:
        dataset_id = new_id("ds")
        path = self.dataset_file(dataset_id, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return dataset_id, path

    def save_json(self, path: Path, payload: dict) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def load_json(self, path: Path) -> dict:
        import json

        if not path.exists():
            from app.core.exceptions import NotFoundError

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
