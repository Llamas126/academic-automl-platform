from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.validation import validate_resource_id
from app.services import dataset_service

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/analyze-dataset", response_model=None)
async def analyze_dataset(file: UploadFile):
    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    content = await dataset_service.read_upload_capped(file, max_bytes)
    filename = file.filename or "upload.csv"
    profile = dataset_service.ingest_dataset(filename, content)
    return JSONResponse(status_code=201, content=profile.model_dump(mode="json"))


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str):
    validate_resource_id(dataset_id, "dataset id")
    profile = dataset_service.get_profile(dataset_id)
    return profile.model_dump(mode="json")


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str):
    validate_resource_id(dataset_id, "dataset id")
    dataset_service.delete_dataset(dataset_id)
    return {"deleted": dataset_id}
