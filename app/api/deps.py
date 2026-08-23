from app.core.config import get_settings
from app.services.job_service import get_job_store


def settings():
    return get_settings()


def job_store():
    return get_job_store()
