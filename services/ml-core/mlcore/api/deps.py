from automl_shared.core.config import get_settings
from automl_shared.jobs.job_store import get_job_store


def settings():
    return get_settings()


def job_store():
    return get_job_store()
