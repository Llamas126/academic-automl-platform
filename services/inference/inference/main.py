"""HTTP entrypoint for the inference service."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from automl_shared.core.config import get_settings
from automl_shared.core.exceptions import register_exception_handlers
from automl_shared.core.logging import configure_logging


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    get_settings().ensure_dirs()
    yield


def create_app() -> FastAPI:
    from inference.api.v1.predict import router as predict_router

    settings = get_settings()
    configure_logging()
    app = FastAPI(
        title="Academic AutoML - Inference",
        version=settings.app_version,
        lifespan=lifespan,
    )
    register_exception_handlers(app)

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "inference"}

    app.include_router(predict_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
