from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import app.api.v1.router as api_router_module
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.core.security import register_security_middlewares


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    get_settings().ensure_dirs()
    yield


def _configure_cors(app: FastAPI, settings) -> None:
    """Enable CORS only for explicitly configured origins.

    Empty ``CORS_ORIGINS`` (the default) means same-origin only: no CORS
    headers are emitted at all. The wildcard must be opted into explicitly.
    """
    origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
    if not origins:
        return
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials="*" not in origins,
    )


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Async AutoML platform producing publication-ready artifacts",
        lifespan=lifespan,
    )
    _configure_cors(app, settings)
    register_exception_handlers(app)
    register_security_middlewares(app)
    app.include_router(api_router_module.api_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
