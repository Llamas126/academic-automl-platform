class DomainError(Exception):
    status_code = 400
    detail = "Domain error"

    def __init__(self, detail: str | None = None):
        self.detail = detail or self.detail
        super().__init__(self.detail)


class NotFoundError(DomainError):
    status_code = 404
    detail = "Resource not found"


class ConflictError(DomainError):
    status_code = 409
    detail = "Conflicting state"


class ValidationFailedError(DomainError):
    status_code = 422
    detail = "Validation failed"


class UnsupportedFileError(DomainError):
    status_code = 415
    detail = "Unsupported file format"


class PayloadTooLargeError(DomainError):
    status_code = 413
    detail = "Payload too large"


class ExportError(DomainError):
    status_code = 500
    detail = "Export failed"


def register_exception_handlers(app) -> None:
    from fastapi.responses import JSONResponse

    @app.exception_handler(DomainError)
    async def domain_error_handler(_request, exc: DomainError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": type(exc).__name__, "detail": exc.detail},
        )
