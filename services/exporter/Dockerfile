FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ARG TECTONIC_VERSION=0.15.0
RUN curl -fsSL "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40${TECTONIC_VERSION}/tectonic-${TECTONIC_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
        | tar -xz -C /usr/local/bin \
    && tectonic --version

WORKDIR /app

RUN useradd --create-home appuser

COPY --chown=appuser:appuser requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser packages/shared ./packages/shared
COPY --chown=appuser:appuser services/exporter ./services/exporter
RUN pip install --no-cache-dir --no-deps \
        ./packages/shared \
        ./services/exporter

USER appuser

CMD ["celery", "-A", "exporter.worker.celery_app.celery_app", "worker", "--loglevel=INFO", "--concurrency=2", "-Q", "export.package"]
