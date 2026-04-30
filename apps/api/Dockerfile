FROM python:3.14-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# WeasyPrint native deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libgdk-pixbuf-2.0-0 \
    libffi-dev libssl-dev fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install -e .

COPY packages /app/packages
COPY apps /app/apps
COPY alembic.ini /app/alembic.ini

ENV PYTHONPATH=/app:/app/packages

# Cloud Run injects $PORT (defaults to 8080). Locally we run on 8000.
# Shell form lets us read $PORT; `exec` makes uvicorn PID 1 so SIGTERM is handled.
EXPOSE 8080
CMD exec uvicorn apps.api.main:app --host 0.0.0.0 --port ${PORT:-8080}
