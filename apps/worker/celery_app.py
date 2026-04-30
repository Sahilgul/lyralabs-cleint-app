"""Celery app instance.

Tasks live in `apps/worker/tasks/`. Eager mode is auto-enabled when
APP_ENV=test for fast unit tests.
"""

from __future__ import annotations

from celery import Celery

from lyra_core.common.config import get_settings
from lyra_core.common.logging import configure_logging

settings = get_settings()
configure_logging(level=settings.log_level, json_logs=settings.is_prod)

celery = Celery(
    "lyralabs",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["apps.worker.tasks.run_agent"],
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_default_queue="default",
    task_routes={
        "apps.worker.tasks.run_agent.run_agent": {"queue": "agent"},
        "apps.worker.tasks.run_agent.resume_agent": {"queue": "agent"},
    },
    broker_transport_options={"visibility_timeout": 3600},
)

if settings.app_env == "test":
    celery.conf.task_always_eager = True
    celery.conf.task_eager_propagates = True
