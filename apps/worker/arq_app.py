"""arq worker configuration.

Replaces celery_app.py. Jobs are native async coroutines — no asyncio.run()
wrappers, no per-task event-loop churn, no stale connection pools.

Run the worker:
    python -m arq apps.worker.arq_app.WorkerSettings
"""

from __future__ import annotations

import traceback
from typing import ClassVar

import httpx
from arq import cron
from arq.connections import RedisSettings
from lyra_core.common.config import get_settings
from lyra_core.common.logging import configure_logging, get_logger

from .tasks.crystallize_skills import crystallize_skills
from .tasks.run_agent import resume_agent, run_agent

log = get_logger(__name__)

_settings = get_settings()


async def startup(ctx: dict) -> None:
    configure_logging(level=_settings.log_level, json_logs=_settings.is_prod)
    # Warm the Postgres pool on startup so the first job doesn't pay the
    # ~1-1.5s TLS handshake cost.
    from lyra_core.agent.checkpointer import _get_pool

    await _get_pool()
    log.info("arq.worker.startup")


async def shutdown(ctx: dict) -> None:
    from lyra_core.agent.checkpointer import _pool

    if _pool is not None:
        await _pool.close()
    log.info("arq.worker.shutdown")


async def on_job_abort(ctx: dict) -> None:
    exc: BaseException | None = ctx.get("exc")
    job_id: str = ctx.get("job_id", "unknown")
    func_name: str = ctx.get("func", "unknown")
    webhook = _settings.slack_error_webhook_url
    if not webhook or not exc:
        return
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-1500:]
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                webhook,
                json={
                    "text": "❌ *Worker job failed*",
                    "attachments": [
                        {
                            "color": "danger",
                            "fields": [
                                {"title": "Job ID", "value": job_id, "short": True},
                                {"title": "Function", "value": func_name, "short": True},
                                {"title": "Error", "value": f"`{type(exc).__name__}: {exc}`"},
                                {"title": "Traceback", "value": f"```{tb}```"},
                            ],
                        }
                    ],
                },
            )
    except Exception:
        log.warning("arq.worker.slack_error_notify_failed")


class WorkerSettings:
    functions: ClassVar[list] = [run_agent, resume_agent, crystallize_skills]
    cron_jobs: ClassVar[list] = [cron(crystallize_skills, hour=3, minute=0)]
    max_jobs = 4
    # 5 min per job — safe because LangGraph interrupt() causes ainvoke()
    # to return immediately (it does NOT block waiting for approval).
    job_timeout = 300
    # 20 slots for thread-lock contention retries (5s x 20 = 100s of patience)
    # + 5 slots for real transient failures. Total: 25.
    max_tries = 25
    on_startup = startup
    on_shutdown = shutdown
    on_job_abort = on_job_abort
    redis_settings = RedisSettings.from_dsn(_settings.celery_broker_url)
    keep_result = 3600
