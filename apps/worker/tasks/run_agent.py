"""Celery tasks: run_agent + resume_agent.

`run_agent`:    invoked from the Slack/Teams webhook handler. Loads the
                tenant, builds the LangGraph, runs it. On `interrupt()`
                the graph's state is checkpointed and the task ends.

`resume_agent`: invoked from the Slack action handler when a user clicks
                Approve/Reject. Resumes the same thread_id with a Command.
"""

from __future__ import annotations

import asyncio

from sqlalchemy import select

from lyra_core.agent.checkpointer import checkpointer
from lyra_core.agent.graph import build_agent_graph
from lyra_core.channels.schema import InboundMessage, Surface
from lyra_core.common.audit import record_event
from lyra_core.common.logging import get_logger
from lyra_core.db.models import Job, Tenant
from lyra_core.db.session import async_session

# Importing the tools modules triggers their registration in the registry.
from lyra_core.tools import artifacts as _artifacts  # noqa: F401
from lyra_core.tools import ghl as _ghl  # noqa: F401
from lyra_core.tools import google as _google  # noqa: F401

from ..celery_app import celery

log = get_logger(__name__)


@celery.task(bind=True, name="apps.worker.tasks.run_agent.run_agent")
def run_agent(self, message_json: str) -> dict:  # noqa: ANN001, ARG001
    return asyncio.run(_run(message_json))


@celery.task(bind=True, name="apps.worker.tasks.run_agent.resume_agent")
def resume_agent(self, *, job_id: str, decision: str, user_id: str) -> dict:  # noqa: ANN001, ARG001
    return asyncio.run(_resume(job_id=job_id, decision=decision, user_id=user_id))


async def _resolve_tenant(external_id: str, channel: Surface) -> Tenant | None:
    async with async_session() as s:
        return (
            await s.execute(select(Tenant).where(Tenant.external_team_id == external_id))
        ).scalar_one_or_none()


async def _run(message_json: str) -> dict:
    msg = InboundMessage.model_validate_json(message_json)
    tenant = await _resolve_tenant(msg.tenant_external_id, msg.surface)
    if tenant is None:
        log.error("run_agent.no_tenant", external_id=msg.tenant_external_id)
        return {"status": "no_tenant"}

    if tenant.status not in {"active"}:
        log.warning("run_agent.tenant_not_active", tenant_id=tenant.id, status=tenant.status)
        return {"status": "tenant_inactive"}

    # Persist a Job row to track this invocation.
    async with async_session() as s:
        job = Job(
            tenant_id=tenant.id,
            thread_id=f"{msg.surface}:{msg.channel_id}:{msg.thread_id}",
            user_id=msg.user_id,
            channel_id=msg.channel_id,
            # Stored for audit/debug only -- mirror the Slack thread_ts the
            # bot will reply into (None for top-level DMs / channel posts).
            parent_message_ts=msg.reply_thread_ts,
            user_request=msg.text,
            status="running",
        )
        s.add(job)
        await s.commit()
        await s.refresh(job)
        job_id = job.id
        thread_id = job.thread_id

    initial_state = {
        "tenant_id": tenant.id,
        "job_id": job_id,
        "channel_id": msg.channel_id,
        "thread_id": msg.thread_id,
        "reply_thread_ts": msg.reply_thread_ts,
        "user_id": msg.user_id,
        "user_request": msg.text,
        "step_results": [],
        "artifacts": [],
        "total_cost_usd": 0.0,
    }

    config = {"configurable": {"thread_id": thread_id}}

    async with checkpointer() as saver:
        graph = build_agent_graph(saver)
        try:
            final = await graph.ainvoke(initial_state, config=config)
        except Exception as exc:  # noqa: BLE001
            log.exception("run_agent.crash", job_id=job_id, error=str(exc))
            await _mark_job(job_id, status="failed", error=str(exc))
            return {"status": "failed", "error": str(exc)}

    # If we got here without an interrupt, the graph completed.
    await _mark_job(
        job_id,
        status="done",
        result_summary=final.get("final_summary"),
        cost_usd=float(final.get("total_cost_usd", 0.0)),
    )
    return {"status": "done", "job_id": job_id}


async def _resume(*, job_id: str, decision: str, user_id: str) -> dict:
    from langgraph.types import Command

    async with async_session() as s:
        job = (await s.execute(select(Job).where(Job.id == job_id))).scalar_one_or_none()
        if job is None:
            log.error("resume_agent.no_job", job_id=job_id)
            return {"status": "no_job"}
        thread_id = job.thread_id
        tenant_id = job.tenant_id

        await record_event(
            s,
            tenant_id=tenant_id,
            actor_user_id=user_id,
            job_id=job_id,
            event_type="approval",
            extra={"decision": decision},
        )
        await s.commit()

    config = {"configurable": {"thread_id": thread_id}}

    async with checkpointer() as saver:
        graph = build_agent_graph(saver)
        try:
            final = await graph.ainvoke(
                Command(resume={"decision": decision}), config=config  # type: ignore[arg-type]
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("resume_agent.crash", job_id=job_id, error=str(exc))
            await _mark_job(job_id, status="failed", error=str(exc))
            return {"status": "failed", "error": str(exc)}

    status = "done" if decision == "approved" else "rejected"
    await _mark_job(
        job_id,
        status=status,
        result_summary=final.get("final_summary"),
        cost_usd=float(final.get("total_cost_usd", 0.0)),
    )
    return {"status": status, "job_id": job_id}


async def _mark_job(
    job_id: str,
    *,
    status: str,
    result_summary: str | None = None,
    error: str | None = None,
    cost_usd: float | None = None,
) -> None:
    async with async_session() as s:
        job = (await s.execute(select(Job).where(Job.id == job_id))).scalar_one_or_none()
        if job is None:
            return
        job.status = status
        if result_summary is not None:
            job.result_summary = result_summary
        if error is not None:
            job.error = error
        if cost_usd is not None:
            job.cost_usd = cost_usd
        await s.commit()
