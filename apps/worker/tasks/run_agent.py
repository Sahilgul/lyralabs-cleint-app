"""Celery tasks: run_agent + resume_agent.

`run_agent`:    invoked from the Slack/Teams webhook handler. Loads the
                tenant, builds the LangGraph, runs it. On `interrupt()`
                the graph's state is checkpointed and the task ends.

`resume_agent`: invoked from the Slack action handler when a user clicks
                Approve/Reject. Resumes the same thread_id with a Command.
"""

from __future__ import annotations

import asyncio
import time

from lyra_core.agent.checkpointer import checkpointer
from lyra_core.agent.graph import build_agent_graph
from lyra_core.channels.schema import InboundMessage
from lyra_core.common.audit import record_event
from lyra_core.common.logging import (
    bind_job_context,
    clear_job_context,
    get_logger,
    phase,
)
from lyra_core.db.models import Job, Tenant
from lyra_core.db.session import async_session

# Importing tools triggers registration in the global registry.
from lyra_core.tools import artifacts as _artifacts  # noqa: F401
from lyra_core.tools import ghl as _ghl  # noqa: F401
from lyra_core.tools import google as _google  # noqa: F401
from lyra_core.tools import meta_tools as _meta_tools  # noqa: F401
from lyra_core.tools import slack as _slack  # noqa: F401
from lyra_core.tools.credentials import get_credentials
from lyra_core.tools.mcp_registry import discover_and_register_tools
from lyra_core.tools.registry import default_registry
from sqlalchemy import select

from ..celery_app import celery

log = get_logger(__name__)


@celery.task(bind=True, name="apps.worker.tasks.run_agent.run_agent")
def run_agent(self, message_json: str) -> dict:
    return asyncio.run(_run(message_json))


@celery.task(bind=True, name="apps.worker.tasks.run_agent.resume_agent")
def resume_agent(self, *, job_id: str, decision: str, user_id: str) -> dict:
    return asyncio.run(_resume(job_id=job_id, decision=decision, user_id=user_id))


async def _run(message_json: str) -> dict:
    msg = InboundMessage.model_validate_json(message_json)
    task_started = time.perf_counter()

    # Bind context the moment we know the tenant/thread/user so EVERY
    # subsequent log line (including from deep modules like litellm) is
    # greppable by job_id once we have one.
    bind_job_context(
        thread_id=msg.agent_thread_id,
        tenant_external_id=msg.tenant_external_id,
        slack_channel=msg.channel_id,
        slack_user=msg.user_id,
    )

    try:
        # One session for tenant lookup + job insert. Each round-trip to
        # Supabase Tokyo from us-east1 is ~200ms; pipelining shaves
        # ~300-400ms vs. the previous separate-session pattern.
        async with phase("worker.tenant_lookup_and_job_insert"):
            async with async_session() as s:
                tenant = (
                    await s.execute(
                        select(Tenant).where(
                            Tenant.external_team_id == msg.tenant_external_id
                        )
                    )
                ).scalar_one_or_none()
                if tenant is None:
                    log.error("run_agent.no_tenant", external_id=msg.tenant_external_id)
                    return {"status": "no_tenant"}
                if tenant.status != "active":
                    log.warning(
                        "run_agent.tenant_not_active",
                        tenant_id=tenant.id,
                        status=tenant.status,
                    )
                    return {"status": "tenant_inactive"}

                tenant_id = tenant.id
                # Use the adapter-provided agent_thread_id as the LangGraph
                # checkpointer key. For DMs that's a stable per-(team, channel,
                # user) key so the agent's memory persists across top-level
                # user messages -- the previous key was per-message-ts and
                # reset every turn, which is why ARLO appeared to forget
                # the user's name mid-conversation.
                thread_id = msg.agent_thread_id
                job = Job(
                    tenant_id=tenant_id,
                    client_id=msg.client_id,
                    thread_id=thread_id,
                    user_id=msg.user_id,
                    channel_id=msg.channel_id,
                    parent_message_ts=msg.reply_thread_ts,
                    user_request=msg.text,
                    status="running",
                )
                s.add(job)
                await s.flush()  # populates job.id without an extra refresh
                job_id = job.id
                await s.commit()

        # Extend the bound context with job_id now that we have one.
        bind_job_context(job_id=job_id, tenant_id=tenant_id)

        # Discover MCP tools for this tenant/client and build auth headers.
        # Best-effort: missing creds or network errors skip MCP for this job.
        async with phase("worker.mcp_discovery"):
            try:
                ghl_creds = await get_credentials(tenant_id, "ghl", msg.client_id)
                ghl_headers = {
                    "Authorization": f"Bearer {ghl_creds.access_token}",
                    "locationId": ghl_creds.external_account_id,
                }
                await discover_and_register_tools(
                    "ghl", tenant_id, msg.client_id, ghl_headers, default_registry
                )
            except Exception as exc:
                log.info("worker.mcp_discovery.skipped", reason=str(exc))

        # Load the living artifact and active skills for this thread.
        from lyra_core.agent.living_artifact import load_artifact
        from lyra_core.agent.skill_crystallizer import load_active_skills

        async with phase("worker.load_context"):
            try:
                living_artifact = await load_artifact(tenant_id, msg.client_id, msg.agent_thread_id)
                active_skills = await load_active_skills(tenant_id, msg.client_id)
            except Exception as exc:
                log.info("worker.load_context.skipped", reason=str(exc))
                living_artifact = {}
                active_skills = []

        initial_state = {
            "tenant_id": tenant_id,
            "client_id": msg.client_id,
            "job_id": job_id,
            "channel_id": msg.channel_id,
            "thread_id": msg.thread_id,
            "reply_thread_ts": msg.reply_thread_ts,
            "assistant_status_thread_ts": msg.thread_id if msg.is_dm else None,
            "user_id": msg.user_id,
            "user_request": msg.text,
            "step_results": [],
            "artifacts": [],
            "total_cost_usd": 0.0,
            "living_artifact": living_artifact,
            "active_skills": active_skills,
        }

        config = {"configurable": {"thread_id": thread_id}}

        async with phase("worker.checkpointer_open"):
            saver_cm = checkpointer()
            saver = await saver_cm.__aenter__()

        try:
            graph = build_agent_graph(saver)
            try:
                async with phase("worker.graph_invoke"):
                    final = await graph.ainvoke(initial_state, config=config)
            except Exception as exc:
                log.exception("run_agent.crash", error=str(exc))
                await _mark_job(job_id, status="failed", error=str(exc))
                return {"status": "failed", "error": str(exc)}
        finally:
            await saver_cm.__aexit__(None, None, None)

        # If we got here without an interrupt, the graph completed.
        async with phase("worker.mark_job_done"):
            await _mark_job(
                job_id,
                status="done",
                result_summary=final.get("final_summary"),
                cost_usd=float(final.get("total_cost_usd", 0.0)),
            )

        log.info(
            "run_agent.task_total",
            duration_ms=int((time.perf_counter() - task_started) * 1000),
            cost_usd=float(final.get("total_cost_usd", 0.0)),
        )
        return {"status": "done", "job_id": job_id}
    finally:
        clear_job_context()


async def _resume(*, job_id: str, decision: str, user_id: str) -> dict:
    from langgraph.types import Command

    bind_job_context(job_id=job_id, slack_user=user_id, decision=decision)
    task_started = time.perf_counter()
    try:
        async with phase("resume.lookup_and_record_approval"):
            async with async_session() as s:
                job = (
                    await s.execute(select(Job).where(Job.id == job_id))
                ).scalar_one_or_none()
                if job is None:
                    log.error("resume_agent.no_job")
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

        bind_job_context(thread_id=thread_id, tenant_id=tenant_id)
        config = {"configurable": {"thread_id": thread_id}}

        async with phase("resume.checkpointer_open"):
            saver_cm = checkpointer()
            saver = await saver_cm.__aenter__()

        try:
            graph = build_agent_graph(saver)
            try:
                async with phase("resume.graph_invoke"):
                    final = await graph.ainvoke(
                        Command(resume={"decision": decision}),  # type: ignore[arg-type]
                        config=config,
                    )
            except Exception as exc:
                log.exception("resume_agent.crash", error=str(exc))
                await _mark_job(job_id, status="failed", error=str(exc))
                return {"status": "failed", "error": str(exc)}
        finally:
            await saver_cm.__aexit__(None, None, None)

        status = "done" if decision == "approved" else "rejected"
        async with phase("resume.mark_job_final"):
            await _mark_job(
                job_id,
                status=status,
                result_summary=final.get("final_summary"),
                cost_usd=float(final.get("total_cost_usd", 0.0)),
            )

        log.info(
            "resume_agent.task_total",
            duration_ms=int((time.perf_counter() - task_started) * 1000),
            cost_usd=float(final.get("total_cost_usd", 0.0)),
        )
        return {"status": status, "job_id": job_id}
    finally:
        clear_job_context()


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
