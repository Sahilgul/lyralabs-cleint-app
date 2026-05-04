"""arq tasks: run_agent + resume_agent.

`run_agent`:    invoked from the Slack/Teams webhook handler. Loads the
                tenant, builds the LangGraph, runs it. On `interrupt()`
                the graph's state is checkpointed and the task ends with
                status="awaiting_approval".

`resume_agent`: invoked from the Slack action handler when a user clicks
                Approve/Reject. Resumes the same thread_id with a Command.

Both tasks acquire a per-thread Redis lock before entering the graph to
prevent concurrent ainvoke() calls on the same thread_id from racing on
the LangGraph Postgres checkpoint.
"""

from __future__ import annotations

import time
from datetime import timedelta

from arq import Retry
from lyra_core.agent.checkpointer import checkpointer
from lyra_core.agent.graph import build_agent_graph
from lyra_core.channels.schema import InboundMessage, OutboundReply
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
from lyra_core.tools import google as _google  # noqa: F401
from lyra_core.tools import meta_tools as _meta_tools  # noqa: F401
from lyra_core.tools import slack as _slack  # noqa: F401
from lyra_core.tools.credentials import get_credentials
from lyra_core.tools.mcp_registry import discover_and_register_tools
from lyra_core.tools.registry import default_registry
from sqlalchemy import select, update

log = get_logger(__name__)

# Must match WorkerSettings.max_tries in arq_app.py.
_MAX_TRIES = 25

# Errors that are clearly permanent (bad data, code bugs). Don't retry these.
_NO_RETRY_TYPES = (
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    AssertionError,
    ImportError,
    NotImplementedError,
)


def _should_retry(exc: BaseException) -> bool:
    return not isinstance(exc, _NO_RETRY_TYPES)


def _is_interrupted(state: dict) -> bool:
    """Return True when LangGraph's interrupt() fired and ainvoke() returned early."""
    return bool(state.get("__interrupt__"))


async def _post_dlq_error(
    tenant_id: str | None,
    channel_id: str | None,
    thread_ts: str | None,
    job_id: str,
) -> None:
    """Best-effort: post a visible error to the Slack thread so the user isn't left in silence."""
    if not tenant_id or not channel_id:
        return
    try:
        from lyra_core.channels.slack.poster import post_reply

        await post_reply(
            tenant_id,
            OutboundReply(
                channel_id=channel_id,
                thread_ts=thread_ts,
                text=(
                    f"Something went wrong processing your request "
                    f"(ref: {job_id[:8]}). The team has been notified."
                ),
            ),
        )
    except Exception:
        log.warning("run_agent.dlq_post_failed", job_id=job_id)


async def run_agent(ctx: dict, message_json: str) -> dict:
    return await _run(ctx, message_json)


async def resume_agent(ctx: dict, *, job_id: str, decision: str, user_id: str) -> dict:
    return await _resume(ctx, job_id=job_id, decision=decision, user_id=user_id)


async def _run(ctx: dict, message_json: str) -> dict:
    msg = InboundMessage.model_validate_json(message_json)
    task_started = time.perf_counter()

    bind_job_context(
        thread_id=msg.agent_thread_id,
        tenant_external_id=msg.tenant_external_id,
        slack_channel=msg.channel_id,
        slack_user=msg.user_id,
    )

    tenant_id: str | None = None
    job_id: str | None = None

    try:
        async with phase("worker.tenant_lookup_and_job_insert"):
            async with async_session() as s:
                tenant = (
                    await s.execute(
                        select(Tenant).where(Tenant.external_team_id == msg.tenant_external_id)
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
                await s.flush()
                job_id = job.id
                await s.commit()

        bind_job_context(job_id=job_id, tenant_id=tenant_id)

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

        # Acquire a per-thread lock so concurrent messages on the same
        # thread_id don't race on the LangGraph Postgres checkpoint.
        lock_key = f"arlo:thread_lock:{thread_id}"
        acquired = await ctx["redis"].set(lock_key, job_id, nx=True, ex=300)
        if not acquired:
            # Another job is active on this thread. Retry without counting
            # against the real-failure budget. With max_tries=25, this gives
            # up to ~100s of patience (20 retries x 5s) before exhausting.
            log.info("run_agent.lock_wait", thread_id=thread_id, job_try=ctx.get("job_try"))
            raise Retry(defer=timedelta(seconds=5))

        try:
            async with phase("worker.checkpointer_open"):
                saver_cm = checkpointer()
                saver = await saver_cm.__aenter__()

            try:
                graph = build_agent_graph(saver)
                async with phase("worker.graph_invoke"):
                    final = await graph.ainvoke(initial_state, config=config)
            except Exception as exc:
                log.exception("run_agent.crash", error=str(exc))
                await _mark_job(job_id, status="failed", error=str(exc))
                is_final = ctx.get("job_try", 1) >= _MAX_TRIES
                if is_final or not _should_retry(exc):
                    await _post_dlq_error(tenant_id, msg.channel_id, msg.reply_thread_ts, job_id)
                    return {"status": "failed", "error": str(exc)}
                raise Retry(defer=timedelta(seconds=30)) from exc
            finally:
                await saver_cm.__aexit__(None, None, None)
        finally:
            await ctx["redis"].delete(lock_key)

        if _is_interrupted(final):
            async with phase("worker.mark_job_awaiting_approval"):
                await _mark_job(job_id, status="awaiting_approval")
            log.info(
                "run_agent.interrupted",
                duration_ms=int((time.perf_counter() - task_started) * 1000),
            )
            return {"status": "awaiting_approval", "job_id": job_id}

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


async def _resume(ctx: dict, *, job_id: str, decision: str, user_id: str) -> dict:
    from langgraph.types import Command

    bind_job_context(job_id=job_id, slack_user=user_id, decision=decision)
    task_started = time.perf_counter()

    tenant_id: str | None = None
    channel_id: str | None = None
    thread_ts: str | None = None

    try:
        async with phase("resume.lookup_and_record_approval"):
            async with async_session() as s:
                # Atomically claim the resume: only one concurrent click wins.
                # If two button clicks arrive in parallel, only one observes
                # status='awaiting_approval' and flips it to 'resuming'; the
                # loser sees 0 affected rows and bails. This closes the TOCTOU
                # window between the read and the write that a select-then-check
                # leaves open.
                claim = await s.execute(
                    update(Job)
                    .where(Job.id == job_id, Job.status == "awaiting_approval")
                    .values(status="resuming")
                    .returning(Job.id)
                )
                if claim.first() is None:
                    # Either the job doesn't exist, or another resume already
                    # claimed it. Look up status for an informative log line.
                    existing = (
                        await s.execute(select(Job).where(Job.id == job_id))
                    ).scalar_one_or_none()
                    if existing is None:
                        log.error("resume_agent.no_job")
                        return {"status": "no_job"}
                    log.info(
                        "resume_agent.already_processed",
                        job_status=existing.status,
                    )
                    return {"status": "already_processed", "job_id": job_id}

                # Claim succeeded — load the rest of the job fields we need.
                job = (await s.execute(select(Job).where(Job.id == job_id))).scalar_one()
                thread_id = job.thread_id
                tenant_id = job.tenant_id
                channel_id = job.channel_id
                thread_ts = job.parent_message_ts

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

        # Same lock as run_agent — serializes approval against any concurrent
        # run_agent that might be starting on the same thread.
        lock_key = f"arlo:thread_lock:{thread_id}"
        acquired = await ctx["redis"].set(lock_key, job_id, nx=True, ex=300)
        if not acquired:
            log.info("resume_agent.lock_wait", thread_id=thread_id)
            raise Retry(defer=timedelta(seconds=5))

        try:
            async with phase("resume.checkpointer_open"):
                saver_cm = checkpointer()
                saver = await saver_cm.__aenter__()

            try:
                graph = build_agent_graph(saver)
                async with phase("resume.graph_invoke"):
                    final = await graph.ainvoke(
                        Command(resume={"decision": decision}),
                        config=config,
                    )
            except Exception as exc:
                log.exception("resume_agent.crash", error=str(exc))
                await _mark_job(job_id, status="failed", error=str(exc))
                is_final = ctx.get("job_try", 1) >= _MAX_TRIES
                if is_final or not _should_retry(exc):
                    await _post_dlq_error(tenant_id, channel_id, thread_ts, job_id)
                    return {"status": "failed", "error": str(exc)}
                raise Retry(defer=timedelta(seconds=30)) from exc
            finally:
                await saver_cm.__aexit__(None, None, None)
        finally:
            await ctx["redis"].delete(lock_key)

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
