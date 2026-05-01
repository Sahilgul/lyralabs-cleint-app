"""Slack adapter: builds the slack_bolt App, wires events to the agent.

Inbound flow:
  Slack event -> bolt request handler -> normalize to InboundMessage
              -> enqueue Celery task `run_agent` -> ack within 3s

Outbound flow:
  Worker calls `post_reply()` with an OutboundReply -> Slack Web API
"""

from __future__ import annotations

from typing import Any

from slack_bolt.adapter.starlette.async_handler import AsyncSlackRequestHandler
from slack_bolt.async_app import AsyncApp
from slack_bolt.oauth.async_oauth_settings import AsyncOAuthSettings
from slack_bolt.response import BoltResponse
from slack_sdk.errors import SlackApiError

from ...common.config import get_settings
from ...common.logging import get_logger
from ..schema import InboundMessage, Surface
from .install_store import PostgresInstallationStore

log = get_logger(__name__)


def build_slack_app() -> tuple[AsyncApp, AsyncSlackRequestHandler]:
    """Construct the Bolt app + a Starlette/FastAPI request handler.

    If SLACK_* env vars are not set (e.g. local boot without secrets) we
    construct a minimal app so the FastAPI process can still come up; OAuth
    routes will return 500 until creds are configured.
    """
    settings = get_settings()

    has_oauth = bool(
        settings.slack_signing_secret
        and settings.slack_client_id
        and settings.slack_client_secret
    )

    if has_oauth:
        oauth_settings = AsyncOAuthSettings(
            client_id=settings.slack_client_id,
            client_secret=settings.slack_client_secret,
            scopes=settings.slack_scopes_list,
            user_scopes=(
                settings.slack_user_scopes.split(",") if settings.slack_user_scopes else []
            ),
            redirect_uri=settings.slack_install_redirect_url or None,
            install_path="/oauth/slack/install",
            redirect_uri_path="/oauth/slack/callback",
            installation_store=PostgresInstallationStore(),
        )
        app = AsyncApp(
            signing_secret=settings.slack_signing_secret,
            oauth_settings=oauth_settings,
            # process_before_response=False (default) is correct for Cloud Run /
            # any long-running ASGI server: Bolt acks Slack within ~50ms and
            # runs handlers in a background asyncio task. With True, every
            # event waited for `authorize` (DB lookup + token refresh) +
            # handler before responding -- our /slack/events latency was
            # ~2.4s on Cloud Run, dangerously close to Slack's 3s retry
            # threshold, which produced a retry storm of duplicate run_agent
            # tasks for a single user message.
            process_before_response=False,
        )
    else:
        log.warning(
            "slack.config.missing",
            reason="SLACK_CLIENT_ID/SECRET/SIGNING_SECRET not set; running stub Slack app",
        )
        app = AsyncApp(
            signing_secret=settings.slack_signing_secret or "missing",
            token="xoxb-missing",
            process_before_response=False,
        )

    @app.middleware
    async def _drop_slack_retries(request, next_):  # type: ignore[no-untyped-def]  # noqa: ANN001
        """Short-circuit Slack's automatic event retries.

        Slack retries an event up to 3x if it doesn't see a 2xx within 3s.
        Even with process_before_response=False we still get retries when
        Cloud Run cold-starts or under transient slowness. Without
        deduplication, every retry enqueues another `run_agent` task for
        the same user message -- that's why we previously saw 4 task IDs
        for a single DM, all replying into the same Slack thread.
        Acking 200 here on retry headers tells Slack we got it, and we
        don't kick off duplicate work.
        """
        retry_num = request.headers.get("x-slack-retry-num")
        if retry_num:
            log.info(
                "slack.retry.skipped",
                retry_num=retry_num if isinstance(retry_num, str) else retry_num[0],
                reason=request.headers.get("x-slack-retry-reason"),
            )
            # MUST return a BoltResponse, not bare `return`. Without an
            # explicit response, slack_bolt's listener loop ends with no
            # match and the Starlette handler emits 404. Slack treats 404
            # as a delivery failure and retries again, producing a noisy
            # loop of 404s in Cloud Run logs (the handler is still
            # short-circuited so no duplicate work happens, but Slack
            # eventually marks the endpoint as flaky).
            return BoltResponse(status=200, body="")
        await next_()

    @app.event("app_mention")
    async def on_mention(body: dict[str, Any], ack: Any, say: Any, client: Any) -> None:
        await ack()
        await _enqueue_from_event(body, client)

    @app.event("message")
    async def on_message(body: dict[str, Any], ack: Any, client: Any) -> None:
        await ack()
        event = body.get("event", {})
        # Only respond in DMs or threads we're in; ignore bot messages.
        if event.get("bot_id") or event.get("subtype"):
            return
        if event.get("channel_type") == "im":
            await _enqueue_from_event(body, client)

    @app.command("/arlo")
    async def slash_command(ack: Any, command: dict[str, Any], client: Any) -> None:
        await ack(f"On it: _{command.get('text', '')}_")
        # Enqueue a synthetic event mirroring the slash command.
        synthetic = {
            "team_id": command.get("team_id"),
            "event": {
                "type": "message",
                "channel": command.get("channel_id"),
                "user": command.get("user_id"),
                "text": command.get("text", ""),
                "ts": command.get("trigger_id"),
                "thread_ts": None,
            },
        }
        await _enqueue_from_event(synthetic, client)

    @app.event("app_uninstalled")
    async def on_uninstalled(ack: Any, body: dict[str, Any]) -> None:
        await ack()
        team_id = body.get("team_id")
        log.info("slack.uninstalled", team_id=team_id)
        await _disable_workspace(team_id)

    @app.event("tokens_revoked")
    async def on_tokens_revoked(ack: Any, body: dict[str, Any]) -> None:
        await ack()
        team_id = body.get("team_id")
        log.info("slack.tokens_revoked", team_id=team_id)
        await _disable_workspace(team_id)

    @app.action("approval")
    async def on_approval_action(ack: Any, body: dict[str, Any]) -> None:
        """User clicked Approve/Reject button on a preview card."""
        await ack()
        from apps.worker.tasks.run_agent import resume_agent  # noqa: PLC0415  - avoid cycle

        action = body["actions"][0]
        value = action.get("value", "")  # "approve:<job_id>" | "reject:<job_id>"
        decision, _, job_id = value.partition(":")
        resume_agent.delay(job_id=job_id, decision=decision, user_id=body["user"]["id"])

    return app, AsyncSlackRequestHandler(app)


async def _disable_workspace(team_id: str | None) -> None:
    """Mark the tenant inactive and zero out the bot token on uninstall/revoke."""
    if not team_id:
        return
    from sqlalchemy import select, update

    from ...db.models import SlackInstallation, Tenant
    from ...db.session import async_session

    async with async_session() as s:
        tenant = (
            await s.execute(select(Tenant).where(Tenant.external_team_id == team_id))
        ).scalar_one_or_none()
        if tenant is None:
            return
        tenant.status = "cancelled"
        await s.execute(
            update(SlackInstallation)
            .where(SlackInstallation.tenant_id == tenant.id)
            .values(bot_token_encrypted=None, bot_refresh_token_encrypted=None)
        )
        await s.commit()


async def _enqueue_from_event(body: dict[str, Any], client: Any = None) -> None:
    """Translate a Slack event into InboundMessage + dispatch to Celery.

    Threading rules (so the DM UX stays linear and channel mentions stay tidy):
      - DM, top-level message  -> reply as a new top-level message (no thread_ts).
      - DM, reply inside thread -> reply in that same thread.
      - Channel @-mention top-level -> reply threaded on the user's message.
      - Channel reply inside thread -> reply in that same thread.

    The agent's checkpointer key (`thread_id`) is independent: we use
    thread_ts when present, else the message ts, so each new top-level
    user message starts a fresh agent conversation rather than leaking
    state from the previous one.

    If `client` is provided we also fire a "Thinking…" status indicator
    so the user sees feedback within ~100ms instead of waiting for the
    worker to post the real reply (~3-5s).
    """
    from apps.worker.tasks.run_agent import run_agent  # noqa: PLC0415

    event = body.get("event") or {}
    is_dm = event.get("channel_type") == "im"
    slack_thread_ts: str | None = event.get("thread_ts")
    slack_msg_ts: str | None = event.get("ts")

    if slack_thread_ts:
        reply_thread_ts: str | None = slack_thread_ts
    elif is_dm:
        reply_thread_ts = None
    else:
        reply_thread_ts = slack_msg_ts

    msg = InboundMessage(
        surface=Surface.SLACK,
        tenant_external_id=body.get("team_id") or body.get("api_app_id") or "",
        channel_id=event.get("channel", ""),
        thread_id=slack_thread_ts or slack_msg_ts or "",
        user_id=event.get("user", ""),
        text=(event.get("text") or "").strip(),
        files=event.get("files", []),
        reply_thread_ts=reply_thread_ts,
        is_dm=is_dm,
        raw=body,
    )
    if not msg.text:
        return
    log.info(
        "slack.event.enqueue",
        tenant=msg.tenant_external_id,
        thread=msg.thread_id,
        reply_thread=reply_thread_ts or "(top-level)",
        is_dm=is_dm,
        text_len=len(msg.text),
    )

    # Fire feedback within ~100ms so the user knows ARLO heard them.
    # DMs: native Slack assistant typing indicator (clears automatically
    # when the bot posts a message). Channel mentions: an :eyes: reaction
    # (no native indicator exists outside assistant threads).
    if client is not None:
        await _post_thinking_indicator(client, msg, is_dm)

    run_agent.delay(message_json=msg.model_dump_json())


async def _post_thinking_indicator(client: Any, msg: InboundMessage, is_dm: bool) -> None:
    """Best-effort feedback. Never raises -- agent must run regardless."""
    try:
        if is_dm:
            await client.assistant_threads_setStatus(
                channel_id=msg.channel_id,
                thread_ts=msg.thread_id,
                status="Thinking…",
            )
        else:
            await client.reactions_add(
                channel=msg.channel_id,
                timestamp=msg.thread_id,
                name="eyes",
            )
    except SlackApiError as e:
        # Common reasons: app missing the assistant feature / scope, the
        # thread isn't an assistant thread, the reaction already exists.
        # All non-fatal -- the agent still runs and posts a real reply.
        log.info(
            "slack.indicator.skipped",
            kind="status" if is_dm else "reaction",
            error=getattr(e.response, "data", {}).get("error", str(e)),
        )
