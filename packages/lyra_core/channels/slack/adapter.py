"""Slack adapter: builds the slack_bolt App, wires events to the agent.

Inbound flow:
  Slack event -> bolt request handler -> normalize to InboundMessage
              -> enqueue Celery task `run_agent` -> ack within 3s

Outbound flow:
  Worker calls `post_reply()` with an OutboundReply -> Slack Web API
"""

from __future__ import annotations

import time
from typing import Any

from slack_bolt.adapter.starlette.async_handler import AsyncSlackRequestHandler
from slack_bolt.async_app import AsyncApp
from slack_bolt.oauth.async_oauth_flow import AsyncOAuthFlow
from slack_bolt.oauth.async_oauth_settings import AsyncOAuthSettings
from slack_bolt.request.async_request import AsyncBoltRequest
from slack_bolt.response import BoltResponse
from slack_sdk.errors import SlackApiError

from ...common.config import get_settings
from ...common.logging import get_logger
from ..schema import InboundMessage, Surface
from .install_store import PostgresInstallationStore, _state_to_tenant

log = get_logger(__name__)


class _TenantAwareOAuthFlow(AsyncOAuthFlow):
    """Extends the default OAuth flow to capture tenant_id from a signed `sig`
    query param during installation and store it for the callback to retrieve."""

    async def issue_new_state(self, request: AsyncBoltRequest) -> str:
        state = await super().issue_new_state(request)
        sig_values = request.query.get("sig")
        sig = sig_values[0] if sig_values else None
        if sig:
            try:
                from apps.api.oauth._state import decode_state  # noqa: PLC0415
                tenant_id, _ = decode_state(sig)
                _state_to_tenant[state] = tenant_id
            except Exception:
                pass
        return state


def build_slack_app() -> tuple[AsyncApp, AsyncSlackRequestHandler]:
    """Construct the Bolt app + a Starlette/FastAPI request handler.

    Used by the HTTPS webhook (POST /slack/events on Cloud Run). For
    Socket Mode the runner builds its own variant via `build_socket_mode_app`.
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
            oauth_flow=_TenantAwareOAuthFlow(settings=oauth_settings),
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

    _register_http_retry_middleware(app)
    _register_event_handlers(app)
    return app, AsyncSlackRequestHandler(app)


def build_socket_mode_app() -> AsyncApp:
    """Build a Bolt app wired for Socket Mode.

    Differences from `build_slack_app`:
      - No OAuthSettings -- OAuth still runs over HTTPS via build_slack_app
        on the API service. The Socket Mode app is INBOUND-ONLY and uses
        the multi-tenant `PostgresInstallationStore` directly to authorize
        each event.
      - No HTTP retry middleware -- Socket Mode doesn't use Slack's
        3-second-retry mechanism (events are delivered over a persistent
        WebSocket; there's no HTTP status to fail on).
      - Caller is responsible for wrapping this in `AsyncSocketModeHandler`
        and passing it the App-Level Token.
    """
    settings = get_settings()
    if not settings.slack_signing_secret:
        log.warning(
            "slack.socket.signing_secret_missing",
            reason="continuing with placeholder; signature verification disabled",
        )
    app = AsyncApp(
        signing_secret=settings.slack_signing_secret or "missing",
        installation_store=PostgresInstallationStore(),
        process_before_response=False,
    )
    _register_event_handlers(app)
    return app


def _register_http_retry_middleware(app: AsyncApp) -> None:
    """Short-circuit Slack's automatic event retries on the HTTPS path.

    Slack retries an event up to 3x if it doesn't see a 2xx within 3s.
    Even with process_before_response=False we still get retries when
    Cloud Run cold-starts or under transient slowness. Without
    deduplication, every retry enqueues another `run_agent` task for
    the same user message -- that's why we previously saw 4 task IDs
    for a single DM, all replying into the same Slack thread.

    Socket Mode has no such retry mechanism, so this middleware is HTTP-only.
    """

    @app.middleware
    async def _drop_slack_retries(request, next_):  # type: ignore[no-untyped-def]  # noqa: ANN001
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


def _register_event_handlers(app: AsyncApp) -> None:
    """Register every inbound Slack event/action handler on `app`.

    Shared between the HTTP webhook and the Socket Mode runner so the two
    transports stay behaviorally identical -- DMs, mentions, slash commands,
    approval clicks, and uninstalls all fan into the same Celery tasks.
    """

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
        from lyra_core.worker.queue import enqueue_resume_agent  # noqa: PLC0415

        action = body["actions"][0]
        value = action.get("value", "")  # "approve:<job_id>" | "reject:<job_id>"
        decision, _, job_id = value.partition(":")
        await enqueue_resume_agent(
            job_id=job_id, decision=decision, user_id=body["user"]["id"]
        )


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


async def _resolve_client_id(team_id: str, channel_id: str) -> str | None:
    """Return the Client.id whose primary_slack_channel_id matches channel_id.

    Joins tenants → clients using the unique index on (tenant_id, primary_slack_channel_id),
    so this is O(1). Returns None for DMs or channels not mapped to any client.
    """
    from sqlalchemy import select

    from ...db.models import Client, Tenant
    from ...db.session import async_session

    async with async_session() as s:
        row = (
            await s.execute(
                select(Client.id)
                .join(Tenant, Tenant.id == Client.tenant_id)
                .where(
                    Tenant.external_team_id == team_id,
                    Client.primary_slack_channel_id == channel_id,
                    Client.status == "active",
                )
                .limit(1)
            )
        ).scalar_one_or_none()
    return row


async def _enqueue_from_event(body: dict[str, Any], client: Any = None) -> None:
    """Translate a Slack event into InboundMessage + dispatch to Celery.

    Reply-threading rules (so the UX matches Slack conventions):
      - DM, top-level message     -> reply as a new top-level message (no thread_ts).
      - DM, reply inside thread   -> reply in that same thread.
      - Channel @-mention top-lvl -> reply threaded on the user's message.
      - Channel reply in a thread -> reply in that same thread.

    Agent-memory rules (the `agent_thread_id` keying the LangGraph checkpointer):
      - DM   -> 'slack:dm:{team}:{channel}:{user}' -- the entire DM with this
                user is ONE continuous agent conversation. Top-level DM
                messages share memory (e.g. user says "my name is sahil"
                in one message, then asks "what's my name?" in the next
                top-level message; ARLO remembers).
      - Thread / channel @-mention -> 'slack:ch:{team}:{channel}:{thread_ts}'
                where thread_ts is the Slack thread root ts (or the message
                ts itself for a fresh top-level mention). One thread = one
                conversation. New threads are independent.

    This is intentionally independent of `reply_thread_ts`: how the bot
    threads its replies is a UX choice, but how the agent remembers
    context should not be coupled to it.

    If `client` is provided we also fire a "Thinking…" status indicator
    so the user sees feedback within ~100ms instead of waiting for the
    worker to post the real reply (~3-5s).
    """
    from lyra_core.worker.queue import enqueue_run_agent  # noqa: PLC0415

    event = body.get("event") or {}
    is_dm = event.get("channel_type") == "im"
    slack_thread_ts: str | None = event.get("thread_ts")
    slack_msg_ts: str | None = event.get("ts")
    team_id: str = body.get("team_id") or body.get("api_app_id") or ""
    channel_id: str = event.get("channel", "")
    user_id: str = event.get("user", "")

    # O(1) via unique index on (tenant_id, primary_slack_channel_id).
    # DMs and unmapped channels return None (agency-internal job).
    client_id: str | None = None
    if not is_dm and channel_id and team_id:
        try:
            client_id = await _resolve_client_id(team_id, channel_id)
        except Exception:
            log.warning("slack.client_resolve.error", team=team_id, channel=channel_id, exc_info=True)

    if slack_thread_ts:
        reply_thread_ts: str | None = slack_thread_ts
    elif is_dm:
        reply_thread_ts = None
    else:
        reply_thread_ts = slack_msg_ts

    if is_dm:
        # One continuous LangGraph thread per (team, DM-channel, user). The
        # DM channel already encodes the bot<->user pair, but we still
        # include user_id explicitly so the same key would work in a
        # multi-party DM (mpim) without leaking memory across participants.
        agent_thread_id = f"slack:dm:{team_id}:{channel_id}:{user_id}"
    else:
        # Channel: scope memory to the Slack thread root. A brand-new
        # @-mention with no thread_ts uses its own ts as the root, which
        # matches how the bot will reply (threaded under that message).
        thread_root = slack_thread_ts or slack_msg_ts or ""
        agent_thread_id = f"slack:ch:{team_id}:{channel_id}:{thread_root}"

    msg = InboundMessage(
        surface=Surface.SLACK,
        tenant_external_id=team_id,
        channel_id=channel_id,
        thread_id=slack_thread_ts or slack_msg_ts or "",
        agent_thread_id=agent_thread_id,
        user_id=user_id,
        text=(event.get("text") or "").strip(),
        files=event.get("files", []),
        reply_thread_ts=reply_thread_ts,
        is_dm=is_dm,
        client_id=client_id,
        raw=body,
    )
    if not msg.text:
        return

    # `event_ts` is when Slack first saw the user's message. The gap
    # between that and `now` is "ingress lag": webhook hop + Bolt
    # signature-verify + our handler -- a useful upper bound on how
    # much time we can shave from the inbound path before the agent
    # even starts. In Socket Mode this lag drops near zero; on HTTPS
    # it's where Cloud Run cold starts show up.
    event_ts_str: str | None = event.get("event_ts") or slack_msg_ts
    ingress_lag_ms: int | None = None
    if event_ts_str:
        try:
            ingress_lag_ms = int((time.time() - float(event_ts_str)) * 1000)
        except (TypeError, ValueError):
            ingress_lag_ms = None
    log.info(
        "slack.event.enqueue",
        tenant=msg.tenant_external_id,
        slack_thread=msg.thread_id,
        agent_thread=msg.agent_thread_id,
        reply_thread=reply_thread_ts or "(top-level)",
        is_dm=is_dm,
        text_len=len(msg.text),
        ingress_lag_ms=ingress_lag_ms,
    )

    # Fire feedback within ~100ms so the user knows ARLO heard them.
    # DMs: native Slack assistant typing indicator (clears automatically
    # when the bot posts a message). Channel mentions: an :eyes: reaction
    # (no native indicator exists outside assistant threads).
    if client is not None:
        await _post_thinking_indicator(client, msg, is_dm)

    await enqueue_run_agent(msg.model_dump_json(), event_ts=event_ts_str)


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
