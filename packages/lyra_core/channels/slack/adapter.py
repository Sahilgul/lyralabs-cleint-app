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
            process_before_response=True,
        )
    else:
        log.warning(
            "slack.config.missing",
            reason="SLACK_CLIENT_ID/SECRET/SIGNING_SECRET not set; running stub Slack app",
        )
        app = AsyncApp(
            signing_secret=settings.slack_signing_secret or "missing",
            token="xoxb-missing",
            process_before_response=True,
        )

    @app.event("app_mention")
    async def on_mention(body: dict[str, Any], ack: Any, say: Any) -> None:
        await ack()
        await _enqueue_from_event(body)

    @app.event("message")
    async def on_message(body: dict[str, Any], ack: Any) -> None:
        await ack()
        event = body.get("event", {})
        # Only respond in DMs or threads we're in; ignore bot messages.
        if event.get("bot_id") or event.get("subtype"):
            return
        if event.get("channel_type") == "im":
            await _enqueue_from_event(body)

    @app.command("/arlo")
    async def slash_command(ack: Any, command: dict[str, Any]) -> None:
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
        await _enqueue_from_event(synthetic)

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


async def _enqueue_from_event(body: dict[str, Any]) -> None:
    """Translate a Slack event into InboundMessage + dispatch to Celery."""
    from apps.worker.tasks.run_agent import run_agent  # noqa: PLC0415

    event = body.get("event") or {}
    msg = InboundMessage(
        surface=Surface.SLACK,
        tenant_external_id=body.get("team_id") or body.get("api_app_id") or "",
        channel_id=event.get("channel", ""),
        thread_id=event.get("thread_ts") or event.get("ts") or "",
        user_id=event.get("user", ""),
        text=(event.get("text") or "").strip(),
        files=event.get("files", []),
        parent_message_ts=event.get("thread_ts"),
        raw=body,
    )
    if not msg.text:
        return
    log.info(
        "slack.event.enqueue",
        tenant=msg.tenant_external_id,
        thread=msg.thread_id,
        text_len=len(msg.text),
    )
    run_agent.delay(message_json=msg.model_dump_json())
