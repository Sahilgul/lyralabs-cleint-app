"""Async helpers for posting replies and uploading artifacts back to Slack."""

from __future__ import annotations

import time

from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient
from sqlalchemy import select

from ...common.crypto import decrypt_for_tenant
from ...common.logging import get_logger, phase
from ...db.models import SlackInstallation
from ...db.session import async_session
from ..schema import OutboundReply

log = get_logger(__name__)


# Process-local cache for the decrypted bot token. Refetching from
# Postgres + Fernet-decrypting on every reply costs ~200-300ms (Tokyo
# pooler round-trip + decrypt). Cache lifetime is bounded so a tenant
# uninstall propagates within ~10 min.
_BOT_TOKEN_TTL_SECONDS = 600.0
_bot_token_cache: dict[str, tuple[float, str]] = {}


async def _bot_token_for(tenant_id: str) -> str:
    """Return the tenant's Slack bot token, hitting the in-process cache first."""
    cached = _bot_token_cache.get(tenant_id)
    if cached is not None:
        cached_at, token = cached
        if time.monotonic() - cached_at < _BOT_TOKEN_TTL_SECONDS:
            return token

    async with phase("slack.token_fetch", tenant_id=tenant_id):
        async with async_session() as s:
            row = (
                await s.execute(
                    select(SlackInstallation)
                    .where(SlackInstallation.tenant_id == tenant_id)
                    .order_by(SlackInstallation.installed_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if row is None or row.bot_token_encrypted is None:
                raise RuntimeError(f"No Slack installation for tenant {tenant_id}")
            token = decrypt_for_tenant(tenant_id, row.bot_token_encrypted)

    _bot_token_cache[tenant_id] = (time.monotonic(), token)
    return token


def invalidate_bot_token_cache(tenant_id: str | None = None) -> None:
    """Drop the cached token for a tenant (or all tenants).

    Call from the OAuth save / uninstall path so a freshly rotated
    token is picked up immediately instead of after the TTL expires.
    """
    if tenant_id is None:
        _bot_token_cache.clear()
    else:
        _bot_token_cache.pop(tenant_id, None)


async def post_reply(tenant_id: str, reply: OutboundReply) -> str:
    """Post text/blocks to a channel/thread and upload any artifacts.

    `reply.thread_ts` controls threading: when set, the message becomes a
    threaded reply on that ts; when None, it goes to the channel/DM as a
    new top-level message. This is what keeps DM conversations from
    collapsing into a single thread on the user's first message.

    Returns the ts of the posted message.
    """
    token = await _bot_token_for(tenant_id)
    client = AsyncWebClient(token=token)

    posted_at = time.perf_counter()
    async with phase(
        "slack.chat_postMessage",
        channel=reply.channel_id,
        threaded=reply.thread_ts is not None,
        n_blocks=len(reply.blocks or []),
    ):
        resp = await client.chat_postMessage(
            channel=reply.channel_id,
            thread_ts=reply.thread_ts,
            text=reply.text or " ",
            blocks=reply.blocks or None,
        )
    parent_ts = resp.get("ts", "")

    for art in reply.artifacts:
        async with phase(
            "slack.files_upload_v2", filename=art.filename, n_bytes=len(art.content)
        ):
            await client.files_upload_v2(
                channel=reply.channel_id,
                thread_ts=reply.thread_ts,
                content=art.content,
                filename=art.filename,
                title=art.description or art.filename,
            )

    if reply.assistant_status_thread_ts:
        try:
            await client.assistant_threads_setStatus(
                channel_id=reply.channel_id,
                thread_ts=reply.assistant_status_thread_ts,
                status="",
            )
        except SlackApiError as e:
            log.info(
                "slack.indicator.clear_skipped",
                error=getattr(e.response, "data", {}).get("error", str(e)),
            )

    log.info(
        "slack.reply.posted",
        tenant=tenant_id,
        channel=reply.channel_id,
        thread_ts=reply.thread_ts or "(top-level)",
        n_artifacts=len(reply.artifacts),
        duration_ms=int((time.perf_counter() - posted_at) * 1000),
    )
    return parent_ts
