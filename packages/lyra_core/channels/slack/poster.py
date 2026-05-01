"""Async helpers for posting replies and uploading artifacts back to Slack."""

from __future__ import annotations

from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient
from sqlalchemy import select

from ...common.crypto import decrypt_for_tenant
from ...common.logging import get_logger
from ...db.models import SlackInstallation
from ...db.session import async_session
from ..schema import OutboundReply

log = get_logger(__name__)


async def _bot_token_for(tenant_id: str) -> str:
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
        return decrypt_for_tenant(tenant_id, row.bot_token_encrypted)


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

    resp = await client.chat_postMessage(
        channel=reply.channel_id,
        thread_ts=reply.thread_ts,
        text=reply.text or " ",
        blocks=reply.blocks or None,
    )
    parent_ts = resp.get("ts", "")

    for art in reply.artifacts:
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
    )
    return parent_ts
