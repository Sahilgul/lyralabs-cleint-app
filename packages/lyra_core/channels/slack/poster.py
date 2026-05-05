"""Async helpers for posting replies and uploading artifacts back to Slack."""

from __future__ import annotations

import hashlib
import json
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


# Reply-dedup window. arq retries (max_tries=25, defer ~30s back-off) cluster
# inside ~12 min in practice, so 5 min covers the realistic crash-and-retry
# storm without holding state forever. A retry that lands AFTER the TTL
# expires would re-post -- accepted as a rare worst case.
_DEDUP_TTL_SECONDS = 300
_DEDUP_KEY_PREFIX = "arlo:posted:"


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


def _compute_content_hash(reply: OutboundReply) -> str:
    """Stable hash of the user-visible payload of a reply.

    Hashes both `text` and `blocks` because approval cards share the same
    fallback text but have plan-specific blocks; without blocks in the key,
    two distinct plans posted to the same thread would dedup as one.
    `sort_keys=True` so structurally-identical dicts hash the same way
    regardless of insertion order; `default=str` covers anything not
    JSON-native (datetimes, enums, etc.) so the hash never crashes the post.

    Uses SHA-256 (truncated to 16 hex chars) purely as a stable identity
    function for dedup. This is NOT a security boundary — collision
    resistance against an adversary is irrelevant here; the worst case of
    a hash collision is one user-visible Slack message getting suppressed.
    SHA-256 over SHA-1 only because ruff's S324 flags SHA-1 and there is
    no upside to keeping it.
    """
    payload = (
        (reply.text or "")
        + "\x1e"
        + json.dumps(
            reply.blocks or [],
            sort_keys=True,
            default=str,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


async def _claim_dedup_slot(key: str) -> bool:
    """Reserve a dedup slot for `key`. Returns True iff this caller is the
    first to post (i.e. should proceed); False if a previous attempt within
    the TTL already posted (caller should skip).

    Backed by Redis ``SET NX EX`` on the arq pool. Fails open: if Redis is
    unreachable or raises, returns True so we never block a user-facing
    reply on a dedup-cache outage. The cost of a false-True (one duplicate
    slipping through) is one extra Slack message; the cost of a false-False
    would be a silent agent -- much worse.
    """
    try:
        # Imported lazily to avoid a parse-time cycle: worker.queue depends
        # on common.config which is loaded everywhere; poster.py is loaded
        # by the Slack adapter at import time.
        from ...worker.queue import _get_pool

        pool = await _get_pool()
        was_set = await pool.set(key, "1", nx=True, ex=_DEDUP_TTL_SECONDS)
        return bool(was_set)
    except Exception as exc:
        log.warning(
            "slack.reply.dedup_unavailable",
            error=str(exc)[:200],
        )
        return True


async def post_reply(tenant_id: str, reply: OutboundReply) -> str:
    """Post text/blocks to a channel/thread and upload any artifacts.

    `reply.thread_ts` controls threading: when set, the message becomes a
    threaded reply on that ts; when None, it goes to the channel/DM as a
    new top-level message. This is what keeps DM conversations from
    collapsing into a single thread on the user's first message.

    Idempotency: agent_node calls post_reply mid-graph-step. If anything
    later in _run raises before LangGraph checkpoints the node, arq retries
    up to ``_MAX_TRIES`` times -- each retry re-runs agent_node and would
    re-post the same Slack message (root cause of the "I hear you /
    Understood" cascade observed in DLQ jobs). We short-circuit duplicates
    via Redis ``SET NX EX`` keyed on
    ``(tenant, channel, thread_ts, content_hash)``. First caller wins;
    later attempts within ``_DEDUP_TTL_SECONDS`` log and return early.

    Returns the ts of the posted message, or an empty string when a
    duplicate was suppressed. Callers currently discard the return value,
    so the sentinel is safe.
    """
    content_hash = _compute_content_hash(reply)
    dedup_key = (
        f"{_DEDUP_KEY_PREFIX}{tenant_id}:{reply.channel_id}:{reply.thread_ts or ''}:{content_hash}"
    )
    if not await _claim_dedup_slot(dedup_key):
        log.info(
            "slack.reply.dedup_skipped",
            tenant=tenant_id,
            channel=reply.channel_id,
            thread_ts=reply.thread_ts or "(top-level)",
            content_hash=content_hash,
        )
        return ""

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
        async with phase("slack.files_upload_v2", filename=art.filename, n_bytes=len(art.content)):
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
