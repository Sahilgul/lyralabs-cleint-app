"""Postgres-backed Slack AsyncInstallationStore.

slack_bolt's AsyncOAuthSettings calls the `async_*` methods on the configured
installation_store. Earlier this file implemented the *sync* InstallationStore
interface and tried to bridge to async with `asyncio.run()`, which:

  1. Crashes inside an already-running event loop (which uvicorn always is), and
  2. Doesn't even get called — slack_bolt's async flow looks up `async_save`,
     not `save`, so the sync interface was dead code.

We now extend AsyncInstallationStore directly. Bot/refresh tokens are
encrypted at rest with the tenant's Fernet key.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from datetime import UTC, datetime
from logging import Logger
from time import time

from slack_sdk.errors import SlackApiError
from slack_sdk.oauth.installation_store import Bot, Installation
from slack_sdk.oauth.installation_store.async_installation_store import (
    AsyncInstallationStore,
)
from slack_sdk.web.async_client import AsyncWebClient
from sqlalchemy import select, text

from ...common.config import get_settings
from ...common.crypto import decrypt_for_tenant, encrypt_for_tenant
from ...common.logging import get_logger
from ...db.models import SlackInstallation, Tenant
from ...db.session import async_session

# Rotate this many seconds before expiry so we never hand Bolt an expired token.
_ROTATION_BUFFER_SECS = 7200  # 2 hours

log = get_logger(__name__)

# Ephemeral map from Bolt's OAuth state token → tenant_id (populated during
# handle_installation, consumed in handle_callback before async_save runs).
_state_to_tenant: dict[str, str] = {}

# ContextVar set by the /oauth/slack/callback endpoint just before calling
# slack_handler.handle(), so async_save can find the right tenant.
_current_tenant_hint: ContextVar[str | None] = ContextVar("_slack_tenant_hint", default=None)


class PostgresInstallationStore(AsyncInstallationStore):
    """Async InstallationStore backed by Postgres + tenant-scoped Fernet encryption."""

    def __init__(self, logger: Logger | None = None) -> None:
        self._logger = logger

    @property
    def logger(self) -> Logger:
        # AsyncInstallationStore exposes `logger` on the base; some Bolt
        # versions read it during error paths.
        return self._logger or log  # type: ignore[return-value]

    @staticmethod
    async def _ensure_tenant(team_id: str, team_name: str | None, tenant_id_hint: str | None = None) -> str:
        """Get-or-create the tenant row keyed on Slack team_id.

        If `tenant_id_hint` is provided (passed via OAuth metadata from a
        prior admin registration), we link the existing pending tenant to this
        Slack team instead of creating a new one.
        """
        from ...db.models import AdminUser  # noqa: PLC0415

        async with async_session() as s:
            # Check if there's already a tenant with this Slack team_id.
            row = (
                await s.execute(select(Tenant).where(Tenant.external_team_id == team_id))
            ).scalar_one_or_none()
            if row is not None:
                return row.id

            # Try to link an existing pending tenant (created during admin registration).
            if tenant_id_hint:
                pending = (
                    await s.execute(select(Tenant).where(Tenant.id == tenant_id_hint))
                ).scalar_one_or_none()
                if pending is not None and pending.external_team_id.startswith("pending-"):
                    pending.external_team_id = team_id
                    if team_name:
                        pending.name = team_name
                    await s.commit()
                    log.info("tenant.linked", tenant_id=pending.id, team_id=team_id)
                    return pending.id

            # No pending tenant found — create a fresh one (direct Slack install path).
            row = Tenant(
                external_team_id=team_id,
                channel="slack",
                name=team_name or team_id,
                plan="trial",
                status="active",
                trial_credit_remaining_usd=100.0,
            )
            s.add(row)
            await s.commit()
            await s.refresh(row)
            log.info("tenant.created", tenant_id=row.id, team_id=team_id)
            return row.id

    async def _async_find(
        self, *, team_id: str | None, enterprise_id: str | None
    ) -> tuple[SlackInstallation | None, str | None]:
        async with async_session() as s:
            stmt = select(SlackInstallation).order_by(SlackInstallation.installed_at.desc())
            if team_id:
                stmt = stmt.where(SlackInstallation.team_id == team_id)
            if enterprise_id:
                stmt = stmt.where(SlackInstallation.enterprise_id == enterprise_id)
            row = (await s.execute(stmt.limit(1))).scalar_one_or_none()
            return row, row.tenant_id if row else None

    # --- AsyncInstallationStore interface -------------------------------------

    async def async_save(self, installation: Installation) -> None:
        team_id = installation.team_id or installation.enterprise_id
        if not team_id:
            raise ValueError("Slack installation missing team_id and enterprise_id")
        tenant_id_hint: str | None = _current_tenant_hint.get()
        tenant_id = await self._ensure_tenant(team_id, installation.team_name, tenant_id_hint)

        async with async_session() as s:
            row = SlackInstallation(
                tenant_id=tenant_id,
                team_id=installation.team_id,
                team_name=installation.team_name,
                enterprise_id=installation.enterprise_id,
                enterprise_name=installation.enterprise_name,
                user_id=installation.user_id,
                bot_token_encrypted=(
                    encrypt_for_tenant(tenant_id, installation.bot_token)
                    if installation.bot_token
                    else None
                ),
                bot_id=installation.bot_id,
                bot_user_id=installation.bot_user_id,
                bot_scopes=",".join(installation.bot_scopes or []),
                bot_refresh_token_encrypted=(
                    encrypt_for_tenant(tenant_id, installation.bot_refresh_token)
                    if installation.bot_refresh_token
                    else None
                ),
                bot_token_expires_at=(
                    datetime.fromtimestamp(installation.bot_token_expires_at, tz=UTC)
                    if installation.bot_token_expires_at
                    else None
                ),
                user_token_encrypted=(
                    encrypt_for_tenant(tenant_id, installation.user_token)
                    if installation.user_token
                    else None
                ),
                user_scopes=",".join(installation.user_scopes or []),
                incoming_webhook_url=installation.incoming_webhook_url,
                incoming_webhook_channel=installation.incoming_webhook_channel,
                incoming_webhook_channel_id=installation.incoming_webhook_channel_id,
                incoming_webhook_configuration_url=installation.incoming_webhook_configuration_url,
                is_enterprise_install=bool(installation.is_enterprise_install),
                token_type=installation.token_type,
            )
            s.add(row)
            await s.commit()
            # New install / re-install: drop any cached bot token from a
            # prior installation so the very next reply uses the new one.
            from .poster import invalidate_bot_token_cache  # noqa: PLC0415

            invalidate_bot_token_cache(tenant_id)
            log.info("slack.install.saved", team_id=team_id, tenant_id=tenant_id)

    async def async_save_bot(self, bot: Bot) -> None:
        """Persist a refreshed bot token in place.

        Bolt calls this in two distinct cases:
          1. Right after the initial OAuth completes -- `async_save` has already
             written a full row, so we'd just be re-writing what's there.
          2. After token rotation refreshes the bot token -- a NEW token has
             been issued and we MUST update the row so subsequent requests
             don't keep using the expired one.

        Originally we no-oped this assuming case 1, which broke case 2 silently:
        every refresh succeeded for the in-flight request but the new token was
        thrown away, so 12 hours later the bot looked broken (slack_bolt logs
        "AuthorizeResult was not found" because find_bot returned a stale row
        whose token Slack rejects). Now we always update -- if the row is
        already current the UPDATE is a no-op at the DB level.

        We never INSERT here; row creation is owned by `async_save`. If no row
        exists yet we log and return; that path indicates the install was never
        completed (or was deleted), which is recoverable by re-installing.
        """
        team_id = bot.team_id or bot.enterprise_id
        if not team_id:
            log.warning("slack.save_bot.skip_no_team_id")
            return

        async with async_session() as s:
            stmt = select(SlackInstallation).order_by(SlackInstallation.installed_at.desc())
            if bot.team_id:
                stmt = stmt.where(SlackInstallation.team_id == bot.team_id)
            if bot.enterprise_id:
                stmt = stmt.where(SlackInstallation.enterprise_id == bot.enterprise_id)
            row = (await s.execute(stmt.limit(1))).scalar_one_or_none()
            if row is None:
                log.warning(
                    "slack.save_bot.no_existing_install",
                    team_id=team_id,
                    enterprise_id=bot.enterprise_id,
                )
                return

            if bot.bot_token:
                row.bot_token_encrypted = encrypt_for_tenant(row.tenant_id, bot.bot_token)
            if bot.bot_refresh_token:
                row.bot_refresh_token_encrypted = encrypt_for_tenant(
                    row.tenant_id, bot.bot_refresh_token
                )
            if bot.bot_token_expires_at is not None:
                row.bot_token_expires_at = datetime.fromtimestamp(
                    bot.bot_token_expires_at, tz=UTC
                )
            if bot.bot_id:
                row.bot_id = bot.bot_id
            if bot.bot_user_id:
                row.bot_user_id = bot.bot_user_id

            await s.commit()
            # Critical: drop the in-process token cache so the very next
            # call uses the freshly rotated token. Without this, the cache
            # would keep returning the stale token until the TTL expired
            # (~10 min later) and Slack would reject every reply with
            # `invalid_auth` for that window -- the original symptom that
            # made us add cache invalidation here in the first place.
            from .poster import invalidate_bot_token_cache  # noqa: PLC0415

            invalidate_bot_token_cache(row.tenant_id)
            log.info(
                "slack.bot.refreshed",
                team_id=team_id,
                tenant_id=row.tenant_id,
                expires_at=(
                    row.bot_token_expires_at.isoformat() if row.bot_token_expires_at else None
                ),
            )

    def _row_to_bot(self, row: SlackInstallation, tenant_id: str) -> Bot:
        return Bot(
            app_id=None,
            enterprise_id=row.enterprise_id,
            enterprise_name=row.enterprise_name,
            team_id=row.team_id,
            team_name=row.team_name,
            bot_token=decrypt_for_tenant(tenant_id, row.bot_token_encrypted),
            bot_id=row.bot_id,
            bot_user_id=row.bot_user_id,
            bot_scopes=(row.bot_scopes or "").split(",") if row.bot_scopes else [],
            # Don't expose refresh_token to Bolt — we handle rotation ourselves.
            # Returning it would cause Bolt to attempt a concurrent rotation.
            bot_refresh_token=None,
            bot_token_expires_at=None,
            installed_at=row.installed_at.timestamp() if row.installed_at else None,
        )

    async def _try_rotate(self, row: SlackInstallation, tenant_id: str) -> bool:
        """Refresh the bot token via Slack's oauth.v2.access (grant_type=refresh_token).

        Uses a Postgres transaction-level advisory lock (pg_try_advisory_xact_lock)
        so only one worker calls Slack at a time for the same team. If the lock is
        held (another worker is already rotating), we wait up to 5 s for that worker
        to finish and then re-check the DB — no second Slack API call.

        Returns True if the row now has a fresh token (either we rotated it or the
        other worker did), False if rotation failed unrecoverably.
        """
        settings = get_settings()
        if not (settings.slack_client_id and settings.slack_client_secret):
            log.warning("slack.rotation.skip_no_creds", team_id=row.team_id)
            return False

        # Stable lock key derived from team_id (fits in int64).
        lock_key = int.from_bytes(row.team_id.encode()[:8].ljust(8, b"\x00"), "big") & 0x7FFFFFFFFFFFFFFF

        new_token: str | None = None
        new_expires_at: datetime | None = None

        async with async_session() as s:
            async with s.begin():
                locked = (
                    await s.execute(text(f"SELECT pg_try_advisory_xact_lock({lock_key})"))
                ).scalar()

                if not locked:
                    # Another worker is rotating — fall through to wait below.
                    pass
                else:
                    # We hold the lock for this entire transaction (released on commit/rollback).
                    current = (
                        await s.execute(
                            select(SlackInstallation).where(SlackInstallation.id == row.id)
                        )
                    ).scalar_one_or_none()
                    if current is None:
                        return False
                    if current.bot_token_expires_at and current.bot_token_expires_at.timestamp() > time() + 60:
                        # Another worker already refreshed.
                        row.bot_token_encrypted = current.bot_token_encrypted
                        row.bot_refresh_token_encrypted = current.bot_refresh_token_encrypted
                        row.bot_token_expires_at = current.bot_token_expires_at
                        return True

                    if not current.bot_refresh_token_encrypted:
                        log.error("slack.rotation.no_refresh_token", team_id=row.team_id)
                        return False

                    refresh_token = decrypt_for_tenant(tenant_id, current.bot_refresh_token_encrypted)
                    try:
                        client = AsyncWebClient(token=None)
                        resp = await client.oauth_v2_access(
                            client_id=settings.slack_client_id,
                            client_secret=settings.slack_client_secret,
                            grant_type="refresh_token",
                            refresh_token=refresh_token,
                        )
                    except SlackApiError as e:
                        log.error("slack.rotation.api_error", team_id=row.team_id, error=str(e))
                        return False

                    if resp.get("token_type") != "bot":
                        log.error("slack.rotation.unexpected_token_type", team_id=row.team_id, data=resp.data)
                        return False

                    new_token = resp["access_token"]
                    new_refresh = resp.get("refresh_token")
                    new_expires_at = datetime.fromtimestamp(int(time()) + int(resp["expires_in"]), tz=UTC)

                    current.bot_token_encrypted = encrypt_for_tenant(tenant_id, new_token)
                    if new_refresh:
                        current.bot_refresh_token_encrypted = encrypt_for_tenant(tenant_id, new_refresh)
                    current.bot_token_expires_at = new_expires_at
                    # s.begin() context commits here, which also releases the advisory lock.

        if not locked:
            # Lock was held by another worker; wait for it then re-read from DB.
            log.info("slack.rotation.waiting_for_lock", team_id=row.team_id)
            await asyncio.sleep(5)
            async with async_session() as s2:
                fresh = (
                    await s2.execute(
                        select(SlackInstallation).where(SlackInstallation.id == row.id)
                    )
                ).scalar_one_or_none()
            if fresh and fresh.bot_token_expires_at and fresh.bot_token_expires_at.timestamp() > time() + 60:
                row.bot_token_encrypted = fresh.bot_token_encrypted
                row.bot_refresh_token_encrypted = fresh.bot_refresh_token_encrypted
                row.bot_token_expires_at = fresh.bot_token_expires_at
                return True
            log.error("slack.rotation.lock_wait_no_refresh", team_id=row.team_id)
            return False

        if new_token is None:
            return False

        row.bot_token_encrypted = current.bot_token_encrypted
        row.bot_refresh_token_encrypted = current.bot_refresh_token_encrypted
        row.bot_token_expires_at = current.bot_token_expires_at

        from .poster import invalidate_bot_token_cache  # noqa: PLC0415
        invalidate_bot_token_cache(tenant_id)
        log.info(
            "slack.rotation.success",
            team_id=row.team_id,
            tenant_id=tenant_id,
            expires_at=new_expires_at.isoformat(),
        )
        return True

    async def async_find_bot(
        self,
        *,
        enterprise_id: str | None,
        team_id: str | None,
        is_enterprise_install: bool | None = False,
    ) -> Bot | None:
        row, tenant_id = await self._async_find(team_id=team_id, enterprise_id=enterprise_id)
        if row is None or tenant_id is None or row.bot_token_encrypted is None:
            return None

        needs_rotation = (
            row.bot_token_expires_at is not None
            and row.bot_token_expires_at.timestamp() < time() + _ROTATION_BUFFER_SECS
            and row.bot_refresh_token_encrypted is not None
        )
        if needs_rotation:
            ok = await self._try_rotate(row, tenant_id)
            if not ok:
                log.error(
                    "slack.rotation.failed_returning_stale",
                    team_id=team_id,
                    expires_at=row.bot_token_expires_at.isoformat() if row.bot_token_expires_at else None,
                )
                # Return None so Bolt surfaces the error clearly rather than
                # attempting to use a token Slack will reject.
                return None

        return self._row_to_bot(row, tenant_id)

    async def async_find_installation(
        self,
        *,
        enterprise_id: str | None,
        team_id: str | None,
        user_id: str | None = None,
        is_enterprise_install: bool | None = False,
    ) -> Installation | None:
        bot = await self.async_find_bot(
            enterprise_id=enterprise_id,
            team_id=team_id,
            is_enterprise_install=is_enterprise_install,
        )
        if bot is None:
            return None
        return Installation(
            app_id=bot.app_id,
            enterprise_id=bot.enterprise_id,
            team_id=bot.team_id,
            team_name=bot.team_name,
            bot_token=bot.bot_token,
            bot_id=bot.bot_id,
            bot_user_id=bot.bot_user_id,
            bot_scopes=bot.bot_scopes,
            user_id=user_id or "",
            installed_at=bot.installed_at,
        )

    async def async_delete_bot(
        self, *, enterprise_id: str | None, team_id: str | None
    ) -> None:  # pragma: no cover
        # Soft-delete on uninstall is handled in adapter._disable_workspace.
        return None

    async def async_delete_installation(
        self,
        *,
        enterprise_id: str | None,
        team_id: str | None,
        user_id: str | None = None,
    ) -> None:  # pragma: no cover
        return None

    async def async_delete_all(
        self, *, enterprise_id: str | None, team_id: str | None
    ) -> None:  # pragma: no cover
        return None
