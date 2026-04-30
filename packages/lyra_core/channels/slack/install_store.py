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

from datetime import UTC, datetime
from logging import Logger

from slack_sdk.oauth.installation_store import Bot, Installation
from slack_sdk.oauth.installation_store.async_installation_store import (
    AsyncInstallationStore,
)
from sqlalchemy import select

from ...common.crypto import decrypt_for_tenant, encrypt_for_tenant
from ...common.logging import get_logger
from ...db.models import SlackInstallation, Tenant
from ...db.session import async_session

log = get_logger(__name__)


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
    async def _ensure_tenant(team_id: str, team_name: str | None) -> str:
        """Get-or-create the tenant row keyed on Slack team_id."""
        async with async_session() as s:
            row = (
                await s.execute(select(Tenant).where(Tenant.external_team_id == team_id))
            ).scalar_one_or_none()
            if row is None:
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
        tenant_id = await self._ensure_tenant(team_id, installation.team_name)

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
            log.info("slack.install.saved", team_id=team_id, tenant_id=tenant_id)

    async def async_save_bot(self, bot: Bot) -> None:
        # Bolt always calls async_save with the full Installation before
        # async_save_bot, so the row is already persisted. No-op here avoids
        # writing duplicate rows that would split bot+install state.
        return None

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
            bot_refresh_token=(
                decrypt_for_tenant(tenant_id, row.bot_refresh_token_encrypted)
                if row.bot_refresh_token_encrypted
                else None
            ),
            bot_token_expires_at=(
                int(row.bot_token_expires_at.timestamp()) if row.bot_token_expires_at else None
            ),
            installed_at=row.installed_at.timestamp() if row.installed_at else None,
        )

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
