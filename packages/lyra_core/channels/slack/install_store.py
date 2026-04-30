"""Postgres-backed Slack InstallationStore.

slack_bolt expects an InstallationStore for OAuth flow persistence. We
implement it on top of our SlackInstallation table so installations are
durable and tied to tenants.

Bot tokens are encrypted at rest with the tenant's Fernet key.
"""

from __future__ import annotations

from datetime import UTC, datetime
from logging import Logger
from typing import Any

from slack_sdk.oauth.installation_store import Bot, Installation, InstallationStore
from sqlalchemy import select

from ...common.crypto import decrypt_for_tenant, encrypt_for_tenant
from ...common.logging import get_logger
from ...db.models import SlackInstallation, Tenant
from ...db.session import async_session

log = get_logger(__name__)


class PostgresInstallationStore(InstallationStore):
    """Sync API surface required by slack_bolt; we run async work via asyncio.run for installs.

    Note: slack_bolt's sync installation store interface is invoked from request
    threads by `AsyncSlackRequestHandler`. We use a thread-safe sync session
    pattern by spawning a fresh asyncio loop call. For high-volume installs,
    swap to AsyncInstallationStore.
    """

    def __init__(self, logger: Logger | None = None) -> None:
        self._logger = logger

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

    async def _async_save(self, installation: Installation) -> None:
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

    # --- slack_bolt sync interface ------------------------------------------

    def save(self, installation: Installation) -> None:
        import asyncio

        asyncio.run(self._async_save(installation))

    def find_bot(
        self,
        *,
        enterprise_id: str | None,
        team_id: str | None,
        is_enterprise_install: bool | None = False,
    ) -> Bot | None:
        import asyncio

        row, tenant_id = asyncio.run(
            self._async_find(team_id=team_id, enterprise_id=enterprise_id)
        )
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

    def find_installation(
        self,
        *,
        enterprise_id: str | None,
        team_id: str | None,
        user_id: str | None = None,
        is_enterprise_install: bool | None = False,
    ) -> Installation | None:
        bot = self.find_bot(
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

    def delete_bot(
        self, *, enterprise_id: str | None, team_id: str | None
    ) -> None:  # pragma: no cover - rarely used
        # Soft delete: leave history, mark token blank. Implement when needed.
        pass

    def delete_installation(
        self, *, enterprise_id: str | None, team_id: str | None, user_id: str | None = None
    ) -> None:  # pragma: no cover
        pass

    def delete_all(
        self, *, enterprise_id: str | None, team_id: str | None
    ) -> None:  # pragma: no cover
        pass
