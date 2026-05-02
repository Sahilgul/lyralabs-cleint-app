"""Internal helpers for the Slack tool surface.

Slack has TWO token types our tools need:

  * Bot token (`xoxb-...`) -- granted at install via the app's bot scopes.
    Used for almost everything: posting, history reads, users.info, canvases.

  * User token (`xoxp-...`) -- granted only when the installer also approved
    user_scopes during OAuth. Required for `search.messages`, which Slack
    deliberately blocks bot tokens from. If the workspace was installed
    before user_scopes were configured, the user token is absent and
    search-style tools must degrade gracefully (telling the user to
    re-authorize) rather than crash.
"""

from __future__ import annotations

from sqlalchemy import select

from ...common.crypto import decrypt_for_tenant
from ...common.logging import get_logger
from ...db.models import SlackInstallation
from ...db.session import async_session

log = get_logger(__name__)


class SlackTokenMissing(RuntimeError):
    """Raised when the requested Slack token is not available for this tenant.

    Tools should catch this and surface a friendly message instructing the
    user to re-install / re-authorize ARLO with the missing scope.
    """


async def _bot_token_for(tenant_id: str) -> str:
    """Return the decrypted bot token (xoxb-) for the tenant.

    Mirrors `lyra_core.channels.slack.poster._bot_token_for` but lives here
    so the tools layer doesn't import from the channels layer (avoids a
    circular dependency between `tools.slack` and `channels.slack`).
    """
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
        raise SlackTokenMissing(
            f"No Slack bot token for tenant {tenant_id!r}. Re-install ARLO."
        )
    return decrypt_for_tenant(tenant_id, row.bot_token_encrypted)


async def _user_token_for(tenant_id: str) -> str:
    """Return the decrypted user token (xoxp-) for the tenant.

    Some Slack methods (notably `search.messages`) only accept a user
    token and reject bot tokens with a `not_allowed_token_type` error.
    Raise `SlackTokenMissing` if the workspace was installed without
    user_scopes; the tool then surfaces a remediation message instead
    of a generic crash.
    """
    async with async_session() as s:
        row = (
            await s.execute(
                select(SlackInstallation)
                .where(SlackInstallation.tenant_id == tenant_id)
                .order_by(SlackInstallation.installed_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
    if row is None or row.user_token_encrypted is None:
        raise SlackTokenMissing(
            "No Slack user token for this tenant. The installer needs to "
            "re-install ARLO with `search:read.*` user scopes for this tool to work."
        )
    return decrypt_for_tenant(tenant_id, row.user_token_encrypted)
