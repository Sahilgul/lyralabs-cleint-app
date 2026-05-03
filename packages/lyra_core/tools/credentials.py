"""Provider-credential loading + refresh.

The agent's tools call `get_credentials(tenant_id, provider)` and never
touch the DB themselves. This module:
  - Decrypts tokens with the per-tenant Fernet key.
  - Refreshes expired access tokens (Google + GHL) and persists rotated tokens.
  - Returns an opaque `ProviderCredentials` carrying access_token + metadata.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
from pydantic import BaseModel
from sqlalchemy import select

from ..common.config import get_settings
from ..common.crypto import decrypt_for_tenant, encrypt_for_tenant
from ..common.logging import get_logger
from ..db.models import IntegrationConnection
from ..db.session import async_session

log = get_logger(__name__)

# Refresh tokens that are within this window of expiring.
_REFRESH_WINDOW = timedelta(minutes=5)


class ProviderCredentials(BaseModel):
    provider: str
    access_token: str
    refresh_token: str | None = None
    expires_at: datetime | None = None
    external_account_id: str
    scopes: str = ""
    metadata: dict = {}


async def get_credentials(
    tenant_id: str,
    provider: str,
    client_id: str | None = None,
) -> ProviderCredentials:
    """Return a fresh, decrypted credential for the given provider.

    If `client_id` is provided, looks for a client-scoped credential first.
    Falls back to a tenant-level credential (client_id IS NULL) if none found.
    This lets agency-level integrations (e.g. Google Workspace) work without
    a client scope while per-client GHL sub-accounts carry client_id.
    """
    async with async_session() as s:
        # Try client-scoped credential first (or tenant-level if client_id is None)
        filters = [
            IntegrationConnection.tenant_id == tenant_id,
            IntegrationConnection.provider == provider,
            IntegrationConnection.status == "active",
        ]
        if client_id is not None:
            filters.append(IntegrationConnection.client_id == client_id)
        else:
            filters.append(IntegrationConnection.client_id.is_(None))

        row = (
            await s.execute(
                select(IntegrationConnection)
                .where(*filters)
                .order_by(IntegrationConnection.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        # Fallback: if a client_id was given but no client-scoped credential exists,
        # try the tenant-level credential for this provider.
        if row is None and client_id is not None:
            row = (
                await s.execute(
                    select(IntegrationConnection)
                    .where(
                        IntegrationConnection.tenant_id == tenant_id,
                        IntegrationConnection.provider == provider,
                        IntegrationConnection.status == "active",
                        IntegrationConnection.client_id.is_(None),
                    )
                    .order_by(IntegrationConnection.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()

        if row is None:
            raise RuntimeError(
                f"No active {provider!r} integration for tenant {tenant_id!r}. "
                f"Ask the user to connect it via the admin panel."
            )

        # Refresh if needed.
        if row.expires_at and row.expires_at <= datetime.now(UTC) + _REFRESH_WINDOW:
            if not row.refresh_token_encrypted:
                raise RuntimeError(
                    f"{provider} token expired and no refresh token; user must re-auth"
                )
            new_access, new_refresh, new_exp = await _refresh_token(
                provider=provider,
                refresh_token=decrypt_for_tenant(tenant_id, row.refresh_token_encrypted),
            )
            row.access_token_encrypted = encrypt_for_tenant(tenant_id, new_access)
            if new_refresh:
                row.refresh_token_encrypted = encrypt_for_tenant(tenant_id, new_refresh)
            row.expires_at = new_exp
            await s.commit()
            access_token = new_access
        else:
            access_token = decrypt_for_tenant(tenant_id, row.access_token_encrypted)

        return ProviderCredentials(
            provider=provider,
            access_token=access_token,
            refresh_token=(
                decrypt_for_tenant(tenant_id, row.refresh_token_encrypted)
                if row.refresh_token_encrypted
                else None
            ),
            expires_at=row.expires_at,
            external_account_id=row.external_account_id,
            scopes=row.scopes,
            metadata=row.metadata_ or {},
        )


async def _refresh_token(*, provider: str, refresh_token: str) -> tuple[str, str | None, datetime]:
    settings = get_settings()
    if provider == "google":
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": settings.google_oauth_client_id,
                    "client_secret": settings.google_oauth_client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
            )
        if r.status_code != 200:
            log.error("google.refresh.error", body=r.text)
            raise RuntimeError(f"google token refresh failed: {r.text}")
        data = r.json()
        return (
            data["access_token"],
            data.get("refresh_token"),  # Google rarely rotates this
            datetime.now(UTC) + timedelta(seconds=int(data.get("expires_in", 3600))),
        )

    if provider == "ghl":
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(
                "https://services.leadconnectorhq.com/oauth/token",
                data={
                    "client_id": settings.ghl_client_id,
                    "client_secret": settings.ghl_client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                    "user_type": "Location",
                },
                headers={"Accept": "application/json"},
            )
        if r.status_code != 200:
            log.error("ghl.refresh.error", body=r.text)
            raise RuntimeError(f"ghl token refresh failed: {r.text}")
        data = r.json()
        return (
            data["access_token"],
            data.get("refresh_token"),
            datetime.now(UTC) + timedelta(seconds=int(data.get("expires_in", 86400))),
        )

    raise ValueError(f"refresh not implemented for provider {provider!r}")
