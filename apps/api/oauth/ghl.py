"""GoHighLevel v2 OAuth.

Marketplace docs: https://highlevel.stoplight.io/docs/integrations/

Notes:
- GHL issues access + refresh tokens. Access expires in 24h.
- The token response includes `locationId` (sub-account) or `companyId`
  (agency) depending on the user's selection during install.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse
from sqlalchemy.dialects.postgresql import insert

from lyra_core.common.config import get_settings
from lyra_core.common.crypto import encrypt_for_tenant
from lyra_core.common.logging import get_logger
from lyra_core.db.models import IntegrationConnection
from lyra_core.db.session import async_session

from ._state import decode_state, encode_state

router = APIRouter()
log = get_logger(__name__)

GHL_AUTH = "https://marketplace.gohighlevel.com/oauth/chooselocation"
GHL_TOKEN = "https://services.leadconnectorhq.com/oauth/token"


@router.get("/install")
async def install(tenant_id: str = Query(...)) -> RedirectResponse:
    settings = get_settings()
    state = encode_state(tenant_id)
    params = {
        "response_type": "code",
        "redirect_uri": settings.ghl_redirect_uri,
        "client_id": settings.ghl_client_id,
        "scope": " ".join(settings.ghl_scopes_list),
        "state": state,
    }
    qs = "&".join(f"{k}={httpx.QueryParams({k: v})[k]}" for k, v in params.items())
    return RedirectResponse(f"{GHL_AUTH}?{qs}")


@router.get("/callback")
async def callback(
    code: str = Query(...),
    state: str = Query(...),
    location_id: str | None = Query(None, alias="locationId"),  # noqa: ARG001
):
    settings = get_settings()
    try:
        tenant_id, _redirect = decode_state(state)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid state") from exc

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            GHL_TOKEN,
            data={
                "client_id": settings.ghl_client_id,
                "client_secret": settings.ghl_client_secret,
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": settings.ghl_redirect_uri,
                "user_type": "Location",
            },
            headers={"Accept": "application/json"},
        )
    if resp.status_code != 200:
        log.error("ghl.token.error", status=resp.status_code, body=resp.text)
        raise HTTPException(status_code=400, detail="token exchange failed")
    data = resp.json()

    expires_at = datetime.now(UTC) + timedelta(seconds=int(data.get("expires_in", 86400)))
    external_id = data.get("locationId") or data.get("companyId") or "unknown"
    display = data.get("locationId") or data.get("companyId") or "GHL account"

    enc_access = encrypt_for_tenant(tenant_id, data["access_token"])
    enc_refresh = (
        encrypt_for_tenant(tenant_id, data["refresh_token"]) if "refresh_token" in data else None
    )

    async with async_session() as s:
        stmt = insert(IntegrationConnection).values(
            tenant_id=tenant_id,
            provider="ghl",
            external_account_id=str(external_id),
            display_name=str(display),
            scopes=data.get("scope", ""),
            access_token_encrypted=enc_access,
            refresh_token_encrypted=enc_refresh,
            expires_at=expires_at,
            metadata_={
                "user_type": data.get("userType"),
                "company_id": data.get("companyId"),
                "location_id": data.get("locationId"),
            },
            status="active",
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_integration_per_account",
            set_={
                "access_token_encrypted": enc_access,
                "refresh_token_encrypted": enc_refresh
                if enc_refresh
                else IntegrationConnection.refresh_token_encrypted,
                "expires_at": expires_at,
                "scopes": data.get("scope", ""),
                "status": "active",
                "updated_at": datetime.now(UTC),
            },
        )
        await s.execute(stmt)
        await s.commit()

    log.info("ghl.install.ok", tenant_id=tenant_id, account=display)
    return RedirectResponse(f"{settings.admin_base_url}/integrations?ghl=connected")
