"""Google Workspace OAuth.

Two endpoints:
  GET /oauth/google/install?tenant_id=...   -> 302 to Google consent
  GET /oauth/google/callback?code=...&state=...  -> exchange + persist

Tokens stored encrypted on IntegrationConnection (provider='google').
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from lyra_core.common.config import get_settings
from lyra_core.common.crypto import encrypt_for_tenant
from lyra_core.common.logging import get_logger
from lyra_core.db.models import IntegrationConnection
from lyra_core.db.session import async_session
from sqlalchemy.dialects.postgresql import insert

from ._state import decode_state, encode_state

router = APIRouter()
log = get_logger(__name__)

GOOGLE_AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"  # noqa: S105 public OAuth token endpoint URL, not a secret
GOOGLE_USERINFO = "https://openidconnect.googleapis.com/v1/userinfo"


@router.get("/install")
async def install(tenant_id: str = Query(...)) -> RedirectResponse:
    settings = get_settings()
    state = encode_state(tenant_id)
    params = {
        "client_id": settings.google_oauth_client_id,
        "redirect_uri": settings.google_oauth_redirect_uri,
        "response_type": "code",
        "scope": " ".join([*settings.google_scopes_list, "openid", "email", "profile"]),
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": state,
    }
    qs = "&".join(f"{k}={httpx.QueryParams({k: v})[k]}" for k, v in params.items())
    return RedirectResponse(f"{GOOGLE_AUTH}?{qs}")


@router.get("/callback")
async def callback(request: Request, code: str = Query(...), state: str = Query(...)):
    settings = get_settings()
    try:
        tenant_id, _redirect_to = decode_state(state)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid state") from exc

    async with httpx.AsyncClient(timeout=20) as client:
        token_resp = await client.post(
            GOOGLE_TOKEN,
            data={
                "code": code,
                "client_id": settings.google_oauth_client_id,
                "client_secret": settings.google_oauth_client_secret,
                "redirect_uri": settings.google_oauth_redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        if token_resp.status_code != 200:
            log.error("google.token.error", body=token_resp.text)
            raise HTTPException(status_code=400, detail="token exchange failed")
        token_data = token_resp.json()

        user_resp = await client.get(
            GOOGLE_USERINFO,
            headers={"Authorization": f"Bearer {token_data['access_token']}"},
        )
        userinfo = user_resp.json() if user_resp.status_code == 200 else {}

    expires_at = datetime.now(UTC) + timedelta(seconds=int(token_data.get("expires_in", 3600)))
    external_id = userinfo.get("sub") or userinfo.get("email") or "unknown"
    display = userinfo.get("email") or userinfo.get("name") or external_id

    enc_access = encrypt_for_tenant(tenant_id, token_data["access_token"])
    enc_refresh = (
        encrypt_for_tenant(tenant_id, token_data["refresh_token"])
        if "refresh_token" in token_data
        else None
    )

    async with async_session() as s:
        stmt = insert(IntegrationConnection).values(
            tenant_id=tenant_id,
            provider="google",
            external_account_id=external_id,
            display_name=display,
            scopes=token_data.get("scope", ""),
            access_token_encrypted=enc_access,
            refresh_token_encrypted=enc_refresh,
            expires_at=expires_at,
            metadata_=userinfo,
            status="active",
        )
        # On re-auth (same google account, same tenant), update tokens in place.
        stmt = stmt.on_conflict_do_update(
            constraint="uq_integration_per_account",
            set_={
                "access_token_encrypted": enc_access,
                "refresh_token_encrypted": enc_refresh
                if enc_refresh
                else IntegrationConnection.refresh_token_encrypted,
                "expires_at": expires_at,
                "scopes": token_data.get("scope", ""),
                "status": "active",
                "updated_at": datetime.now(UTC),
            },
        )
        await s.execute(stmt)
        await s.commit()

    log.info("google.install.ok", tenant_id=tenant_id, account=display)
    return RedirectResponse(f"{settings.admin_base_url}/integrations?google=connected")
