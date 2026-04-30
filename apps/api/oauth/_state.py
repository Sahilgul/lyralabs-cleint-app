"""Signed OAuth state helper used by Google + GHL flows.

State carries `tenant_id` so we know which workspace to attach the
resulting credential to after the user finishes the consent flow.
"""

from __future__ import annotations

import time

import jwt

from lyra_core.common.config import get_settings

_STATE_TTL_SECONDS = 600


def encode_state(tenant_id: str, redirect_to: str | None = None) -> str:
    settings = get_settings()
    payload = {
        "tid": tenant_id,
        "rt": redirect_to or "",
        "exp": int(time.time()) + _STATE_TTL_SECONDS,
        "iss": settings.admin_jwt_issuer,
    }
    return jwt.encode(payload, settings.admin_jwt_secret, algorithm="HS256")


def decode_state(token: str) -> tuple[str, str | None]:
    settings = get_settings()
    payload = jwt.decode(
        token,
        settings.admin_jwt_secret,
        algorithms=["HS256"],
        issuer=settings.admin_jwt_issuer,
    )
    return payload["tid"], payload.get("rt") or None
