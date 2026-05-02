"""apps.api.admin.auth."""

from __future__ import annotations

import time

import jwt
import pytest
from fastapi import HTTPException

from apps.api.admin.auth import AdminPrincipal, current_admin
from lyra_core.common.config import get_settings


def _token(claims: dict) -> str:
    s = get_settings()
    base = {"iss": s.admin_jwt_issuer, "exp": int(time.time()) + 60}
    base.update(claims)
    return jwt.encode(base, s.admin_jwt_secret, algorithm="HS256")


@pytest.mark.asyncio
async def test_returns_principal_when_token_valid() -> None:
    tok = _token({"tenant_id": "t-1", "email": "a@x.com", "role": "admin"})
    p = await current_admin(authorization=f"Bearer {tok}")
    assert isinstance(p, AdminPrincipal)
    assert p.tenant_id == "t-1"
    assert p.email == "a@x.com"
    assert p.role == "admin"


@pytest.mark.asyncio
async def test_default_role_is_owner() -> None:
    tok = _token({"tenant_id": "t-1", "email": "a@x.com"})
    p = await current_admin(authorization=f"Bearer {tok}")
    assert p.role == "owner"


@pytest.mark.asyncio
async def test_missing_authorization_raises_401() -> None:
    with pytest.raises(HTTPException) as exc:
        await current_admin(authorization=None)
    assert exc.value.status_code == 401
    assert "missing bearer token" in exc.value.detail


@pytest.mark.asyncio
async def test_non_bearer_scheme_raises_401() -> None:
    with pytest.raises(HTTPException) as exc:
        await current_admin(authorization="Basic abc")
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_invalid_signature_raises_401() -> None:
    bad = jwt.encode(
        {"tenant_id": "t", "email": "x", "iss": get_settings().admin_jwt_issuer,
         "exp": int(time.time()) + 60},
        "wrong-secret-padded-to-32-bytes!!",
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        await current_admin(authorization=f"Bearer {bad}")
    assert exc.value.status_code == 401
    assert "invalid token" in exc.value.detail


@pytest.mark.asyncio
async def test_expired_token_raises_401() -> None:
    s = get_settings()
    expired = jwt.encode(
        {"tenant_id": "t", "email": "x", "iss": s.admin_jwt_issuer, "exp": int(time.time()) - 1},
        s.admin_jwt_secret,
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        await current_admin(authorization=f"Bearer {expired}")
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_missing_tenant_id_claim_raises_401() -> None:
    tok = _token({"email": "x@y.com"})
    with pytest.raises(HTTPException) as exc:
        await current_admin(authorization=f"Bearer {tok}")
    assert exc.value.status_code == 401
    assert "missing claims" in exc.value.detail


@pytest.mark.asyncio
async def test_missing_email_claim_raises_401() -> None:
    tok = _token({"tenant_id": "t-1"})
    with pytest.raises(HTTPException) as exc:
        await current_admin(authorization=f"Bearer {tok}")
    assert exc.value.status_code == 401
