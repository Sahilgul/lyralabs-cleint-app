"""Admin-panel login / register endpoints.

POST /admin/auth/register  — passcode-gated account creation
POST /admin/auth/login     — returns a signed JWT

Registration is restricted by a server-side passcode (ADMIN_REGISTER_PASSCODE
env var, defaults to "7172"). Set it to something else in production or disable
registration once real auth (Clerk/Supabase) replaces this flow.
"""

from __future__ import annotations

import bcrypt
import jwt
from datetime import UTC, datetime, timedelta
from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from typing import Annotated

from lyra_core.common.config import get_settings
from lyra_core.db.models import AdminUser, Client, Tenant
from lyra_core.db.session import async_session

router = APIRouter()

_TOKEN_TTL_HOURS = 720  # 30 days


class RegisterIn(BaseModel):
    email: EmailStr
    password: str
    passcode: str


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


def _mint_jwt(tenant_id: str, email: str, role: str) -> str:
    settings = get_settings()
    now = datetime.now(UTC)
    payload = {
        "tenant_id": tenant_id,
        "email": email,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=_TOKEN_TTL_HOURS)).timestamp()),
        "iss": settings.admin_jwt_issuer,
    }
    return jwt.encode(payload, settings.admin_jwt_secret, algorithm="HS256")


@router.post("/register", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterIn) -> TokenOut:
    settings = get_settings()
    expected_passcode = getattr(settings, "admin_register_passcode", "7172")
    if body.passcode != expected_passcode:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "invalid passcode")

    if len(body.password) < 8:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "password must be at least 8 characters")

    async with async_session() as s:
        # Prevent duplicate emails.
        existing = (
            await s.execute(select(AdminUser).where(AdminUser.email == body.email.lower()))
        ).scalar_one_or_none()
        if existing is not None:
            raise HTTPException(status.HTTP_409_CONFLICT, "email already registered")

        # Auto-create a Tenant. external_team_id is a placeholder until Slack connects.
        email_username = body.email.lower().split("@")[0]
        tenant = Tenant(
            external_team_id=f"pending-{email_username}-{int(__import__('time').time())}",
            channel="slack",
            name=email_username,
            plan="trial",
            status="active",
            trial_credit_remaining_usd=100.0,
        )
        s.add(tenant)
        await s.flush()  # populates tenant.id

        client = Client(
            tenant_id=tenant.id,
            name=email_username,
            slug="agency_internal",
            status="active",
        )
        s.add(client)

        pw_hash = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode()
        user = AdminUser(
            tenant_id=tenant.id,
            email=body.email.lower(),
            password_hash=pw_hash,
            role="owner",
        )
        s.add(user)
        await s.commit()
        tenant_id = tenant.id
        user_id = user.id

    # Update placeholder to include user.id for uniqueness
    async with async_session() as s:
        t = (await s.execute(select(Tenant).where(Tenant.id == tenant_id))).scalar_one()
        t.external_team_id = f"pending-{user_id}"
        await s.commit()

    return TokenOut(access_token=_mint_jwt(tenant_id, body.email.lower(), "owner"))


@router.post("/login", response_model=TokenOut)
async def login(body: LoginIn) -> TokenOut:
    async with async_session() as s:
        user = (
            await s.execute(select(AdminUser).where(AdminUser.email == body.email.lower()))
        ).scalar_one_or_none()

    invalid = HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid email or password")
    if user is None:
        raise invalid

    if not bcrypt.checkpw(body.password.encode(), user.password_hash.encode()):
        raise invalid

    return TokenOut(access_token=_mint_jwt(user.tenant_id, user.email, user.role))


@router.get("/slack-install-url")
async def slack_install_url(authorization: Annotated[str | None, Header()] = None) -> dict[str, str]:
    """Return a short-lived signed Slack install URL for the calling tenant.

    The `sig` param is a 10-minute JWT carrying tenant_id. The
    /oauth/slack/install endpoint verifies the signature before passing
    tenant_id to Bolt's metadata, preventing CSRF-style binding attacks.
    """
    from apps.api.admin.auth import current_admin  # noqa: PLC0415
    from apps.api.oauth._state import encode_state  # noqa: PLC0415

    principal = await current_admin(authorization=authorization)
    sig = encode_state(principal.tenant_id)
    settings = get_settings()
    base = settings.app_base_url.rstrip("/")
    return {"url": f"{base}/oauth/slack/install?sig={sig}"}
