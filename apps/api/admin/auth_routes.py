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
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from typing import Annotated

from lyra_core.common.config import get_settings
from lyra_core.db.models import AdminUser, SlackInstallation, Tenant
from lyra_core.db.session import async_session

router = APIRouter()

_TOKEN_TTL_HOURS = 720  # 30 days


class RegisterIn(BaseModel):
    email: EmailStr
    password: str
    passcode: str
    team_id: str  # Slack team_id to find the tenant (e.g. T0A9B4DPW2W)


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

    async with async_session() as s:
        # Find tenant by Slack team_id.
        tenant = (
            await s.execute(select(Tenant).where(Tenant.external_team_id == body.team_id))
        ).scalar_one_or_none()
        if tenant is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"no workspace found for team_id '{body.team_id}' — install the Slack app first",
            )

        # Prevent duplicate emails.
        existing = (
            await s.execute(select(AdminUser).where(AdminUser.email == body.email.lower()))
        ).scalar_one_or_none()
        if existing is not None:
            raise HTTPException(status.HTTP_409_CONFLICT, "email already registered")

        if len(body.password) < 8:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "password must be at least 8 characters")

        pw_hash = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode()
        user = AdminUser(
            tenant_id=tenant.id,
            email=body.email.lower(),
            password_hash=pw_hash,
            role="owner",
        )
        s.add(user)
        await s.commit()

    return TokenOut(access_token=_mint_jwt(tenant.id, body.email.lower(), "owner"))


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
