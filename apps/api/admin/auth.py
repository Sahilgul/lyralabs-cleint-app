"""Admin-panel auth.

For MVP we accept a JWT signed with ADMIN_JWT_SECRET (could be issued by
Clerk, Supabase Auth, or our own login route). The JWT must contain a
`tenant_id` claim AND an `email` claim. Replace with Clerk middleware when
you adopt it.
"""

from __future__ import annotations

from typing import Annotated

import jwt
from fastapi import Depends, Header, HTTPException, status
from pydantic import BaseModel

from lyra_core.common.config import get_settings


class AdminPrincipal(BaseModel):
    tenant_id: str
    email: str
    role: str = "owner"


async def current_admin(
    authorization: Annotated[str | None, Header()] = None,
) -> AdminPrincipal:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    token = authorization.split(" ", 1)[1]
    settings = get_settings()
    try:
        claims = jwt.decode(
            token,
            settings.admin_jwt_secret,
            algorithms=["HS256"],
            issuer=settings.admin_jwt_issuer,
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"invalid token: {exc}") from exc

    if "tenant_id" not in claims or "email" not in claims:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing claims")

    return AdminPrincipal(
        tenant_id=claims["tenant_id"], email=claims["email"], role=claims.get("role", "owner")
    )


CurrentAdmin = Annotated[AdminPrincipal, Depends(current_admin)]
