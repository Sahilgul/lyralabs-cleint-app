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


# -----------------------------------------------------------------------------
# Super-admin (platform operator)
# -----------------------------------------------------------------------------
#
# Tenant admins can manage their own workspace via `current_admin`. Platform
# concerns -- swapping LLM providers, rotating master secrets, viewing
# cross-tenant analytics -- need a higher bar. Mint a JWT with `role:
# "super_admin"` (using the same ADMIN_JWT_SECRET) and pass it in the same
# Authorization: Bearer header.
#
# Keep this lightweight on purpose. When you adopt Clerk/Supabase Auth for
# the admin UI, replace this with a Clerk role check.

SUPER_ADMIN_ROLE = "super_admin"


async def current_super_admin(
    principal: Annotated[AdminPrincipal, Depends(current_admin)],
) -> AdminPrincipal:
    if principal.role != SUPER_ADMIN_ROLE:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "super-admin role required",
        )
    return principal


CurrentSuperAdmin = Annotated[AdminPrincipal, Depends(current_super_admin)]
