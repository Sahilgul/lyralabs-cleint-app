#!/usr/bin/env python
"""Dev-only helper: insert a single tenant row so admin endpoints stop 500ing.

Use the SAME tenant_id you mint JWTs with via scripts/mint_admin_jwt.py.
The default below matches the dev convention: 00000000-0000-0000-0000-000000000001.

Usage:
    python scripts/seed_dev_tenant.py
    python scripts/seed_dev_tenant.py --tenant <UUID> --name "My Org"

Idempotent: if the tenant already exists it just prints the row and exits 0.
Connects via DATABASE_URL from .env (or env), so this works against local
Postgres, Supabase, or whatever DATABASE_URL points at.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from uuid import UUID

from sqlalchemy import select

from lyra_core.db.models import Tenant
from lyra_core.db.session import async_session


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tenant", default="00000000-0000-0000-0000-000000000001")
    p.add_argument("--external-team-id", default="lyralabs-dev-001")
    p.add_argument("--name", default="Lyralabs Dev Workspace")
    p.add_argument("--channel", default="slack", choices=["slack", "teams"])
    args = p.parse_args()

    try:
        UUID(args.tenant)
    except ValueError:
        print(f"ERROR: --tenant must be a UUID, got {args.tenant!r}", file=sys.stderr)
        return 2

    async with async_session() as s:
        existing = (
            await s.execute(select(Tenant).where(Tenant.id == args.tenant))
        ).scalar_one_or_none()
        if existing is not None:
            print(f"tenant already exists: {existing.id} ({existing.name})")
            return 0

        t = Tenant(
            id=args.tenant,
            external_team_id=args.external_team_id,
            channel=args.channel,
            name=args.name,
        )
        s.add(t)
        await s.commit()
        print(f"created tenant: {t.id} ({t.name})")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
