#!/usr/bin/env python
"""Dev-only helper: mint an admin JWT for a tenant.

Usage:
    python scripts/mint_admin_jwt.py --tenant <UUID> --email you@example.com

Reads ADMIN_JWT_SECRET (required) and ADMIN_JWT_ISSUER (default
"lyralabs-admin") directly from the environment. Does NOT instantiate the
full Settings object, so you can mint a token without configuring Postgres /
the master encryption key. Loads .env if present.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import jwt


def _load_dotenv() -> None:
    """Tiny, dependency-free .env loader."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def main() -> None:
    _load_dotenv()

    p = argparse.ArgumentParser()
    p.add_argument("--tenant", required=True)
    p.add_argument("--email", required=True)
    p.add_argument("--role", default="owner")
    p.add_argument("--ttl", type=int, default=24 * 3600)
    args = p.parse_args()

    secret = os.getenv("ADMIN_JWT_SECRET")
    if not secret:
        print(
            "ERROR: ADMIN_JWT_SECRET is not set. Add it to .env or export it.",
            file=sys.stderr,
        )
        sys.exit(1)
    issuer = os.getenv("ADMIN_JWT_ISSUER", "lyralabs-admin")

    payload = {
        "tenant_id": args.tenant,
        "email": args.email,
        "role": args.role,
        "exp": int(time.time()) + args.ttl,
        "iss": issuer,
    }
    print(jwt.encode(payload, secret, algorithm="HS256"))


if __name__ == "__main__":
    main()
