"""Sanity-check the DATABASE_URL — connects to Postgres and prints version().

Usage:
    .venv/bin/python scripts/db_check.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

load_dotenv()


async def main() -> None:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL is not set in .env", file=sys.stderr)
        sys.exit(1)
    engine = create_async_engine(url)
    async with engine.connect() as conn:
        row = await conn.execute(text("select version()"))
        print("OK:", row.scalar())
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
