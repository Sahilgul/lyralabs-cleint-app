"""Quick DB connection test.

Usage:
    python scripts/test_db_connection.py
    python scripts/test_db_connection.py "postgresql+psycopg://user:pass@host:5432/postgres"
"""

import sys
import sqlalchemy as sa

url = sys.argv[1] if len(sys.argv) > 1 else None

if url is None:
    import os, sys
    sys.path.insert(0, "packages")
    from dotenv import load_dotenv
    load_dotenv()
    url = os.environ.get("DATABASE_URL_SYNC")
    if not url:
        print("ERROR: DATABASE_URL_SYNC not set and no URL passed as argument")
        sys.exit(1)

# Mask password for display
try:
    from urllib.parse import urlparse
    p = urlparse(url.replace("+psycopg", "").replace("+asyncpg", ""))
    display = url.replace(p.password or "", "****") if p.password else url
except Exception:
    display = url

print(f"Testing connection to: {display}")

try:
    # Force sync driver
    sync_url = url.replace("postgresql+asyncpg", "postgresql+psycopg")
    engine = sa.create_engine(sync_url, connect_args={"connect_timeout": 10})
    with engine.connect() as conn:
        version = conn.execute(sa.text("SELECT version()")).scalar()
        print(f"\n✓ Connected successfully!")
        print(f"  Postgres: {version.split(',')[0]}")
except Exception as e:
    print(f"\n✗ Connection failed: {e}")
    sys.exit(1)
