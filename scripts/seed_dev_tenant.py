#!/usr/bin/env python
"""Dev-only: seed a tenant + admin user + a handful of jobs/audit events
so the admin UI has real data to render on the first boot.

Idempotent: re-running will reuse the same external_team_id ("dev-workspace")
and only insert sample jobs/audit rows if the tenant has none yet.

Usage:
    PYTHONPATH=packages:. .venv/bin/python scripts/seed_dev_tenant.py

Prints the tenant_id, the admin email, and a 30-day JWT you can paste into
the Vite admin UI's login screen.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import UTC, datetime, timedelta

import jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# Local import path: scripts/ sits next to packages/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from lyra_core.common.config import get_settings  # noqa: E402
from lyra_core.db.models import (  # noqa: E402
    AuditEvent,
    IntegrationConnection,
    Job,
    Tenant,
    User,
)


DEV_TEAM_ID = "T_DEV_WORKSPACE"
DEV_WORKSPACE_NAME = "Sahil's Dev Workspace"
DEV_ADMIN_EMAIL = "sahil@dev.local"


async def main() -> None:
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    async with Session() as s:
        existing = (
            await s.execute(select(Tenant).where(Tenant.external_team_id == DEV_TEAM_ID))
        ).scalar_one_or_none()

        if existing:
            tenant = existing
            print(f"reusing tenant id={tenant.id} name={tenant.name!r}")
        else:
            tenant = Tenant(
                external_team_id=DEV_TEAM_ID,
                channel="slack",
                name=DEV_WORKSPACE_NAME,
                plan="trial",
                status="active",
                trial_credit_remaining_usd=87.42,
                settings={"facts": {"timezone": "Asia/Karachi", "default_drive_folder": None}},
            )
            s.add(tenant)
            await s.flush()
            print(f"created tenant id={tenant.id}")

        admin = (
            await s.execute(
                select(User).where(
                    User.tenant_id == tenant.id,
                    User.email == DEV_ADMIN_EMAIL,
                )
            )
        ).scalar_one_or_none()
        if not admin:
            admin = User(
                tenant_id=tenant.id,
                external_user_id="dev-admin",
                channel="admin",
                display_name="Sahil (dev)",
                email=DEV_ADMIN_EMAIL,
                role="owner",
            )
            s.add(admin)
            await s.flush()
            print(f"created admin user id={admin.id}")
        else:
            print(f"reusing admin user id={admin.id}")

        # Sample jobs only if the tenant has none.
        existing_jobs = (
            await s.execute(select(Job).where(Job.tenant_id == tenant.id).limit(1))
        ).scalar_one_or_none()
        if not existing_jobs:
            now = datetime.now(UTC)
            jobs = [
                Job(
                    tenant_id=tenant.id,
                    thread_id="thread-001",
                    user_id="U_DEV_USER",
                    channel_id="C_DEV_CHAN",
                    user_request="Pull last 7 days of GHL leads into a Google Sheet",
                    status="done",
                    result_summary="Created sheet 'Leads — last 7 days' with 142 rows",
                    artifact_urls=["https://docs.google.com/spreadsheets/d/EXAMPLE_ID"],
                    cost_usd=0.0421,
                    created_at=now - timedelta(hours=4),
                    updated_at=now - timedelta(hours=4),
                ),
                Job(
                    tenant_id=tenant.id,
                    thread_id="thread-002",
                    user_id="U_DEV_USER",
                    channel_id="C_DEV_CHAN",
                    user_request="Draft a follow-up email for stale opportunities",
                    status="running",
                    cost_usd=0.0083,
                    created_at=now - timedelta(minutes=3),
                    updated_at=now - timedelta(minutes=1),
                ),
                Job(
                    tenant_id=tenant.id,
                    thread_id="thread-003",
                    user_id="U_DEV_USER",
                    channel_id="C_DEV_CHAN",
                    user_request="Schedule a meeting with @maria for tomorrow 3pm",
                    status="awaiting_approval",
                    plan_json={
                        "steps": [
                            {"tool": "google.calendar.create_event", "args": {"title": "..."}}
                        ]
                    },
                    cost_usd=0.0017,
                    created_at=now - timedelta(minutes=12),
                    updated_at=now - timedelta(minutes=10),
                ),
                Job(
                    tenant_id=tenant.id,
                    thread_id="thread-004",
                    user_id="U_DEV_USER",
                    channel_id="C_DEV_CHAN",
                    user_request="Sync GHL pipeline to a Google Doc weekly summary",
                    status="failed",
                    error="GHL token expired — reconnect required",
                    cost_usd=0.0009,
                    created_at=now - timedelta(hours=18),
                    updated_at=now - timedelta(hours=18),
                ),
            ]
            s.add_all(jobs)
            await s.flush()

            audit_rows = [
                AuditEvent(
                    tenant_id=tenant.id,
                    actor_user_id=admin.external_user_id,
                    job_id=jobs[0].id,
                    event_type="tool_call",
                    tool_name="ghl.contacts.list",
                    cost_usd=0.0,
                    model_used="dashscope/qwen-turbo",
                    extra={"count": 142},
                ),
                AuditEvent(
                    tenant_id=tenant.id,
                    actor_user_id=admin.external_user_id,
                    job_id=jobs[0].id,
                    event_type="tool_call",
                    tool_name="google.sheets.create",
                    cost_usd=0.0,
                    model_used="dashscope/qwen-max",
                    extra={"rows": 142, "sheet_id": "EXAMPLE_ID"},
                ),
                AuditEvent(
                    tenant_id=tenant.id,
                    actor_user_id=admin.external_user_id,
                    job_id=jobs[0].id,
                    event_type="llm_call",
                    cost_usd=0.0421,
                    model_used="anthropic/claude-sonnet-4-5",
                    extra={"prompt_tokens": 1820, "completion_tokens": 412},
                ),
                AuditEvent(
                    tenant_id=tenant.id,
                    actor_user_id=admin.external_user_id,
                    job_id=jobs[3].id,
                    event_type="error",
                    cost_usd=0.0009,
                    model_used="gemini/gemini-2.5-flash",
                    extra={"error": "GHL 401 unauthorized"},
                    result_status="error",
                ),
            ]
            s.add_all(audit_rows)
            print(f"inserted {len(jobs)} sample jobs and {len(audit_rows)} audit events")
        else:
            print("tenant already has jobs — skipping sample data")

        # A "connected" Google integration so the Integrations page shows
        # something other than the empty state. Token field is a placeholder
        # — the agent will fail to *use* it, but the UI list renders fine.
        existing_int = (
            await s.execute(
                select(IntegrationConnection).where(
                    IntegrationConnection.tenant_id == tenant.id,
                    IntegrationConnection.provider == "google",
                )
            )
        ).scalar_one_or_none()
        if not existing_int:
            s.add(
                IntegrationConnection(
                    tenant_id=tenant.id,
                    provider="google",
                    external_account_id="dev-google-account",
                    display_name="dev@workspace.com",
                    scopes="drive,docs,sheets,calendar",
                    access_token_encrypted="placeholder-not-real",
                    status="active",
                    metadata_={"display_email": "dev@workspace.com"},
                )
            )
            print("inserted sample Google integration (placeholder token)")

        await s.commit()
        await engine.dispose()

    secret = settings.admin_jwt_secret
    if not secret:
        print("\nWARN: ADMIN_JWT_SECRET is not set in .env, skipping JWT mint.")
        return

    payload = {
        "tenant_id": tenant.id,
        "email": DEV_ADMIN_EMAIL,
        "role": "owner",
        "exp": int(time.time()) + 30 * 24 * 3600,
        "iss": settings.admin_jwt_issuer,
    }
    token = jwt.encode(payload, secret, algorithm="HS256")
    print()
    print("=" * 70)
    print("Admin JWT (valid 30 days) — paste into the Vite admin UI's login:")
    print("=" * 70)
    print(token)
    print()
    print(f"tenant_id : {tenant.id}")
    print(f"email     : {DEV_ADMIN_EMAIL}")


if __name__ == "__main__":
    asyncio.run(main())
