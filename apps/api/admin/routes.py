"""Admin REST API consumed by the Next.js panel.

Scoped to the caller's tenant via the JWT `tenant_id` claim.
"""

from __future__ import annotations

from typing import Annotated

import stripe
from fastapi import APIRouter, Depends, HTTPException
from lyra_core.common.config import get_settings
from lyra_core.db.models import (
    AuditEvent,
    IntegrationConnection,
    Job,
    Tenant,
)
from lyra_core.db.session import get_session
from pydantic import BaseModel
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .auth import CurrentAdmin

router = APIRouter()


# --- DTOs ---------------------------------------------------------------------


class TenantOut(BaseModel):
    id: str
    external_team_id: str
    channel: str
    name: str
    plan: str
    status: str
    trial_credit_remaining_usd: float
    stripe_customer_id: str | None
    stripe_subscription_id: str | None


class IntegrationOut(BaseModel):
    id: str
    provider: str
    external_account_id: str
    display_name: str | None
    status: str
    scopes: str


class JobOut(BaseModel):
    id: str
    thread_id: str
    user_id: str | None
    user_request: str
    status: str
    result_summary: str | None
    cost_usd: float
    created_at: str


class AuditOut(BaseModel):
    id: int
    event_type: str
    tool_name: str | None
    result_status: str
    cost_usd: float
    ts: str


class CostSummary(BaseModel):
    total_usd: float
    by_model: dict[str, float]
    n_events: int


# --- Endpoints ----------------------------------------------------------------


@router.get("/me", response_model=TenantOut)
async def me(
    admin: CurrentAdmin,
    s: Annotated[AsyncSession, Depends(get_session)],
) -> TenantOut:
    t = (await s.execute(select(Tenant).where(Tenant.id == admin.tenant_id))).scalar_one()
    return TenantOut.model_validate(t.__dict__)


@router.get("/integrations", response_model=list[IntegrationOut])
async def list_integrations(
    admin: CurrentAdmin,
    s: Annotated[AsyncSession, Depends(get_session)],
) -> list[IntegrationOut]:
    rows = (
        (
            await s.execute(
                select(IntegrationConnection)
                .where(IntegrationConnection.tenant_id == admin.tenant_id)
                .order_by(IntegrationConnection.provider)
            )
        )
        .scalars()
        .all()
    )
    return [IntegrationOut.model_validate(r.__dict__) for r in rows]


@router.delete("/integrations/{integration_id}")
async def delete_integration(
    integration_id: str,
    admin: CurrentAdmin,
    s: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, str]:
    row = (
        await s.execute(
            select(IntegrationConnection).where(
                IntegrationConnection.id == integration_id,
                IntegrationConnection.tenant_id == admin.tenant_id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(404, "integration not found")
    await s.delete(row)
    return {"status": "deleted"}


@router.get("/jobs", response_model=list[JobOut])
async def list_jobs(
    admin: CurrentAdmin,
    s: Annotated[AsyncSession, Depends(get_session)],
    limit: int = 50,
) -> list[JobOut]:
    rows = (
        (
            await s.execute(
                select(Job)
                .where(Job.tenant_id == admin.tenant_id)
                .order_by(desc(Job.created_at))
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return [
        JobOut(
            id=r.id,
            thread_id=r.thread_id,
            user_id=r.user_id,
            user_request=r.user_request,
            status=r.status,
            result_summary=r.result_summary,
            cost_usd=r.cost_usd,
            created_at=r.created_at.isoformat(),
        )
        for r in rows
    ]


@router.get("/audit", response_model=list[AuditOut])
async def list_audit(
    admin: CurrentAdmin,
    s: Annotated[AsyncSession, Depends(get_session)],
    limit: int = 200,
) -> list[AuditOut]:
    rows = (
        (
            await s.execute(
                select(AuditEvent)
                .where(AuditEvent.tenant_id == admin.tenant_id)
                .order_by(desc(AuditEvent.ts))
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return [
        AuditOut(
            id=r.id,
            event_type=r.event_type,
            tool_name=r.tool_name,
            result_status=r.result_status,
            cost_usd=r.cost_usd,
            ts=r.ts.isoformat(),
        )
        for r in rows
    ]


@router.get("/cost", response_model=CostSummary)
async def cost_summary(
    admin: CurrentAdmin,
    s: Annotated[AsyncSession, Depends(get_session)],
) -> CostSummary:
    total = (
        await s.execute(
            select(func.coalesce(func.sum(AuditEvent.cost_usd), 0.0)).where(
                AuditEvent.tenant_id == admin.tenant_id
            )
        )
    ).scalar_one()
    n = (
        await s.execute(
            select(func.count(AuditEvent.id)).where(AuditEvent.tenant_id == admin.tenant_id)
        )
    ).scalar_one()
    by_model_rows = (
        await s.execute(
            select(AuditEvent.model_used, func.sum(AuditEvent.cost_usd))
            .where(AuditEvent.tenant_id == admin.tenant_id)
            .group_by(AuditEvent.model_used)
        )
    ).all()
    by_model = {(m or "unknown"): float(v or 0.0) for m, v in by_model_rows}
    return CostSummary(total_usd=float(total), by_model=by_model, n_events=int(n))


# --- Stripe Customer Portal ---------------------------------------------------


@router.post("/billing/portal")
async def billing_portal(
    admin: CurrentAdmin,
    s: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, str]:
    settings = get_settings()
    stripe.api_key = settings.stripe_secret_key

    t = (await s.execute(select(Tenant).where(Tenant.id == admin.tenant_id))).scalar_one()
    if not t.stripe_customer_id:
        cust = stripe.Customer.create(
            name=t.name, metadata={"tenant_id": t.id, "email": admin.email}
        )
        t.stripe_customer_id = cust.id
        await s.flush()

    portal = stripe.billing_portal.Session.create(
        customer=t.stripe_customer_id,
        return_url=f"{settings.admin_base_url}/billing",
    )
    return {"url": portal.url}


@router.post("/billing/checkout")
async def checkout(
    admin: CurrentAdmin,
    s: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, str]:
    settings = get_settings()
    stripe.api_key = settings.stripe_secret_key

    t = (await s.execute(select(Tenant).where(Tenant.id == admin.tenant_id))).scalar_one()
    if not t.stripe_customer_id:
        cust = stripe.Customer.create(
            name=t.name, metadata={"tenant_id": t.id, "email": admin.email}
        )
        t.stripe_customer_id = cust.id
        await s.flush()

    sess = stripe.checkout.Session.create(
        mode="subscription",
        customer=t.stripe_customer_id,
        line_items=[{"price": settings.stripe_price_id_team_monthly, "quantity": 1}],
        success_url=f"{settings.admin_base_url}/billing?status=success",
        cancel_url=f"{settings.admin_base_url}/billing?status=cancelled",
        subscription_data={"metadata": {"tenant_id": t.id}},
    )
    return {"url": sess.url or ""}
