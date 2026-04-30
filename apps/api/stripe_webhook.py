"""Stripe billing webhook.

Subscribes to:
  - customer.subscription.created
  - customer.subscription.updated
  - customer.subscription.deleted
  - invoice.payment_succeeded
  - invoice.payment_failed

Maps Stripe events onto Tenant.plan / Tenant.status.
"""

from __future__ import annotations

import stripe
from fastapi import APIRouter, Header, HTTPException, Request
from sqlalchemy import select

from lyra_core.common.config import get_settings
from lyra_core.common.logging import get_logger
from lyra_core.db.models import Tenant
from lyra_core.db.session import async_session

router = APIRouter()
log = get_logger(__name__)


@router.post("/stripe")
async def stripe_webhook(req: Request, stripe_signature: str = Header(...)):
    settings = get_settings()
    payload = await req.body()
    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=stripe_signature,
            secret=settings.stripe_webhook_secret,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"signature verification failed: {exc}")

    etype = event["type"]
    obj = event["data"]["object"]
    log.info("stripe.event", type=etype, id=obj.get("id"))

    async with async_session() as s:
        if etype in {"customer.subscription.created", "customer.subscription.updated"}:
            customer_id = obj["customer"]
            sub_id = obj["id"]
            status = obj["status"]
            tenant = (
                await s.execute(select(Tenant).where(Tenant.stripe_customer_id == customer_id))
            ).scalar_one_or_none()
            if tenant:
                tenant.stripe_subscription_id = sub_id
                tenant.plan = "team" if status in {"active", "trialing"} else "cancelled"
                tenant.status = (
                    "active"
                    if status in {"active", "trialing"}
                    else ("past_due" if status == "past_due" else "cancelled")
                )

        elif etype == "customer.subscription.deleted":
            customer_id = obj["customer"]
            tenant = (
                await s.execute(select(Tenant).where(Tenant.stripe_customer_id == customer_id))
            ).scalar_one_or_none()
            if tenant:
                tenant.plan = "cancelled"
                tenant.status = "cancelled"

        elif etype == "invoice.payment_failed":
            customer_id = obj["customer"]
            tenant = (
                await s.execute(select(Tenant).where(Tenant.stripe_customer_id == customer_id))
            ).scalar_one_or_none()
            if tenant:
                tenant.status = "past_due"

        await s.commit()

    return {"received": True}
