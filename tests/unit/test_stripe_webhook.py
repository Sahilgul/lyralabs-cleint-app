"""apps.api.stripe_webhook."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from apps.api import stripe_webhook as webhook_mod
from lyra_core.db.models import Tenant


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(webhook_mod.router, prefix="/webhooks")
    return app


def _tenant(customer_id: str = "cus_1") -> Tenant:
    t = Tenant(external_team_id="T1", channel="slack", name="Acme")
    t.id = "t-1"
    t.stripe_customer_id = customer_id
    t.plan = "trial"
    t.status = "active"
    return t


def _patch_construct_event(monkeypatch, event: dict) -> None:
    monkeypatch.setattr(
        webhook_mod.stripe.Webhook,
        "construct_event",
        lambda payload, sig_header, secret: event,
    )


def _patch_session(monkeypatch, tenant: Tenant | None) -> dict:
    captured = {"committed": False}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _stmt):
            r = MagicMock()
            r.scalar_one_or_none.return_value = tenant
            return r

        async def commit(self):
            captured["committed"] = True

    monkeypatch.setattr(webhook_mod, "async_session", FakeSession)
    return captured


@pytest.mark.asyncio
async def test_invalid_signature_returns_400(monkeypatch) -> None:
    def boom(**kw):
        raise ValueError("bad sig")

    monkeypatch.setattr(webhook_mod.stripe.Webhook, "construct_event", boom)
    app = _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/webhooks/stripe", content=b"{}", headers={"stripe-signature": "x"})
    assert r.status_code == 400
    assert "signature" in r.text


@pytest.mark.asyncio
async def test_missing_signature_header_returns_422() -> None:
    app = _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/webhooks/stripe", content=b"{}")
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_subscription_created_active_marks_team_active(monkeypatch) -> None:
    t = _tenant()
    _patch_session(monkeypatch, t)
    _patch_construct_event(
        monkeypatch,
        {
            "type": "customer.subscription.created",
            "data": {"object": {"customer": "cus_1", "id": "sub_1", "status": "active"}},
        },
    )

    app = _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/webhooks/stripe", content=b"{}", headers={"stripe-signature": "x"})

    assert r.status_code == 200
    assert r.json() == {"received": True}
    assert t.plan == "team"
    assert t.status == "active"
    assert t.stripe_subscription_id == "sub_1"


@pytest.mark.asyncio
async def test_subscription_updated_past_due(monkeypatch) -> None:
    t = _tenant()
    _patch_session(monkeypatch, t)
    _patch_construct_event(
        monkeypatch,
        {
            "type": "customer.subscription.updated",
            "data": {"object": {"customer": "cus_1", "id": "sub_1", "status": "past_due"}},
        },
    )

    app = _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/webhooks/stripe", content=b"{}", headers={"stripe-signature": "x"})
    assert r.status_code == 200
    assert t.status == "past_due"
    assert t.plan == "cancelled"


@pytest.mark.asyncio
async def test_subscription_deleted_cancels(monkeypatch) -> None:
    t = _tenant()
    _patch_session(monkeypatch, t)
    _patch_construct_event(
        monkeypatch,
        {
            "type": "customer.subscription.deleted",
            "data": {"object": {"customer": "cus_1", "id": "sub_1", "status": "canceled"}},
        },
    )
    app = _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/webhooks/stripe", content=b"{}", headers={"stripe-signature": "x"})
    assert r.status_code == 200
    assert t.plan == "cancelled"
    assert t.status == "cancelled"


@pytest.mark.asyncio
async def test_invoice_payment_failed_marks_past_due(monkeypatch) -> None:
    t = _tenant()
    _patch_session(monkeypatch, t)
    _patch_construct_event(
        monkeypatch,
        {
            "type": "invoice.payment_failed",
            "data": {"object": {"customer": "cus_1", "id": "in_1"}},
        },
    )
    app = _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/webhooks/stripe", content=b"{}", headers={"stripe-signature": "x"})
    assert r.status_code == 200
    assert t.status == "past_due"


@pytest.mark.asyncio
async def test_unknown_tenant_no_op(monkeypatch) -> None:
    captured = _patch_session(monkeypatch, None)
    _patch_construct_event(
        monkeypatch,
        {
            "type": "customer.subscription.created",
            "data": {"object": {"customer": "cus_unknown", "id": "sub_1", "status": "active"}},
        },
    )
    app = _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/webhooks/stripe", content=b"{}", headers={"stripe-signature": "x"})
    assert r.status_code == 200
    # No tenant to mutate; commit still runs at end
    assert captured["committed"] is True


@pytest.mark.asyncio
async def test_other_events_ignored(monkeypatch) -> None:
    t = _tenant()
    _patch_session(monkeypatch, t)
    _patch_construct_event(
        monkeypatch,
        {"type": "checkout.session.completed", "data": {"object": {"id": "cs_1"}}},
    )
    app = _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/webhooks/stripe", content=b"{}", headers={"stripe-signature": "x"})
    assert r.status_code == 200
    # plan/status unchanged
    assert t.plan == "trial"
    assert t.status == "active"
