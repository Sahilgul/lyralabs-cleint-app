"""apps.api.admin.routes — uses dependency overrides + a mocked AsyncSession."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from apps.api.admin.auth import AdminPrincipal, current_admin
from apps.api.admin.routes import router as admin_router
from lyra_core.db.models import AuditEvent, IntegrationConnection, Job, Tenant
from lyra_core.db.session import get_session


def _build(session: MagicMock, principal: AdminPrincipal) -> FastAPI:
    app = FastAPI()
    app.include_router(admin_router, prefix="/admin")

    async def override_session():
        yield session

    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[current_admin] = lambda: principal
    return app


def _make_session() -> MagicMock:
    s = MagicMock()
    s.execute = AsyncMock()
    s.commit = AsyncMock()
    s.flush = AsyncMock()
    s.delete = AsyncMock()
    return s


def _admin(tenant_id: str = "t-1") -> AdminPrincipal:
    return AdminPrincipal(tenant_id=tenant_id, email="a@x.com", role="owner")


def _tenant() -> Tenant:
    t = Tenant(external_team_id="T1", channel="slack", name="Acme")
    t.id = "t-1"
    t.plan = "trial"
    t.status = "active"
    t.trial_credit_remaining_usd = 100.0
    t.stripe_customer_id = None
    t.stripe_subscription_id = None
    return t


@pytest.mark.asyncio
async def test_me_returns_tenant_info() -> None:
    s = _make_session()
    r = MagicMock()
    r.scalar_one.return_value = _tenant()
    s.execute.return_value = r

    app = _build(s, _admin())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get("/admin/me")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "t-1"
    assert body["channel"] == "slack"
    assert body["plan"] == "trial"


@pytest.mark.asyncio
async def test_list_integrations() -> None:
    s = _make_session()
    integ = IntegrationConnection(
        tenant_id="t-1",
        provider="google",
        external_account_id="user@x.com",
        scopes="scope.a",
        access_token_encrypted="ct",
    )
    integ.id = "int-1"
    integ.display_name = "Google"
    integ.status = "active"

    scalars = MagicMock()
    scalars.all.return_value = [integ]
    res = MagicMock()
    res.scalars.return_value = scalars
    s.execute.return_value = res

    app = _build(s, _admin())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/admin/integrations")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["provider"] == "google"
    assert body[0]["display_name"] == "Google"


@pytest.mark.asyncio
async def test_delete_integration_404_when_missing() -> None:
    s = _make_session()
    res = MagicMock()
    res.scalar_one_or_none.return_value = None
    s.execute.return_value = res

    app = _build(s, _admin())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.delete("/admin/integrations/abc")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_delete_integration_success() -> None:
    s = _make_session()
    integ = IntegrationConnection(
        tenant_id="t-1", provider="google", external_account_id="x", access_token_encrypted="x"
    )
    integ.id = "int-1"
    res = MagicMock()
    res.scalar_one_or_none.return_value = integ
    s.execute.return_value = res

    app = _build(s, _admin())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.delete("/admin/integrations/int-1")
    assert r.status_code == 200
    assert r.json() == {"status": "deleted"}
    s.delete.assert_awaited_once_with(integ)


@pytest.mark.asyncio
async def test_list_jobs_returns_serialized_rows() -> None:
    s = _make_session()
    j = Job(
        tenant_id="t-1",
        thread_id="thr",
        user_id="u-1",
        user_request="do x",
        status="done",
    )
    j.id = "j-1"
    j.result_summary = "ok"
    j.cost_usd = 0.0123
    j.created_at = datetime(2026, 5, 1, 12, tzinfo=UTC)

    scalars = MagicMock()
    scalars.all.return_value = [j]
    res = MagicMock()
    res.scalars.return_value = scalars
    s.execute.return_value = res

    app = _build(s, _admin())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/admin/jobs?limit=10")
    assert r.status_code == 200
    body = r.json()
    assert body[0]["id"] == "j-1"
    assert body[0]["cost_usd"] == 0.0123
    assert body[0]["created_at"].startswith("2026-05-01")


@pytest.mark.asyncio
async def test_list_audit_returns_rows() -> None:
    s = _make_session()
    e = AuditEvent(tenant_id="t-1", event_type="tool_call")
    e.id = 1
    e.tool_name = "google.drive.search"
    e.result_status = "ok"
    e.cost_usd = 0.0
    e.ts = datetime(2026, 5, 1, tzinfo=UTC)

    scalars = MagicMock()
    scalars.all.return_value = [e]
    res = MagicMock()
    res.scalars.return_value = scalars
    s.execute.return_value = res

    app = _build(s, _admin())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/admin/audit")
    assert r.status_code == 200
    body = r.json()
    assert body[0]["tool_name"] == "google.drive.search"


@pytest.mark.asyncio
async def test_cost_summary_aggregates() -> None:
    s = _make_session()
    # Three execute calls in cost_summary: total, count, group-by
    total_res = MagicMock(); total_res.scalar_one.return_value = 0.4242
    count_res = MagicMock(); count_res.scalar_one.return_value = 7
    group_res = MagicMock()
    group_res.all.return_value = [
        ("anthropic/claude-sonnet-4-5", 0.3),
        (None, 0.1242),
    ]
    s.execute = AsyncMock(side_effect=[total_res, count_res, group_res])

    app = _build(s, _admin())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/admin/cost")
    assert r.status_code == 200
    body = r.json()
    assert body["total_usd"] == pytest.approx(0.4242)
    assert body["n_events"] == 7
    assert "anthropic/claude-sonnet-4-5" in body["by_model"]
    assert "unknown" in body["by_model"]


@pytest.mark.asyncio
async def test_billing_portal_creates_customer_when_missing(monkeypatch) -> None:
    import apps.api.admin.routes as routes

    s = _make_session()
    t = _tenant()  # stripe_customer_id is None
    res = MagicMock(); res.scalar_one.return_value = t
    s.execute.return_value = res

    fake_cust = MagicMock(); fake_cust.id = "cus_X"
    monkeypatch.setattr(routes.stripe.Customer, "create", lambda **kw: fake_cust)
    fake_portal = MagicMock(); fake_portal.url = "https://billing.example/portal/abc"
    monkeypatch.setattr(routes.stripe.billing_portal.Session, "create", lambda **kw: fake_portal)

    app = _build(s, _admin())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/admin/billing/portal")

    assert r.status_code == 200
    assert r.json() == {"url": "https://billing.example/portal/abc"}
    assert t.stripe_customer_id == "cus_X"


@pytest.mark.asyncio
async def test_checkout_uses_existing_customer(monkeypatch) -> None:
    import apps.api.admin.routes as routes

    s = _make_session()
    t = _tenant()
    t.stripe_customer_id = "cus_existing"
    res = MagicMock(); res.scalar_one.return_value = t
    s.execute.return_value = res

    fake_sess = MagicMock(); fake_sess.url = "https://checkout.example/sess/x"
    monkeypatch.setattr(routes.stripe.checkout.Session, "create", lambda **kw: fake_sess)

    customer_create_called = False

    def _create(**kw):
        nonlocal customer_create_called
        customer_create_called = True

    monkeypatch.setattr(routes.stripe.Customer, "create", _create)

    app = _build(s, _admin())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/admin/billing/checkout")

    assert r.status_code == 200
    assert r.json()["url"] == "https://checkout.example/sess/x"
    assert customer_create_called is False  # didn't recreate
