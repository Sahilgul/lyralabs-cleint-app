"""apps.api.admin.llm -- super-admin model-switcher REST API.

Mirrors test_admin_routes.py: build a FastAPI app, override `get_session`
+ `current_super_admin` dependencies, exercise the endpoints. The router
cache is invalidated in autouse so each test starts clean.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from apps.api.admin.auth import AdminPrincipal, current_super_admin
from apps.api.admin.llm import router as llm_router
from lyra_core.common.crypto import encrypt_platform
from lyra_core.db.models import LlmModelAssignment, LlmProvider
from lyra_core.db.session import get_session
from lyra_core.llm import router as router_mod


@pytest.fixture(autouse=True)
def _reset_router_cache():
    router_mod.invalidate_router_cache()
    yield
    router_mod.invalidate_router_cache()


def _build(session: MagicMock) -> FastAPI:
    app = FastAPI()
    app.include_router(llm_router, prefix="/admin/llm")

    async def override_session():
        yield session

    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[current_super_admin] = lambda: AdminPrincipal(
        tenant_id="t-1", email="boss@platform.com", role="super_admin"
    )
    return app


def _make_session() -> MagicMock:
    s = MagicMock()
    s.execute = AsyncMock()
    s.commit = AsyncMock()
    s.flush = AsyncMock()
    s.refresh = AsyncMock()
    s.delete = AsyncMock()
    s.add = MagicMock()
    return s


def _provider_row(
    key: str = "qwen", api_key: str | None = "sk-real", enabled: bool = True
) -> LlmProvider:
    p = LlmProvider(
        provider_key=key,
        api_key_encrypted=encrypt_platform(api_key) if api_key else None,
        api_base=None,
        extra_config={},
        enabled=enabled,
    )
    p.id = f"prov-{key}"
    p.created_at = datetime.now(UTC)
    p.updated_at = datetime.now(UTC)
    return p


def _scalar_result(value):
    r = MagicMock()
    r.scalar_one_or_none = MagicMock(return_value=value)
    return r


def _scalars_result(values):
    r = MagicMock()
    scalars = MagicMock()
    scalars.all = MagicMock(return_value=values)
    r.scalars = MagicMock(return_value=scalars)
    return r


# --- catalog ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_catalog_lists_qwen_and_deepseek() -> None:
    s = _make_session()
    app = _build(s)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        r = await ac.get("/admin/llm/catalog")
    assert r.status_code == 200
    keys = {p["key"] for p in r.json()}
    assert "qwen" in keys
    assert "deepseek" in keys


@pytest.mark.asyncio
async def test_catalog_never_includes_secrets() -> None:
    """Sanity: the catalog is pure metadata. If we ever start joining DB
    state into it, this test will fail and force us to stay disciplined."""
    s = _make_session()
    app = _build(s)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        r = await ac.get("/admin/llm/catalog")
    body = r.text.lower()
    assert "api_key" not in body
    assert "encrypted" not in body


# --- providers ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_providers_merges_catalog_with_db_state(monkeypatch) -> None:
    """list_configured_providers() is the right unit to exercise here -- the
    HTTP route just wraps it. Patch async_session in the router module."""

    class _LiveSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _stmt):
            return _scalars_result([_provider_row(key="qwen", api_key="sk-real")])

    monkeypatch.setattr(router_mod, "async_session", lambda: _LiveSession())

    s = _make_session()
    app = _build(s)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        r = await ac.get("/admin/llm/providers")

    assert r.status_code == 200
    payload = {p["key"]: p for p in r.json()}
    assert payload["qwen"]["configured"] is True
    assert payload["qwen"]["has_api_key"] is True
    # DB-less providers still appear, just with configured=False
    assert payload["openai"]["configured"] is False
    assert payload["openai"]["has_api_key"] is False
    # No raw key field or value anywhere in the response. (Substring "api_key"
    # alone would match "has_api_key", so check the JSON field name and the
    # plaintext secret explicitly.)
    assert '"api_key"' not in r.text
    assert "sk-real" not in r.text


@pytest.mark.asyncio
async def test_upsert_provider_encrypts_key_and_invalidates_cache() -> None:
    s = _make_session()
    # No existing row for this provider.
    s.execute.side_effect = [
        _scalar_result(None),  # initial select in upsert
        # second select happens via list_configured_providers; see live session below
    ]

    # Patch list_configured_providers (it would otherwise hit a real DB).
    import apps.api.admin.llm as llm_admin

    async def fake_list():
        return [
            {
                "key": "deepseek",
                "display_name": "DeepSeek",
                "litellm_prefix": "deepseek",
                "default_api_base": "https://api.deepseek.com/v1",
                "docs_url": "x",
                "extra_config_keys": [],
                "known_models": [],
                "configured": True,
                "enabled": True,
                "has_api_key": True,
                "api_base": None,
                "extra_config": {},
                "last_tested_at": None,
                "last_test_status": None,
                "last_test_error": None,
                "updated_by_email": "boss@platform.com",
            }
        ]

    invalidations = {"n": 0}

    def _spy_invalidate():
        invalidations["n"] += 1

    app = _build(s)
    # Patch in the admin module (where it's bound) -- this is the live import path.
    app.dependency_overrides  # smoke
    import apps.api.admin.llm as admin_mod

    monkey_set = []

    def _patch_attr(obj, name, val):
        monkey_set.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)

    _patch_attr(admin_mod, "list_configured_providers", fake_list)
    _patch_attr(admin_mod, "invalidate_router_cache", _spy_invalidate)

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t"
        ) as ac:
            r = await ac.put(
                "/admin/llm/providers/deepseek",
                json={"api_key": "ds-secret", "enabled": True, "extra_config": {}},
            )
    finally:
        for obj, name, orig in monkey_set:
            setattr(obj, name, orig)

    assert r.status_code == 200, r.text
    assert s.add.called  # row was inserted
    added: LlmProvider = s.add.call_args.args[0]
    assert added.provider_key == "deepseek"
    # The plaintext key is NOT stored on the row -- only the ciphertext is.
    assert added.api_key_encrypted is not None
    assert "ds-secret" not in (added.api_key_encrypted or "")
    assert s.commit.called
    assert invalidations["n"] == 1


@pytest.mark.asyncio
async def test_upsert_rejects_unknown_provider() -> None:
    s = _make_session()
    app = _build(s)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        r = await ac.put(
            "/admin/llm/providers/fake_provider",
            json={"api_key": "x"},
        )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_delete_provider_blocked_when_assigned() -> None:
    """Refusing a delete that would orphan an assignment forces the operator
    to flip the tier first -- prevents a silent fall-through to env."""
    s = _make_session()
    s.execute.side_effect = [
        _scalar_result(_provider_row(key="qwen")),  # SELECT row
        _scalars_result(["primary"]),  # SELECT in_use tiers
    ]
    app = _build(s)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        r = await ac.delete("/admin/llm/providers/qwen")
    assert r.status_code == 409
    assert not s.delete.called


# --- assignments --------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_assignment_requires_configured_provider() -> None:
    s = _make_session()
    s.execute.side_effect = [_scalar_result(None)]  # provider not configured
    app = _build(s)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        r = await ac.put(
            "/admin/llm/active/primary",
            json={"provider_key": "deepseek", "model_id": "deepseek/deepseek-chat"},
        )
    assert r.status_code == 409
    assert "must be configured" in r.text.lower()


@pytest.mark.asyncio
async def test_set_assignment_writes_row_and_invalidates() -> None:
    s = _make_session()
    s.execute.side_effect = [
        _scalar_result(_provider_row(key="qwen", api_key="sk-x")),  # provider exists
        _scalar_result(None),  # no existing assignment
    ]

    # Simulate the DB-side default firing on refresh — server_default/onupdate
    # populate updated_at after commit, which the endpoint then serializes.
    async def _refresh(row):
        row.updated_at = datetime.now(UTC)

    s.refresh = AsyncMock(side_effect=_refresh)

    import apps.api.admin.llm as admin_mod

    invalidations = {"n": 0}
    orig = admin_mod.invalidate_router_cache
    admin_mod.invalidate_router_cache = lambda: invalidations.__setitem__("n", invalidations["n"] + 1)

    try:
        app = _build(s)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
            r = await ac.put(
                "/admin/llm/active/cheap",
                json={
                    "provider_key": "qwen",
                    "model_id": "dashscope/qwen-turbo",
                    "notes": "smoke test",
                },
            )
    finally:
        admin_mod.invalidate_router_cache = orig

    assert r.status_code == 200, r.text
    assert s.add.called
    added: LlmModelAssignment = s.add.call_args.args[0]
    assert added.tier == "cheap"
    assert added.provider_key == "qwen"
    assert added.model_id == "dashscope/qwen-turbo"
    assert s.commit.called
    assert invalidations["n"] == 1


@pytest.mark.asyncio
async def test_set_assignment_rejects_unknown_tier() -> None:
    s = _make_session()
    app = _build(s)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        r = await ac.put(
            "/admin/llm/active/some_made_up_tier",
            json={"provider_key": "qwen", "model_id": "dashscope/qwen-turbo"},
        )
    assert r.status_code == 400


# --- auth gate ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_endpoints_reject_non_super_admin() -> None:
    """If we forget the auth dep on a new endpoint, this test catches it."""
    from apps.api.admin.auth import current_admin

    s = _make_session()
    app = FastAPI()
    app.include_router(llm_router, prefix="/admin/llm")

    async def override_session():
        yield s

    app.dependency_overrides[get_session] = override_session
    # Authenticate as a *tenant* admin, not super-admin -- should be 403.
    app.dependency_overrides[current_admin] = lambda: AdminPrincipal(
        tenant_id="t-1", email="member@example.com", role="owner"
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        r = await ac.get("/admin/llm/catalog")
    assert r.status_code == 403
