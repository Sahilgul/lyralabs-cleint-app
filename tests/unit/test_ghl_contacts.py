"""lyra_core.tools.ghl.contacts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import respx
from lyra_core.tools.base import ToolError
from lyra_core.tools.credentials import ProviderCredentials
from lyra_core.tools.ghl.client import GHL_BASE
from lyra_core.tools.ghl.contacts import (
    GhlContactsCreate,
    GhlContactsCreateInput,
    GhlContactsSearch,
    GhlContactsSearchInput,
)


def _ghl_creds():
    return ProviderCredentials(
        provider="ghl",
        access_token="ghl-token",
        refresh_token="rt",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        external_account_id="loc-1",
        scopes="contacts.readonly contacts.write",
        metadata={"location_id": "loc-1"},
    )


@pytest.mark.asyncio
async def test_search_returns_normalized_contacts(make_ctx) -> None:
    ctx = make_ctx(creds=_ghl_creds())
    with respx.mock(base_url=GHL_BASE) as mock:
        mock.post("/contacts/search").respond(
            200,
            json={
                "contacts": [
                    {
                        "id": "c-1",
                        "contactName": "Alice Smith",
                        "firstName": "Alice",
                        "lastName": "Smith",
                        "email": "a@x.com",
                        "phone": "555-0001",
                        "tags": ["lead", "vip"],
                    },
                    {
                        "id": "c-2",
                        "name": "Bob Jones",
                    },
                ]
            },
        )
        out = await GhlContactsSearch().run(ctx, GhlContactsSearchInput(query="alice"))

    assert out.count == 2
    assert out.contacts[0].name == "Alice Smith"
    assert out.contacts[0].tags == ["lead", "vip"]
    assert out.contacts[1].id == "c-2"
    # Falls back to "name" when "contactName" missing
    assert out.contacts[1].name == "Bob Jones"


@pytest.mark.asyncio
async def test_search_includes_location_id_in_body(make_ctx) -> None:
    ctx = make_ctx(creds=_ghl_creds())
    with respx.mock(base_url=GHL_BASE) as mock:
        route = mock.post("/contacts/search").respond(200, json={"contacts": []})
        await GhlContactsSearch().run(ctx, GhlContactsSearchInput(query="x", limit=5))

        body = route.calls[0].request.read().decode()
        assert "loc-1" in body
        assert '"pageLimit":5' in body or '"pageLimit": 5' in body


@pytest.mark.asyncio
async def test_search_caps_results_to_limit(make_ctx) -> None:
    ctx = make_ctx(creds=_ghl_creds())
    with respx.mock(base_url=GHL_BASE) as mock:
        mock.post("/contacts/search").respond(
            200,
            json={"contacts": [{"id": f"c-{i}"} for i in range(20)]},
        )
        out = await GhlContactsSearch().run(ctx, GhlContactsSearchInput(query="x", limit=3))
    assert out.count == 3


@pytest.mark.asyncio
async def test_search_validates_limit_max() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GhlContactsSearchInput(query="x", limit=999)


# -----------------------------------------------------------------------------
# Create
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_requires_email_or_phone(make_ctx) -> None:
    ctx = make_ctx(creds=_ghl_creds())
    with pytest.raises(ToolError, match="email or phone"):
        await GhlContactsCreate().run(ctx, GhlContactsCreateInput(first_name="Alice"))


@pytest.mark.asyncio
async def test_create_dry_run_returns_dry_run_id(make_ctx) -> None:
    ctx = make_ctx(creds=_ghl_creds(), dry_run=True)
    out = await GhlContactsCreate().run(
        ctx, GhlContactsCreateInput(first_name="A", email="a@x.com")
    )
    assert out.id == "dry-run"
    assert out.email == "a@x.com"


@pytest.mark.asyncio
async def test_create_strips_none_fields(make_ctx) -> None:
    ctx = make_ctx(creds=_ghl_creds())
    with respx.mock(base_url=GHL_BASE) as mock:
        route = mock.post("/contacts/").respond(
            200, json={"contact": {"id": "c-X", "email": "a@x.com"}}
        )
        out = await GhlContactsCreate().run(
            ctx, GhlContactsCreateInput(first_name="A", email="a@x.com")
        )
        body = route.calls[0].request.read().decode()
        assert '"firstName":"A"' in body or '"firstName": "A"' in body
        # last_name was None -> excluded
        assert "lastName" not in body
        assert "phone" not in body
        assert "tags" in body  # empty list still sent
        assert out.id == "c-X"
        assert out.email == "a@x.com"


@pytest.mark.asyncio
async def test_create_handles_response_without_contact_wrapper(make_ctx) -> None:
    ctx = make_ctx(creds=_ghl_creds())
    with respx.mock(base_url=GHL_BASE) as mock:
        mock.post("/contacts/").respond(200, json={"id": "c-1", "email": "x@y.com"})
        out = await GhlContactsCreate().run(
            ctx,
            GhlContactsCreateInput(first_name="A", phone="555"),
        )
    assert out.id == "c-1"
    assert out.email == "x@y.com"


def test_create_requires_approval() -> None:
    assert GhlContactsCreate.requires_approval is True


def test_search_does_not_require_approval() -> None:
    assert GhlContactsSearch.requires_approval is False
