"""lyra_core.tools.ghl.client.GhlClient."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx
from lyra_core.tools.base import ToolError
from lyra_core.tools.credentials import ProviderCredentials
from lyra_core.tools.ghl.client import GHL_API_VERSION, GHL_BASE, GhlClient


def _creds(loc_id: str | None = "loc-acme-1", external: str = "loc-acme-1"):
    return ProviderCredentials(
        provider="ghl",
        access_token="ghl-token",
        refresh_token="rt",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        external_account_id=external,
        scopes="contacts.readonly",
        metadata={"location_id": loc_id} if loc_id else {},
    )


class TestLocationId:
    def test_prefers_metadata_location_id(self) -> None:
        c = GhlClient(_creds(loc_id="loc-A"))
        assert c.location_id == "loc-A"

    def test_falls_back_to_external_account_id_when_no_metadata(self) -> None:
        c = GhlClient(_creds(loc_id=None, external="loc-fallback"))
        assert c.location_id == "loc-fallback"


class TestHeaders:
    def test_includes_bearer_version_accept(self) -> None:
        c = GhlClient(_creds())
        h = c._headers()
        assert h["Authorization"] == "Bearer ghl-token"
        assert h["Version"] == GHL_API_VERSION
        assert h["Accept"] == "application/json"
        assert h["Content-Type"] == "application/json"


class TestRequest:
    @pytest.mark.asyncio
    async def test_get_returns_json_body(self) -> None:
        c = GhlClient(_creds())
        with respx.mock(base_url=GHL_BASE) as mock:
            mock.get("/contacts/").respond(200, json={"contacts": [{"id": "c1"}]})
            data = await c.request("GET", "/contacts/")
        assert data == {"contacts": [{"id": "c1"}]}

    @pytest.mark.asyncio
    async def test_post_passes_body(self) -> None:
        c = GhlClient(_creds())
        with respx.mock(base_url=GHL_BASE) as mock:
            route = mock.post("/contacts/").respond(200, json={"id": "x"})
            await c.request("POST", "/contacts/", json={"firstName": "Bob"})
            assert route.called
            assert route.calls[0].request.read().decode().count("Bob") == 1

    @pytest.mark.asyncio
    async def test_returns_empty_dict_on_no_content(self) -> None:
        c = GhlClient(_creds())
        with respx.mock(base_url=GHL_BASE) as mock:
            mock.delete("/contacts/x").respond(204)
            data = await c.request("DELETE", "/contacts/x")
        assert data == {}

    @pytest.mark.asyncio
    async def test_400_raises_tool_error(self) -> None:
        c = GhlClient(_creds())
        with respx.mock(base_url=GHL_BASE) as mock:
            mock.get("/contacts/").respond(400, json={"error": "bad"})
            with pytest.raises(ToolError, match="GHL GET /contacts/ -> 400"):
                await c.request("GET", "/contacts/")

    @pytest.mark.asyncio
    async def test_429_retried_then_succeeds(self) -> None:
        c = GhlClient(_creds())
        with respx.mock(base_url=GHL_BASE) as mock:
            route = mock.get("/contacts/")
            route.side_effect = [
                httpx.Response(429, json={}),
                httpx.Response(200, json={"ok": True}),
            ]
            data = await c.request("GET", "/contacts/")
        assert data == {"ok": True}
        assert route.call_count == 2

    @pytest.mark.asyncio
    async def test_500_retried_three_attempts_then_raises(self) -> None:
        c = GhlClient(_creds())
        with respx.mock(base_url=GHL_BASE) as mock:
            route = mock.get("/contacts/").mock(return_value=httpx.Response(503, json={}))
            with pytest.raises(httpx.HTTPStatusError):
                await c.request("GET", "/contacts/")
        assert route.call_count == 3

    @pytest.mark.asyncio
    async def test_passes_query_params(self) -> None:
        c = GhlClient(_creds())
        with respx.mock(base_url=GHL_BASE) as mock:
            route = mock.get("/opportunities/search").respond(200, json={})
            await c.request("GET", "/opportunities/search", params={"q": "abc", "n": 5})
            url = str(route.calls[0].request.url)
            assert "q=abc" in url
            assert "n=5" in url
