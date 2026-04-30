"""lyra_core.tools.ghl.pipelines."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import respx

from lyra_core.tools.base import ToolError
from lyra_core.tools.credentials import ProviderCredentials
from lyra_core.tools.ghl.client import GHL_BASE
from lyra_core.tools.ghl.pipelines import (
    GhlPipelineOpportunities,
    GhlPipelineOpportunitiesInput,
)


def _ghl_creds(loc_id: str | None = "loc-1"):
    return ProviderCredentials(
        provider="ghl",
        access_token="token",
        refresh_token="rt",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        external_account_id=loc_id or "",
        scopes="opportunities.readonly",
        metadata={"location_id": loc_id} if loc_id else {},
    )


@pytest.mark.asyncio
async def test_returns_normalized_opportunities(make_ctx) -> None:
    ctx = make_ctx(creds=_ghl_creds())
    recent = (datetime.now(UTC) - timedelta(days=3)).isoformat().replace("+00:00", "Z")
    with respx.mock(base_url=GHL_BASE) as mock:
        mock.get("/opportunities/search").respond(
            200,
            json={
                "opportunities": [
                    {
                        "id": "o-1",
                        "name": "Acme Deal",
                        "contact": {"id": "ct-1", "name": "Alice"},
                        "pipelineId": "p-1",
                        "pipelineStageId": "s-1",
                        "pipelineStageName": "Negotiation",
                        "monetaryValue": 5000.0,
                        "status": "open",
                        "updatedAt": recent,
                    }
                ]
            },
        )
        out = await GhlPipelineOpportunities().run(
            ctx, GhlPipelineOpportunitiesInput(pipeline_id="p-1")
        )

    assert out.count == 1
    o = out.opportunities[0]
    assert o.id == "o-1"
    assert o.name == "Acme Deal"
    assert o.contact_id == "ct-1"
    assert o.contact_name == "Alice"
    assert o.monetary_value == 5000.0
    assert o.days_in_stage in (3, 4)  # depending on second tick


@pytest.mark.asyncio
async def test_filters_by_stuck_for_days(make_ctx) -> None:
    """Opportunities updated within the cutoff window are filtered out."""
    ctx = make_ctx(creds=_ghl_creds())
    recent = (datetime.now(UTC) - timedelta(days=2)).isoformat().replace("+00:00", "Z")
    old = (datetime.now(UTC) - timedelta(days=15)).isoformat().replace("+00:00", "Z")
    with respx.mock(base_url=GHL_BASE) as mock:
        mock.get("/opportunities/search").respond(
            200,
            json={
                "opportunities": [
                    {"id": "fresh", "name": "F", "pipelineId": "p", "pipelineStageId": "s",
                     "updatedAt": recent},
                    {"id": "stale", "name": "S", "pipelineId": "p", "pipelineStageId": "s",
                     "updatedAt": old},
                ]
            },
        )
        out = await GhlPipelineOpportunities().run(
            ctx, GhlPipelineOpportunitiesInput(stuck_for_days=7)
        )

    ids = {o.id for o in out.opportunities}
    assert ids == {"stale"}


@pytest.mark.asyncio
async def test_handles_unparseable_updated_at(make_ctx) -> None:
    """If updated_at is malformed, the opportunity is still returned."""
    ctx = make_ctx(creds=_ghl_creds())
    with respx.mock(base_url=GHL_BASE) as mock:
        mock.get("/opportunities/search").respond(
            200,
            json={
                "opportunities": [
                    {"id": "o-x", "name": "x", "pipelineId": "p", "pipelineStageId": "s",
                     "updatedAt": "not-a-date"}
                ]
            },
        )
        out = await GhlPipelineOpportunities().run(
            ctx, GhlPipelineOpportunitiesInput()
        )

    assert out.count == 1
    assert out.opportunities[0].days_in_stage is None


@pytest.mark.asyncio
async def test_passes_pipeline_and_stage_filters(make_ctx) -> None:
    ctx = make_ctx(creds=_ghl_creds())
    with respx.mock(base_url=GHL_BASE) as mock:
        route = mock.get("/opportunities/search").respond(200, json={"opportunities": []})
        await GhlPipelineOpportunities().run(
            ctx,
            GhlPipelineOpportunitiesInput(pipeline_id="p-A", stage_id="s-B", limit=20),
        )
        url = str(route.calls[0].request.url)
        assert "pipeline_id=p-A" in url
        assert "pipeline_stage_id=s-B" in url
        assert "limit=20" in url
        assert "location_id=loc-1" in url


@pytest.mark.asyncio
async def test_raises_when_no_location_id(make_ctx) -> None:
    ctx = make_ctx(creds=_ghl_creds(loc_id=None))
    with pytest.raises(ToolError, match="missing GHL location id"):
        await GhlPipelineOpportunities().run(
            ctx, GhlPipelineOpportunitiesInput()
        )


@pytest.mark.asyncio
async def test_uses_dateAdded_when_updatedAt_missing(make_ctx) -> None:
    ctx = make_ctx(creds=_ghl_creds())
    added = (datetime.now(UTC) - timedelta(days=10)).isoformat().replace("+00:00", "Z")
    with respx.mock(base_url=GHL_BASE) as mock:
        mock.get("/opportunities/search").respond(
            200,
            json={
                "opportunities": [
                    {"id": "x", "name": "x", "pipelineId": "p", "pipelineStageId": "s",
                     "dateAdded": added}
                ]
            },
        )
        out = await GhlPipelineOpportunities().run(ctx, GhlPipelineOpportunitiesInput())
    assert out.opportunities[0].updated_at == added


def test_does_not_require_approval() -> None:
    assert GhlPipelineOpportunities.requires_approval is False
