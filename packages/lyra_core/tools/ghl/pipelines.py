"""GHL Pipelines: list opportunities, optionally filtered by stuck-in-stage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, Field

from ..base import Tool, ToolContext, ToolError
from ..registry import default_registry
from .client import GhlClient


class GhlOpportunity(BaseModel):
    id: str
    name: str
    contact_id: str | None = None
    contact_name: str | None = None
    pipeline_id: str
    stage_id: str
    stage_name: str | None = None
    monetary_value: float | None = None
    status: str | None = None
    updated_at: str | None = None
    days_in_stage: int | None = None


class GhlPipelineOpportunitiesInput(BaseModel):
    pipeline_id: str | None = Field(
        default=None,
        description="Pipeline id. If omitted, return opportunities across all pipelines.",
    )
    stage_id: str | None = None
    stuck_for_days: int | None = Field(
        default=None,
        description="If set, return only opportunities not updated in the last N days.",
    )
    limit: int = Field(default=50, le=200)


class GhlPipelineOpportunitiesOutput(BaseModel):
    opportunities: list[GhlOpportunity]
    count: int


class GhlPipelineOpportunities(
    Tool[GhlPipelineOpportunitiesInput, GhlPipelineOpportunitiesOutput]
):
    name = "ghl.pipelines.opportunities"
    description = (
        "List opportunities in GHL pipelines, optionally filtered by stage and 'stuck' duration. "
        "Read-only."
    )
    provider = "ghl"
    Input = GhlPipelineOpportunitiesInput
    Output = GhlPipelineOpportunitiesOutput

    async def run(
        self, ctx: ToolContext, args: GhlPipelineOpportunitiesInput
    ) -> GhlPipelineOpportunitiesOutput:
        creds = await ctx.creds_lookup("ghl")
        client = GhlClient(creds)

        if not client.location_id:
            raise ToolError("missing GHL location id; reconnect the integration")

        params: dict[str, str | int] = {
            "location_id": client.location_id,
            "limit": args.limit,
        }
        if args.pipeline_id:
            params["pipeline_id"] = args.pipeline_id
        if args.stage_id:
            params["pipeline_stage_id"] = args.stage_id

        data = await client.request("GET", "/opportunities/search", params=params)

        cutoff = (
            datetime.now(UTC) - timedelta(days=args.stuck_for_days)
            if args.stuck_for_days
            else None
        )
        opps: list[GhlOpportunity] = []
        for o in data.get("opportunities", []):
            updated = o.get("updatedAt") or o.get("dateAdded")
            updated_dt: datetime | None = None
            if updated:
                try:
                    updated_dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                except ValueError:
                    updated_dt = None
            if cutoff and updated_dt and updated_dt > cutoff:
                continue
            days_in_stage = (
                (datetime.now(UTC) - updated_dt).days if updated_dt else None
            )
            opps.append(
                GhlOpportunity(
                    id=o.get("id", ""),
                    name=o.get("name", ""),
                    contact_id=(o.get("contact") or {}).get("id"),
                    contact_name=(o.get("contact") or {}).get("name"),
                    pipeline_id=o.get("pipelineId", ""),
                    stage_id=o.get("pipelineStageId", ""),
                    stage_name=o.get("pipelineStageName"),
                    monetary_value=o.get("monetaryValue"),
                    status=o.get("status"),
                    updated_at=updated,
                    days_in_stage=days_in_stage,
                )
            )

        return GhlPipelineOpportunitiesOutput(opportunities=opps, count=len(opps))


default_registry.register(GhlPipelineOpportunities())
