"""Slack canvas tool.

Creates a standalone Slack canvas with markdown content. Marked as a
WRITE tool (`requires_approval=True`) so it routes through the existing
`submit_plan_for_approval` gate -- the user sees a Block Kit preview
before any canvas is created. The tool itself never bypasses approval;
the executor only runs it after the gate flips to approved.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from ..base import Tool, ToolContext, ToolError
from ..registry import default_registry
from ._client import SlackTokenMissing, _bot_token_for


class CanvasCreateInput(BaseModel):
    title: str = Field(description="Canvas title shown in Slack.")
    markdown: str = Field(
        description=(
            "Canvas body in Slack-flavored markdown. Supports headings, lists, "
            "links, inline code, code fences, and checkboxes."
        )
    )
    channel_id: str | None = Field(
        default=None,
        description=(
            "If provided, attach the new canvas to this channel so its members "
            "can see it. Omit for a personal/standalone canvas."
        ),
    )


class CanvasCreateOutput(BaseModel):
    canvas_id: str
    url: str | None = None


class CanvasCreate(Tool[CanvasCreateInput, CanvasCreateOutput]):
    name = "slack.canvas.create"
    description = (
        "Create a Slack canvas (a rich, shareable doc) with markdown content "
        "and optionally attach it to a channel. Use this for project briefs, "
        "meeting summaries, or anything the user wants to keep / share later. "
        "REQUIRES APPROVAL: include this in a submit_plan_for_approval plan, "
        "do NOT call directly."
    )
    provider = "slack"
    requires_approval = True
    Input = CanvasCreateInput
    Output = CanvasCreateOutput

    async def run(self, ctx: ToolContext, args: CanvasCreateInput) -> CanvasCreateOutput:
        try:
            token = await _bot_token_for(ctx.tenant_id)
        except SlackTokenMissing as exc:
            raise ToolError(str(exc)) from exc

        client = AsyncWebClient(token=token)
        try:
            # `canvases.create` lives under client.canvases_create in slack-sdk.
            # The API expects `document_content={"type": "markdown", "markdown": "..."}`
            # and a `title` string. `channel_id` is optional and only valid
            # for canvases attached to a single channel.
            kwargs: dict = {
                "title": args.title,
                "document_content": {"type": "markdown", "markdown": args.markdown},
            }
            if args.channel_id:
                kwargs["channel_id"] = args.channel_id
            resp = await client.canvases_create(**kwargs)
        except SlackApiError as exc:
            err = (exc.response.data or {}).get("error", str(exc))
            raise ToolError(f"slack.canvas.create failed: {err}") from exc

        canvas_id = resp.data.get("canvas_id") or ""
        # Slack doesn't return a canonical URL on create -- best-effort
        # construction; the model can also fall back to "your canvas is
        # ready" without a link if absent.
        url = None
        return CanvasCreateOutput(canvas_id=canvas_id, url=url)


default_registry.register(CanvasCreate())
