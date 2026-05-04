"""Slack canvas tool.

Creates a standalone Slack canvas with markdown content. Marked as a
WRITE tool (`requires_approval=True`) so it routes through the existing
`submit_plan_for_approval` gate -- the user sees a Block Kit preview
before any canvas is created. The tool itself never bypasses approval;
the executor only runs it after the gate flips to approved.
"""

from __future__ import annotations

from typing import Any, Literal

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

        assert isinstance(resp.data, dict)
        canvas_id = resp.data.get("canvas_id") or ""
        # Slack doesn't return a canonical URL on create -- best-effort
        # construction; the model can also fall back to "your canvas is
        # ready" without a link if absent.
        url = None
        return CanvasCreateOutput(canvas_id=canvas_id, url=url)


# -----------------------------------------------------------------------------


class CanvasUpdateInput(BaseModel):
    canvas_id: str = Field(description="Slack canvas id (e.g. 'F0CANVAS123').")
    operation: Literal["insert_at_end", "insert_at_start", "replace"] = Field(
        default="insert_at_end",
        description=(
            "What to do with `markdown`. `insert_at_end` appends to the canvas, "
            "`insert_at_start` prepends, `replace` swaps the entire body."
        ),
    )
    markdown: str = Field(description="Slack-flavored markdown to insert / replace.")


class CanvasUpdateOutput(BaseModel):
    canvas_id: str
    ok: bool = True


class CanvasUpdate(Tool[CanvasUpdateInput, CanvasUpdateOutput]):
    name = "slack.canvas.update"
    description = (
        "Append, prepend, or replace content on an existing Slack canvas. "
        "Use to keep a living doc current — 'add today's standup notes', "
        "'replace the project status section', etc. REQUIRES APPROVAL: "
        "include in a submit_plan_for_approval plan; the user sees the "
        "operation + markdown preview before any edit goes through."
    )
    provider = "slack"
    requires_approval = True
    Input = CanvasUpdateInput
    Output = CanvasUpdateOutput

    async def run(self, ctx: ToolContext, args: CanvasUpdateInput) -> CanvasUpdateOutput:
        try:
            token = await _bot_token_for(ctx.tenant_id)
        except SlackTokenMissing as exc:
            raise ToolError(str(exc)) from exc

        client = AsyncWebClient(token=token)
        change: dict[str, Any] = {
            "operation": args.operation,
            "document_content": {"type": "markdown", "markdown": args.markdown},
        }
        try:
            await client.canvases_edit(canvas_id=args.canvas_id, changes=[change])
        except SlackApiError as exc:
            err = (exc.response.data or {}).get("error", str(exc))
            raise ToolError(f"slack.canvas.update failed: {err}") from exc

        return CanvasUpdateOutput(canvas_id=args.canvas_id, ok=True)


# -----------------------------------------------------------------------------


class _CanvasSection(BaseModel):
    section_id: str
    markdown: str


class CanvasReadInput(BaseModel):
    canvas_id: str = Field(description="Slack canvas id to read.")


class CanvasReadOutput(BaseModel):
    canvas_id: str
    sections: list[_CanvasSection]
    full_markdown: str


class CanvasRead(Tool[CanvasReadInput, CanvasReadOutput]):
    name = "slack.canvas.read"
    description = (
        "Read the markdown content of a Slack canvas. Returns both per-section "
        "blocks (so you can target a specific section in a follow-up "
        "`slack.canvas.update` call) and the fully concatenated markdown for "
        "summarization. Use to reason about an existing canvas before editing it."
    )
    provider = "slack"
    Input = CanvasReadInput
    Output = CanvasReadOutput

    async def run(self, ctx: ToolContext, args: CanvasReadInput) -> CanvasReadOutput:
        try:
            token = await _bot_token_for(ctx.tenant_id)
        except SlackTokenMissing as exc:
            raise ToolError(str(exc)) from exc

        client = AsyncWebClient(token=token)
        # `canvases.sections.lookup` with an empty criteria returns every
        # section. Slack's API surface here is awkward (`criteria` is an
        # object, not a list), but this call is the canonical way to dump
        # canvas content as markdown.
        try:
            resp = await client.api_call(
                "canvases.sections.lookup",
                params={
                    "canvas_id": args.canvas_id,
                    "criteria": {"section_types": ["any_header", "any_text"]},
                },
            )
        except SlackApiError as exc:
            err = (exc.response.data or {}).get("error", str(exc))
            raise ToolError(f"slack.canvas.read failed: {err}") from exc

        data = resp.data if hasattr(resp, "data") else resp
        assert isinstance(data, dict)
        raw_sections: list[dict[str, Any]] = data.get("sections") or []
        sections = [
            _CanvasSection(
                section_id=s.get("id", ""),
                markdown=s.get("markdown") or s.get("text") or "",
            )
            for s in raw_sections
        ]
        full_markdown = "\n\n".join(s.markdown for s in sections if s.markdown)
        return CanvasReadOutput(
            canvas_id=args.canvas_id, sections=sections, full_markdown=full_markdown
        )


default_registry.register(CanvasCreate())
default_registry.register(CanvasUpdate())
default_registry.register(CanvasRead())
