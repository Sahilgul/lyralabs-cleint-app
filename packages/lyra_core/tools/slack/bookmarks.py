"""Slack bookmarks tool: pin a link to a channel's bookmark bar.

Bookmarks live in the channel header (above the message stream) — they
are higher-visibility than pins for canonical references like project
specs, dashboards, or external docs. Classified LOW per the permissive
policy: bookmarks are channel-scoped, fully reversible, and visible only
to the same audience as the channel itself.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from ..base import Tool, ToolContext, ToolError, TrustTier
from ..registry import default_registry
from ._client import SlackTokenMissing, _bot_token_for


class BookmarksAddInput(BaseModel):
    channel_id: str = Field(description="Channel/group id to attach the bookmark to.")
    title: str = Field(
        description="Title shown on the bookmark chip (max ~50 chars).",
        min_length=1,
        max_length=120,
    )
    link: str = Field(description="URL the bookmark should point to.")
    emoji: str | None = Field(
        default=None,
        description="Optional emoji (with colons, e.g. ':link:') shown on the chip.",
    )


class BookmarksAddOutput(BaseModel):
    bookmark_id: str
    channel_id: str


class BookmarksAdd(Tool[BookmarksAddInput, BookmarksAddOutput]):
    name = "slack.bookmarks.add"
    description = (
        "Add a link bookmark to a channel's bookmark bar (the chips above "
        "the message stream). Use for canonical references — project specs, "
        "dashboards, runbooks. Higher visibility than `slack.pins.add` for "
        "external links."
    )
    provider = "slack"
    requires_approval = False
    trust_tier = TrustTier.LOW
    Input = BookmarksAddInput
    Output = BookmarksAddOutput

    async def run(
        self, ctx: ToolContext, args: BookmarksAddInput
    ) -> BookmarksAddOutput:
        try:
            token = await _bot_token_for(ctx.tenant_id)
        except SlackTokenMissing as exc:
            raise ToolError(str(exc)) from exc

        client = AsyncWebClient(token=token)
        kwargs: dict = {
            "channel_id": args.channel_id,
            "title": args.title,
            "type": "link",
            "link": args.link,
        }
        if args.emoji:
            kwargs["emoji"] = args.emoji

        try:
            resp = await client.bookmarks_add(**kwargs)
        except SlackApiError as exc:
            err = (exc.response.data or {}).get("error", str(exc))
            raise ToolError(f"slack.bookmarks.add failed: {err}") from exc

        assert isinstance(resp.data, dict)
        bookmark = resp.data.get("bookmark") or {}
        return BookmarksAddOutput(
            bookmark_id=bookmark.get("id", ""),
            channel_id=bookmark.get("channel_id", args.channel_id),
        )


default_registry.register(BookmarksAdd())
