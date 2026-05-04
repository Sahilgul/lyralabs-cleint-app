"""Slack search tool.

`search.messages` is the workspace-wide keyword search. Slack rejects
bot tokens for this endpoint with `not_allowed_token_type`, so the tool
uses the user token (xoxp-) captured at install time. If the workspace
was installed before user_scopes were configured, we surface a typed
"missing permission" error rather than crashing -- the model can then
tell the user to reauthorize.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from ..base import Tool, ToolContext, ToolError
from ..registry import default_registry
from ._client import SlackTokenMissing, _user_token_for


class _SearchHit(BaseModel):
    channel_id: str
    channel_name: str | None = None
    user_id: str | None = None
    user_name: str | None = None
    text: str
    ts: str
    permalink: str | None = None


class SearchMessagesInput(BaseModel):
    query: str = Field(
        description=(
            "Slack search query. Supports Slack's modifiers: in:#channel, "
            "from:@user, before:YYYY-MM-DD, has:link, etc."
        )
    )
    count: int = Field(default=20, ge=1, le=100)
    sort: Literal["score", "timestamp"] = Field(
        default="score",
        description="'score' = relevance ranking, 'timestamp' = newest first.",
    )


class SearchMessagesOutput(BaseModel):
    query: str
    matches: list[_SearchHit]
    total: int = 0


class SearchMessages(Tool[SearchMessagesInput, SearchMessagesOutput]):
    name = "slack.search.messages"
    description = (
        "Search across the workspace for messages matching a query. Use this "
        "when the user asks 'what did we decide about X' or 'find that link "
        "Bob shared' and the answer isn't in the live conversation. Requires "
        "the user-token search:read.* scopes (Slack disallows bot tokens here)."
    )
    provider = "slack"
    Input = SearchMessagesInput
    Output = SearchMessagesOutput

    async def run(self, ctx: ToolContext, args: SearchMessagesInput) -> SearchMessagesOutput:
        try:
            token = await _user_token_for(ctx.tenant_id)
        except SlackTokenMissing as exc:
            raise ToolError(str(exc)) from exc

        client = AsyncWebClient(token=token)
        try:
            resp = await client.search_messages(query=args.query, count=args.count, sort=args.sort)
        except SlackApiError as exc:
            err = (exc.response.data or {}).get("error", str(exc))
            raise ToolError(f"slack.search.messages failed: {err}") from exc

        assert isinstance(resp.data, dict)
        msgs_block = resp.data.get("messages") or {}
        matches_raw: list[dict[str, Any]] = msgs_block.get("matches", []) or []
        return SearchMessagesOutput(
            query=args.query,
            total=int(msgs_block.get("total", 0)),
            matches=[
                _SearchHit(
                    channel_id=(m.get("channel") or {}).get("id", ""),
                    channel_name=(m.get("channel") or {}).get("name"),
                    user_id=m.get("user"),
                    user_name=m.get("username"),
                    text=m.get("text", ""),
                    ts=m.get("ts", ""),
                    permalink=m.get("permalink"),
                )
                for m in matches_raw
            ],
        )


# -----------------------------------------------------------------------------


class _FileHit(BaseModel):
    id: str
    name: str | None = None
    title: str | None = None
    filetype: str | None = None
    mimetype: str | None = None
    size: int | None = None
    user_id: str | None = None
    permalink: str | None = None
    url_private: str | None = None
    timestamp: int | None = None


class SearchFilesInput(BaseModel):
    query: str = Field(
        description=(
            "Slack file search query. Supports modifiers: in:#channel, "
            "from:@user, before:YYYY-MM-DD, after:YYYY-MM-DD, "
            "filetype:pdf, has:link, etc."
        )
    )
    count: int = Field(default=20, ge=1, le=100)
    sort: Literal["score", "timestamp"] = Field(
        default="score",
        description="'score' = relevance ranking, 'timestamp' = newest first.",
    )


class SearchFilesOutput(BaseModel):
    query: str
    matches: list[_FileHit]
    total: int = 0


class SearchFiles(Tool[SearchFilesInput, SearchFilesOutput]):
    name = "slack.search.files"
    description = (
        "Search files (PDFs, images, snippets, uploads) shared anywhere in the "
        "workspace. Use when the user asks 'find that PDF Bob shared' or "
        "'what was the deck from last week'. Like `search.messages`, this "
        "needs the user-token `search:read.files` scope (Slack disallows "
        "bot tokens here). Returns file metadata + permalinks."
    )
    provider = "slack"
    Input = SearchFilesInput
    Output = SearchFilesOutput

    async def run(self, ctx: ToolContext, args: SearchFilesInput) -> SearchFilesOutput:
        from ._client import _user_token_for

        try:
            token = await _user_token_for(ctx.tenant_id)
        except SlackTokenMissing as exc:
            raise ToolError(str(exc)) from exc

        client = AsyncWebClient(token=token)
        try:
            resp = await client.search_files(query=args.query, count=args.count, sort=args.sort)
        except SlackApiError as exc:
            err = (exc.response.data or {}).get("error", str(exc))
            raise ToolError(f"slack.search.files failed: {err}") from exc

        assert isinstance(resp.data, dict)
        files_block = resp.data.get("files") or {}
        matches_raw: list[dict[str, Any]] = files_block.get("matches", []) or []
        return SearchFilesOutput(
            query=args.query,
            total=int(files_block.get("total", 0)),
            matches=[
                _FileHit(
                    id=m.get("id", ""),
                    name=m.get("name"),
                    title=m.get("title"),
                    filetype=m.get("filetype"),
                    mimetype=m.get("mimetype"),
                    size=m.get("size"),
                    user_id=m.get("user"),
                    permalink=m.get("permalink"),
                    url_private=m.get("url_private"),
                    timestamp=m.get("timestamp"),
                )
                for m in matches_raw
            ],
        )


default_registry.register(SearchMessages())
default_registry.register(SearchFiles())
