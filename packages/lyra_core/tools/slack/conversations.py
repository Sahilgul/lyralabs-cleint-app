"""Slack conversations tools: history + replies.

These mirror the surface of the official `mcp.slack.com` server's
`conversations_history` tool, but live in-process so we don't pay a
JSON-RPC round-trip just to call our own bot. They give the agent a
"native listener + skill" pattern (a la OpenClaw): Socket Mode delivers
the live event, and these tools let the model fetch older context on
demand.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from ..base import Tool, ToolContext, ToolError
from ..registry import default_registry
from ._client import SlackTokenMissing, _bot_token_for


class _SlackMessage(BaseModel):
    user_id: str | None = None
    text: str
    ts: str
    thread_ts: str | None = None
    is_bot: bool = False


class ConversationsHistoryInput(BaseModel):
    channel_id: str = Field(
        description=(
            "Slack channel/DM/group id (e.g. 'C0123' for a public channel, "
            "'D0123' for a DM, 'G0123' for a private channel/group DM). "
            "Use the same channel_id you saw in the inbound event."
        )
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=200,
        description=(
            "How many messages to return (most recent first). 20 is a good "
            "default for 'remind me what we just talked about'."
        ),
    )
    oldest: float | None = Field(
        default=None,
        description="Optional Unix timestamp; only return messages newer than this.",
    )
    latest: float | None = Field(
        default=None,
        description="Optional Unix timestamp; only return messages older than this.",
    )


class ConversationsHistoryOutput(BaseModel):
    channel_id: str
    messages: list[_SlackMessage]
    has_more: bool = False


class ConversationsHistory(Tool[ConversationsHistoryInput, ConversationsHistoryOutput]):
    name = "slack.conversations.history"
    description = (
        "Read recent messages from a Slack channel or DM. Use this when you "
        "need older context that's no longer in your live conversation memory "
        "(e.g. user references something they said earlier in the day, or you "
        "joined a thread late). Returns the most recent N messages."
    )
    provider = "slack"
    Input = ConversationsHistoryInput
    Output = ConversationsHistoryOutput

    async def run(
        self, ctx: ToolContext, args: ConversationsHistoryInput
    ) -> ConversationsHistoryOutput:
        try:
            token = await _bot_token_for(ctx.tenant_id)
        except SlackTokenMissing as exc:
            raise ToolError(str(exc)) from exc

        client = AsyncWebClient(token=token)
        kwargs: dict[str, Any] = {"channel": args.channel_id, "limit": args.limit}
        if args.oldest is not None:
            kwargs["oldest"] = str(args.oldest)
        if args.latest is not None:
            kwargs["latest"] = str(args.latest)

        try:
            resp = await client.conversations_history(**kwargs)
        except SlackApiError as exc:
            err = (exc.response.data or {}).get("error", str(exc))
            # `not_in_channel` / `channel_not_found` are common when the bot
            # was removed; surface clearly so the model doesn't loop.
            raise ToolError(f"slack.conversations.history failed: {err}") from exc

        raw_msgs = resp.data.get("messages", []) or []
        messages = [
            _SlackMessage(
                user_id=m.get("user") or m.get("bot_id"),
                text=m.get("text", ""),
                ts=m.get("ts", ""),
                thread_ts=m.get("thread_ts"),
                is_bot=bool(m.get("bot_id") or m.get("subtype") == "bot_message"),
            )
            for m in raw_msgs
        ]
        return ConversationsHistoryOutput(
            channel_id=args.channel_id,
            messages=messages,
            has_more=bool(resp.data.get("has_more", False)),
        )


# -----------------------------------------------------------------------------


class ConversationsRepliesInput(BaseModel):
    channel_id: str
    thread_ts: str = Field(description="The ts of the thread root message.")
    limit: int = Field(default=20, ge=1, le=200)


class ConversationsRepliesOutput(BaseModel):
    channel_id: str
    thread_ts: str
    messages: list[_SlackMessage]


class ConversationsReplies(
    Tool[ConversationsRepliesInput, ConversationsRepliesOutput]
):
    name = "slack.conversations.replies"
    description = (
        "Read every message in a specific Slack thread. Useful when the user "
        "asks you to summarize or follow up on a thread you weren't pinged in."
    )
    provider = "slack"
    Input = ConversationsRepliesInput
    Output = ConversationsRepliesOutput

    async def run(
        self, ctx: ToolContext, args: ConversationsRepliesInput
    ) -> ConversationsRepliesOutput:
        try:
            token = await _bot_token_for(ctx.tenant_id)
        except SlackTokenMissing as exc:
            raise ToolError(str(exc)) from exc

        client = AsyncWebClient(token=token)
        try:
            resp = await client.conversations_replies(
                channel=args.channel_id, ts=args.thread_ts, limit=args.limit
            )
        except SlackApiError as exc:
            err = (exc.response.data or {}).get("error", str(exc))
            raise ToolError(f"slack.conversations.replies failed: {err}") from exc

        raw_msgs = resp.data.get("messages", []) or []
        return ConversationsRepliesOutput(
            channel_id=args.channel_id,
            thread_ts=args.thread_ts,
            messages=[
                _SlackMessage(
                    user_id=m.get("user") or m.get("bot_id"),
                    text=m.get("text", ""),
                    ts=m.get("ts", ""),
                    thread_ts=m.get("thread_ts"),
                    is_bot=bool(m.get("bot_id") or m.get("subtype") == "bot_message"),
                )
                for m in raw_msgs
            ],
        )


default_registry.register(ConversationsHistory())
default_registry.register(ConversationsReplies())
