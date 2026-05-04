"""Slack chat tools: post + schedule messages anywhere in the workspace.

`poster.py` already does `chat.postMessage`, but it's bound to the agent's
*final reply* in the current thread. These tools let the model post a
message to ANY channel/DM/thread as a deliberate write step (e.g.
"summarize this thread to #leadership", "DM Alice the meeting time").

Both are MEDIUM-tier writes: they route through `submit_plan_for_approval`
so the user sees a Block Kit preview before any message goes out to real
people. The executor only invokes `run` after the gate flips to approved.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from ..base import Tool, ToolContext, ToolError
from ..registry import default_registry
from ._client import SlackTokenMissing, _bot_token_for


class SendMessageInput(BaseModel):
    channel_id: str = Field(
        description=(
            "Destination Slack channel/DM/group id (e.g. 'C0123' for a "
            "channel, 'D0123' for a DM, 'G0123' for a private group). "
            "Use `slack.conversations.open` first if you only have a "
            "user_id and want to start a DM."
        )
    )
    text: str = Field(
        description=(
            "Plain-text fallback for the message. Always provide this — Slack "
            "uses it for notifications and accessibility even when blocks are set."
        )
    )
    thread_ts: str | None = Field(
        default=None,
        description=(
            "If set, the message posts as a threaded reply on this ts. "
            "Omit to post a new top-level message in the channel."
        ),
    )
    blocks: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Optional Slack Block Kit array for rich formatting. If provided, "
            "Slack renders these and uses `text` only as the notification fallback."
        ),
    )


class SendMessageOutput(BaseModel):
    channel_id: str
    ts: str
    permalink: str | None = None


class ChatSendMessage(Tool[SendMessageInput, SendMessageOutput]):
    name = "slack.chat.send_message"
    description = (
        "Post a message to any Slack channel, DM, or thread. Use this to "
        "proactively reach out — e.g. 'DM the meeting summary to Alice', "
        "'announce the launch in #general', 'reply in the project thread'. "
        "REQUIRES APPROVAL: include in a submit_plan_for_approval plan, "
        "do NOT call directly. The user sees the message preview before send."
    )
    provider = "slack"
    requires_approval = True
    Input = SendMessageInput
    Output = SendMessageOutput

    async def run(self, ctx: ToolContext, args: SendMessageInput) -> SendMessageOutput:
        try:
            token = await _bot_token_for(ctx.tenant_id)
        except SlackTokenMissing as exc:
            raise ToolError(str(exc)) from exc

        client = AsyncWebClient(token=token)
        kwargs: dict[str, Any] = {"channel": args.channel_id, "text": args.text}
        if args.thread_ts is not None:
            kwargs["thread_ts"] = args.thread_ts
        if args.blocks is not None:
            kwargs["blocks"] = args.blocks

        try:
            resp = await client.chat_postMessage(**kwargs)
        except SlackApiError as exc:
            err = (exc.response.data or {}).get("error", str(exc))
            raise ToolError(f"slack.chat.send_message failed: {err}") from exc

        assert isinstance(resp.data, dict)
        ts = resp.data.get("ts", "")

        permalink: str | None = None
        try:
            link_resp = await client.chat_getPermalink(channel=args.channel_id, message_ts=ts)
            assert isinstance(link_resp.data, dict)
            permalink = link_resp.data.get("permalink")
        except SlackApiError:
            # Permalink lookup is best-effort — the user got the message either way.
            pass

        return SendMessageOutput(channel_id=args.channel_id, ts=ts, permalink=permalink)


# -----------------------------------------------------------------------------


class ScheduleMessageInput(BaseModel):
    channel_id: str = Field(description="Destination channel/DM/group id.")
    text: str = Field(description="Message text (also serves as notification fallback).")
    post_at: int = Field(
        description=(
            "Unix epoch seconds at which Slack should post the message. "
            "Must be at least ~10 seconds in the future; Slack rejects "
            "schedules more than 120 days out."
        )
    )
    thread_ts: str | None = Field(
        default=None, description="If set, posts as a threaded reply on send."
    )
    blocks: list[dict[str, Any]] | None = Field(
        default=None, description="Optional Block Kit blocks."
    )


class ScheduleMessageOutput(BaseModel):
    channel_id: str
    scheduled_message_id: str
    post_at: int


class ChatScheduleMessage(Tool[ScheduleMessageInput, ScheduleMessageOutput]):
    name = "slack.chat.schedule_message"
    description = (
        "Schedule a Slack message to post at a future time (e.g. 'send the "
        "kickoff reminder Monday at 9am'). The message is queued by Slack "
        "and dispatched automatically — no further action needed. "
        "REQUIRES APPROVAL: include in a submit_plan_for_approval plan."
    )
    provider = "slack"
    requires_approval = True
    Input = ScheduleMessageInput
    Output = ScheduleMessageOutput

    async def run(
        self, ctx: ToolContext, args: ScheduleMessageInput
    ) -> ScheduleMessageOutput:
        try:
            token = await _bot_token_for(ctx.tenant_id)
        except SlackTokenMissing as exc:
            raise ToolError(str(exc)) from exc

        client = AsyncWebClient(token=token)
        kwargs: dict[str, Any] = {
            "channel": args.channel_id,
            "text": args.text,
            "post_at": args.post_at,
        }
        if args.thread_ts is not None:
            kwargs["thread_ts"] = args.thread_ts
        if args.blocks is not None:
            kwargs["blocks"] = args.blocks

        try:
            resp = await client.chat_scheduleMessage(**kwargs)
        except SlackApiError as exc:
            err = (exc.response.data or {}).get("error", str(exc))
            raise ToolError(f"slack.chat.schedule_message failed: {err}") from exc

        assert isinstance(resp.data, dict)
        return ScheduleMessageOutput(
            channel_id=args.channel_id,
            scheduled_message_id=resp.data.get("scheduled_message_id", ""),
            post_at=int(resp.data.get("post_at") or args.post_at),
        )


default_registry.register(ChatSendMessage())
default_registry.register(ChatScheduleMessage())
