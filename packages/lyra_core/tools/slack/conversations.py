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

from ..base import Tool, ToolContext, ToolError, TrustTier
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

        assert isinstance(resp.data, dict)
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


class ConversationsReplies(Tool[ConversationsRepliesInput, ConversationsRepliesOutput]):
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

        assert isinstance(resp.data, dict)
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


# -----------------------------------------------------------------------------


class _SlackChannel(BaseModel):
    id: str
    name: str | None = None
    is_channel: bool = False
    is_group: bool = False
    is_im: bool = False
    is_private: bool = False
    is_archived: bool = False
    num_members: int | None = None
    topic: str | None = None
    purpose: str | None = None


def _to_channel(c: dict[str, Any]) -> _SlackChannel:
    topic = (c.get("topic") or {}).get("value") or None
    purpose = (c.get("purpose") or {}).get("value") or None
    return _SlackChannel(
        id=c.get("id", ""),
        name=c.get("name"),
        is_channel=bool(c.get("is_channel")),
        is_group=bool(c.get("is_group")),
        is_im=bool(c.get("is_im")),
        is_private=bool(c.get("is_private")),
        is_archived=bool(c.get("is_archived")),
        num_members=c.get("num_members"),
        topic=topic,
        purpose=purpose,
    )


# -----------------------------------------------------------------------------


class ConversationsOpenInput(BaseModel):
    user_ids: list[str] = Field(
        description=(
            "One or more Slack user ids (e.g. ['U0123']). One id opens a 1:1 "
            "DM; multiple ids open a multi-person DM (mpim). The bot must be "
            "able to message the listed users."
        ),
        min_length=1,
        max_length=8,
    )


class ConversationsOpenOutput(BaseModel):
    channel_id: str
    is_new: bool = False


class ConversationsOpen(Tool[ConversationsOpenInput, ConversationsOpenOutput]):
    name = "slack.conversations.open"
    description = (
        "Open or fetch a DM/group-DM with one or more users, returning the "
        "channel id you can pass to `slack.chat.send_message`. Idempotent — "
        "re-opening an existing DM returns the same channel id. Use this "
        "first whenever you only have user ids and want to message them."
    )
    provider = "slack"
    trust_tier = TrustTier.LOW  # creating a DM channel is fully reversible
    Input = ConversationsOpenInput
    Output = ConversationsOpenOutput

    async def run(
        self, ctx: ToolContext, args: ConversationsOpenInput
    ) -> ConversationsOpenOutput:
        try:
            token = await _bot_token_for(ctx.tenant_id)
        except SlackTokenMissing as exc:
            raise ToolError(str(exc)) from exc

        client = AsyncWebClient(token=token)
        try:
            resp = await client.conversations_open(users=",".join(args.user_ids))
        except SlackApiError as exc:
            err = (exc.response.data or {}).get("error", str(exc))
            raise ToolError(f"slack.conversations.open failed: {err}") from exc

        assert isinstance(resp.data, dict)
        channel = resp.data.get("channel") or {}
        return ConversationsOpenOutput(
            channel_id=channel.get("id", ""),
            is_new=not bool(resp.data.get("already_open", False)),
        )


# -----------------------------------------------------------------------------


class ConversationsListInput(BaseModel):
    types: str = Field(
        default="public_channel,private_channel",
        description=(
            "Comma-separated list of channel types to include. Valid: "
            "`public_channel`, `private_channel`, `mpim`, `im`. Defaults "
            "to public + private channels (excludes DMs)."
        ),
    )
    name_filter: str | None = Field(
        default=None,
        description=(
            "Optional case-insensitive substring filter applied to channel "
            "names client-side. Useful for 'find channels matching X'."
        ),
    )
    exclude_archived: bool = Field(default=True)
    cursor: str | None = Field(
        default=None, description="Pagination cursor returned by a previous call."
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Channels per page (Slack tier-2; default 100 is sane).",
    )


class ConversationsListOutput(BaseModel):
    channels: list[_SlackChannel]
    next_cursor: str | None = None


class ConversationsList(Tool[ConversationsListInput, ConversationsListOutput]):
    name = "slack.conversations.list"
    description = (
        "List channels in the workspace, optionally filtered by name "
        "substring. Use to answer 'is there a #design channel?' or "
        "'list all archived channels'. Cheaper than searching messages "
        "when you only need channel-level metadata."
    )
    provider = "slack"
    Input = ConversationsListInput
    Output = ConversationsListOutput

    async def run(
        self, ctx: ToolContext, args: ConversationsListInput
    ) -> ConversationsListOutput:
        try:
            token = await _bot_token_for(ctx.tenant_id)
        except SlackTokenMissing as exc:
            raise ToolError(str(exc)) from exc

        client = AsyncWebClient(token=token)
        kwargs: dict[str, Any] = {
            "types": args.types,
            "exclude_archived": args.exclude_archived,
            "limit": args.limit,
        }
        if args.cursor:
            kwargs["cursor"] = args.cursor

        try:
            resp = await client.conversations_list(**kwargs)
        except SlackApiError as exc:
            err = (exc.response.data or {}).get("error", str(exc))
            raise ToolError(f"slack.conversations.list failed: {err}") from exc

        assert isinstance(resp.data, dict)
        raw = resp.data.get("channels") or []
        channels = [_to_channel(c) for c in raw]
        if args.name_filter:
            needle = args.name_filter.lower()
            channels = [c for c in channels if c.name and needle in c.name.lower()]
        next_cursor = (resp.data.get("response_metadata") or {}).get("next_cursor") or None
        return ConversationsListOutput(channels=channels, next_cursor=next_cursor or None)


# -----------------------------------------------------------------------------


class ConversationsInfoInput(BaseModel):
    channel_id: str = Field(description="Channel/DM/group id to look up.")
    include_num_members: bool = Field(default=True)


class ConversationsInfoOutput(BaseModel):
    channel: _SlackChannel


class ConversationsInfo(Tool[ConversationsInfoInput, ConversationsInfoOutput]):
    name = "slack.conversations.info"
    description = (
        "Fetch metadata for a single Slack channel — name, topic, purpose, "
        "member count, archived/private flags. Use after `conversations.list` "
        "narrowed things down, or whenever you have a channel id and need "
        "human-readable context."
    )
    provider = "slack"
    Input = ConversationsInfoInput
    Output = ConversationsInfoOutput

    async def run(
        self, ctx: ToolContext, args: ConversationsInfoInput
    ) -> ConversationsInfoOutput:
        try:
            token = await _bot_token_for(ctx.tenant_id)
        except SlackTokenMissing as exc:
            raise ToolError(str(exc)) from exc

        client = AsyncWebClient(token=token)
        try:
            resp = await client.conversations_info(
                channel=args.channel_id, include_num_members=args.include_num_members
            )
        except SlackApiError as exc:
            err = (exc.response.data or {}).get("error", str(exc))
            raise ToolError(f"slack.conversations.info failed: {err}") from exc

        assert isinstance(resp.data, dict)
        return ConversationsInfoOutput(channel=_to_channel(resp.data.get("channel") or {}))


# -----------------------------------------------------------------------------


class ConversationsInviteInput(BaseModel):
    channel_id: str = Field(description="Target channel/group id.")
    user_ids: list[str] = Field(
        description="Slack user ids to invite (max 30 per call).",
        min_length=1,
        max_length=30,
    )


class ConversationsInviteOutput(BaseModel):
    channel_id: str
    invited: list[str]


class ConversationsInvite(Tool[ConversationsInviteInput, ConversationsInviteOutput]):
    name = "slack.conversations.invite"
    description = (
        "Invite one or more users to an existing channel. Use to add "
        "stakeholders to a project channel, e.g. 'add @alice and @bob "
        "to #project-atlas'. REQUIRES APPROVAL: include in a "
        "submit_plan_for_approval plan — invites are visible to the channel."
    )
    provider = "slack"
    requires_approval = True
    Input = ConversationsInviteInput
    Output = ConversationsInviteOutput

    async def run(
        self, ctx: ToolContext, args: ConversationsInviteInput
    ) -> ConversationsInviteOutput:
        try:
            token = await _bot_token_for(ctx.tenant_id)
        except SlackTokenMissing as exc:
            raise ToolError(str(exc)) from exc

        client = AsyncWebClient(token=token)
        try:
            await client.conversations_invite(
                channel=args.channel_id, users=",".join(args.user_ids)
            )
        except SlackApiError as exc:
            err = (exc.response.data or {}).get("error", str(exc))
            # `already_in_channel` for an individual user shouldn't fail the
            # whole batch, but the Slack API treats it as a single error code
            # for the whole call. Best-effort: surface a helpful message.
            raise ToolError(f"slack.conversations.invite failed: {err}") from exc

        return ConversationsInviteOutput(
            channel_id=args.channel_id, invited=list(args.user_ids)
        )


# -----------------------------------------------------------------------------


class ConversationsCreateInput(BaseModel):
    name: str = Field(
        description=(
            "Channel name (no `#`). Slack normalizes to lowercase, "
            "max 80 chars, only letters/digits/hyphens/underscores."
        ),
        min_length=1,
        max_length=80,
    )
    is_private: bool = Field(
        default=False,
        description="If true, creates a private channel (group); else public.",
    )


class ConversationsCreateOutput(BaseModel):
    channel: _SlackChannel


class ConversationsCreate(Tool[ConversationsCreateInput, ConversationsCreateOutput]):
    name = "slack.conversations.create"
    description = (
        "Create a new Slack channel (public or private). Use for "
        "'spin up #project-atlas for the new launch'. The bot is "
        "automatically a member after creation. REQUIRES APPROVAL: "
        "include in a submit_plan_for_approval plan."
    )
    provider = "slack"
    requires_approval = True
    Input = ConversationsCreateInput
    Output = ConversationsCreateOutput

    async def run(
        self, ctx: ToolContext, args: ConversationsCreateInput
    ) -> ConversationsCreateOutput:
        try:
            token = await _bot_token_for(ctx.tenant_id)
        except SlackTokenMissing as exc:
            raise ToolError(str(exc)) from exc

        client = AsyncWebClient(token=token)
        try:
            resp = await client.conversations_create(
                name=args.name, is_private=args.is_private
            )
        except SlackApiError as exc:
            err = (exc.response.data or {}).get("error", str(exc))
            raise ToolError(f"slack.conversations.create failed: {err}") from exc

        assert isinstance(resp.data, dict)
        return ConversationsCreateOutput(channel=_to_channel(resp.data.get("channel") or {}))


default_registry.register(ConversationsHistory())
default_registry.register(ConversationsReplies())
default_registry.register(ConversationsOpen())
default_registry.register(ConversationsList())
default_registry.register(ConversationsInfo())
default_registry.register(ConversationsInvite())
default_registry.register(ConversationsCreate())
