"""Slack reactions tools: add + remove emoji on any message.

These exist for the case where the user EXPLICITLY asks for a reaction
("react with 🚀 when the deploy ships", "drop a ✅ on Bob's message").
The system prompt instructs ARLO to default to text replies in every
other case — reactions add no context, and ARLO is meant to be a
proactive teammate, not a silent emoji bot.

Classified LOW (no approval) per the permissive trust policy: a reaction
is reversible and single-message blast radius. The previous channel-mention
auto-:eyes: behavior in `channels/slack/adapter.py` is intentionally
disabled so reactions only fire when deliberately requested.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from ..base import Tool, ToolContext, ToolError, TrustTier
from ._client import SlackTokenMissing, _bot_token_for


class _ReactionInput(BaseModel):
    channel_id: str = Field(description="Channel/DM/group id where the message lives.")
    timestamp: str = Field(
        description=(
            "The `ts` of the target message (e.g. '1717174800.123456'). Get "
            "this from a `slack.conversations.history` / `replies` result, "
            "or from the inbound event you're responding to."
        )
    )
    name: str = Field(
        description=(
            "Emoji name without colons — 'eyes', 'white_check_mark', 'rocket'. "
            "Custom workspace emoji are also valid."
        )
    )


class _ReactionOutput(BaseModel):
    ok: bool = True


class ReactionsAdd(Tool[_ReactionInput, _ReactionOutput]):
    name = "slack.reactions.add"
    description = (
        "React to a Slack message with an emoji. Use ONLY when the user "
        "explicitly asks for a reaction (e.g. 'react with 🚀 when done'). "
        "Do not call this to acknowledge or signal task completion — reply "
        "with a short value-add text instead. `already_reacted` errors are "
        "swallowed (the reaction was already there)."
    )
    provider = "slack"
    requires_approval = False
    trust_tier = TrustTier.LOW
    Input = _ReactionInput
    Output = _ReactionOutput

    async def run(self, ctx: ToolContext, args: _ReactionInput) -> _ReactionOutput:
        try:
            token = await _bot_token_for(ctx.tenant_id)
        except SlackTokenMissing as exc:
            raise ToolError(str(exc)) from exc

        client = AsyncWebClient(token=token)
        try:
            await client.reactions_add(
                channel=args.channel_id, timestamp=args.timestamp, name=args.name
            )
        except SlackApiError as exc:
            err = (exc.response.data or {}).get("error", str(exc))
            # `already_reacted` is fine — desired end state achieved.
            if err == "already_reacted":
                return _ReactionOutput(ok=True)
            raise ToolError(f"slack.reactions.add failed: {err}") from exc

        return _ReactionOutput(ok=True)


class ReactionsRemove(Tool[_ReactionInput, _ReactionOutput]):
    name = "slack.reactions.remove"
    description = (
        "Remove an emoji reaction from a Slack message. Use only when the "
        "user explicitly asks ('un-react', 'remove the ✅'). `no_reaction` "
        "errors are swallowed."
    )
    provider = "slack"
    requires_approval = False
    trust_tier = TrustTier.LOW
    Input = _ReactionInput
    Output = _ReactionOutput

    async def run(self, ctx: ToolContext, args: _ReactionInput) -> _ReactionOutput:
        try:
            token = await _bot_token_for(ctx.tenant_id)
        except SlackTokenMissing as exc:
            raise ToolError(str(exc)) from exc

        client = AsyncWebClient(token=token)
        try:
            await client.reactions_remove(
                channel=args.channel_id, timestamp=args.timestamp, name=args.name
            )
        except SlackApiError as exc:
            err = (exc.response.data or {}).get("error", str(exc))
            if err == "no_reaction":
                return _ReactionOutput(ok=True)
            raise ToolError(f"slack.reactions.remove failed: {err}") from exc

        return _ReactionOutput(ok=True)


# Reactions disabled (2026-05) — the LLM should speak, not react.
# Re-enable by uncommenting these two lines and restoring the __init__.py imports.
# default_registry.register(ReactionsAdd())
# default_registry.register(ReactionsRemove())
