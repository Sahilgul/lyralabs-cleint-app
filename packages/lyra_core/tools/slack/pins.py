"""Slack pins tool: pin a message to its channel.

Pins surface "this matters" to everyone in a channel without sending a
reply. Use to mark decisions, key links, or canonical specs. Classified
LOW under the permissive policy — pinning is reversible and visible to
the same people who already see the channel.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from ..base import Tool, ToolContext, ToolError, TrustTier
from ..registry import default_registry
from ._client import SlackTokenMissing, _bot_token_for


class PinsAddInput(BaseModel):
    channel_id: str = Field(description="Channel/group id where the message lives.")
    timestamp: str = Field(description="ts of the message to pin (e.g. '1717174800.123456').")


class PinsAddOutput(BaseModel):
    ok: bool = True


class PinsAdd(Tool[PinsAddInput, PinsAddOutput]):
    name = "slack.pins.add"
    description = (
        "Pin a Slack message to its channel so it shows up in the channel's "
        "pinned items. Use for 'pin the decision', 'pin the spec doc'. "
        "`already_pinned` errors are swallowed — desired end state achieved."
    )
    provider = "slack"
    requires_approval = False
    trust_tier = TrustTier.LOW
    Input = PinsAddInput
    Output = PinsAddOutput

    async def run(self, ctx: ToolContext, args: PinsAddInput) -> PinsAddOutput:
        try:
            token = await _bot_token_for(ctx.tenant_id)
        except SlackTokenMissing as exc:
            raise ToolError(str(exc)) from exc

        client = AsyncWebClient(token=token)
        try:
            await client.pins_add(channel=args.channel_id, timestamp=args.timestamp)
        except SlackApiError as exc:
            err = (exc.response.data or {}).get("error", str(exc))
            if err == "already_pinned":
                return PinsAddOutput(ok=True)
            raise ToolError(f"slack.pins.add failed: {err}") from exc

        return PinsAddOutput(ok=True)


default_registry.register(PinsAdd())
