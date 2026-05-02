"""Slack users tools: users.info + users.list.

The `users.list` tool is what lets ARLO answer "what's my name?" by
mapping the inbound `user_id` (e.g. U06ABCDEF) back to a display name
when it isn't already in workspace_facts. The `users.info` tool is the
single-user variant -- call it after a search/list narrowed down the
user, or when the model already has the user_id from a Slack event.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from ..base import Tool, ToolContext, ToolError
from ..registry import default_registry
from ._client import SlackTokenMissing, _bot_token_for


class _SlackUser(BaseModel):
    id: str
    name: str | None = None
    real_name: str | None = None
    display_name: str | None = None
    email: str | None = None
    tz: str | None = None
    is_bot: bool = False
    is_deleted: bool = False
    title: str | None = None


class UsersInfoInput(BaseModel):
    user_id: str = Field(description="Slack user id, e.g. 'U0123ABCD'.")


class UsersInfoOutput(BaseModel):
    user: _SlackUser


class UsersInfo(Tool[UsersInfoInput, UsersInfoOutput]):
    name = "slack.users.info"
    description = (
        "Look up a Slack user's profile (display name, real name, email, "
        "title, timezone) by user id. Use this to translate a 'U...' id "
        "into a human-readable name before you reply."
    )
    provider = "slack"
    Input = UsersInfoInput
    Output = UsersInfoOutput

    async def run(self, ctx: ToolContext, args: UsersInfoInput) -> UsersInfoOutput:
        try:
            token = await _bot_token_for(ctx.tenant_id)
        except SlackTokenMissing as exc:
            raise ToolError(str(exc)) from exc

        client = AsyncWebClient(token=token)
        try:
            resp = await client.users_info(user=args.user_id)
        except SlackApiError as exc:
            err = (exc.response.data or {}).get("error", str(exc))
            raise ToolError(f"slack.users.info failed: {err}") from exc

        return UsersInfoOutput(user=_to_user(resp.data.get("user") or {}))


# -----------------------------------------------------------------------------


class UsersListInput(BaseModel):
    cursor: str | None = Field(
        default=None,
        description=(
            "Pagination cursor returned by a previous call. Omit on the first call."
        ),
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=200,
        description="Members per page (Slack rate-limits this; default 100 is sane).",
    )


class UsersListOutput(BaseModel):
    members: list[_SlackUser]
    next_cursor: str | None = None


class UsersList(Tool[UsersListInput, UsersListOutput]):
    name = "slack.users.list"
    description = (
        "List the workspace's members (paginated). Use to resolve a name like "
        "'sahil' to a Slack user id when the user mentions someone by name."
    )
    provider = "slack"
    Input = UsersListInput
    Output = UsersListOutput

    async def run(self, ctx: ToolContext, args: UsersListInput) -> UsersListOutput:
        try:
            token = await _bot_token_for(ctx.tenant_id)
        except SlackTokenMissing as exc:
            raise ToolError(str(exc)) from exc

        client = AsyncWebClient(token=token)
        kwargs: dict[str, Any] = {"limit": args.limit}
        if args.cursor:
            kwargs["cursor"] = args.cursor

        try:
            resp = await client.users_list(**kwargs)
        except SlackApiError as exc:
            err = (exc.response.data or {}).get("error", str(exc))
            raise ToolError(f"slack.users.list failed: {err}") from exc

        members = [_to_user(u) for u in resp.data.get("members", []) or []]
        next_cursor = (
            (resp.data.get("response_metadata") or {}).get("next_cursor") or None
        )
        return UsersListOutput(members=members, next_cursor=next_cursor or None)


def _to_user(u: dict[str, Any]) -> _SlackUser:
    profile = u.get("profile") or {}
    return _SlackUser(
        id=u.get("id", ""),
        name=u.get("name"),
        real_name=u.get("real_name") or profile.get("real_name"),
        display_name=profile.get("display_name") or None,
        email=profile.get("email"),
        tz=u.get("tz"),
        is_bot=bool(u.get("is_bot")),
        is_deleted=bool(u.get("deleted")),
        title=profile.get("title"),
    )


default_registry.register(UsersInfo())
default_registry.register(UsersList())
