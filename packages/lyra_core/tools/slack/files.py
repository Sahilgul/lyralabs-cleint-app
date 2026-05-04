"""Slack files tool: upload arbitrary content to a channel/DM/thread.

`poster.py` already calls `files_upload_v2` for artifacts attached to the
agent's *final reply*. This tool exposes the same capability as a
deliberate write step the model can plan — e.g. "render the report as a
PDF and post it to #leadership", or "upload this CSV to the procurement
thread".

Content is base64-encoded in the input model so it's safe to round-trip
through the OpenAI tool-call pipeline (which is JSON, not multipart).
The artifact pipeline (`tools/artifacts/*`) typically writes binary into
`ctx.extra["artifacts"]`; this tool is for cases where the LLM has the
bytes from a different source and wants to upload them by name.
"""

from __future__ import annotations

import base64
from typing import Any

from pydantic import BaseModel, Field
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from ..base import Tool, ToolContext, ToolError
from ..registry import default_registry
from ._client import SlackTokenMissing, _bot_token_for


class FilesUploadInput(BaseModel):
    channel_id: str = Field(description="Destination channel/DM/group id.")
    filename: str = Field(
        description="Filename including extension, e.g. 'q3-report.pdf'.",
        min_length=1,
        max_length=255,
    )
    content_b64: str = Field(
        description=(
            "Base64-encoded file content. Required because tool-call "
            "transports are JSON-only. For text files, base64-encode the "
            "UTF-8 bytes; for binary files, base64-encode the raw bytes."
        )
    )
    title: str | None = Field(
        default=None,
        description="Optional human-readable title shown in Slack (defaults to filename).",
    )
    initial_comment: str | None = Field(
        default=None,
        description="Optional message posted alongside the file upload.",
    )
    thread_ts: str | None = Field(
        default=None, description="If set, attaches the upload to this thread."
    )


class FilesUploadOutput(BaseModel):
    channel_id: str
    file_id: str
    permalink: str | None = None


class FilesUpload(Tool[FilesUploadInput, FilesUploadOutput]):
    name = "slack.files.upload"
    description = (
        "Upload a file (PDF, CSV, image, text, anything) to a Slack channel, "
        "DM, or thread. Use to share generated artifacts, exports, or other "
        "binary content the user wants attached. REQUIRES APPROVAL: include "
        "in a submit_plan_for_approval plan."
    )
    provider = "slack"
    requires_approval = True
    Input = FilesUploadInput
    Output = FilesUploadOutput

    async def run(self, ctx: ToolContext, args: FilesUploadInput) -> FilesUploadOutput:
        try:
            token = await _bot_token_for(ctx.tenant_id)
        except SlackTokenMissing as exc:
            raise ToolError(str(exc)) from exc

        try:
            content = base64.b64decode(args.content_b64, validate=True)
        except Exception as exc:
            raise ToolError(
                f"slack.files.upload: content_b64 is not valid base64 ({exc})"
            ) from exc

        client = AsyncWebClient(token=token)
        kwargs: dict[str, Any] = {
            "channel": args.channel_id,
            "content": content,
            "filename": args.filename,
            "title": args.title or args.filename,
        }
        if args.initial_comment is not None:
            kwargs["initial_comment"] = args.initial_comment
        if args.thread_ts is not None:
            kwargs["thread_ts"] = args.thread_ts

        try:
            resp = await client.files_upload_v2(**kwargs)
        except SlackApiError as exc:
            err = (exc.response.data or {}).get("error", str(exc))
            raise ToolError(f"slack.files.upload failed: {err}") from exc

        assert isinstance(resp.data, dict)
        # files_upload_v2 returns either `file` (single) or `files` (batch).
        file_obj: dict[str, Any] = resp.data.get("file") or {}
        if not file_obj:
            files_list = resp.data.get("files") or []
            file_obj = files_list[0] if files_list else {}

        return FilesUploadOutput(
            channel_id=args.channel_id,
            file_id=file_obj.get("id", ""),
            permalink=file_obj.get("permalink"),
        )


default_registry.register(FilesUpload())
