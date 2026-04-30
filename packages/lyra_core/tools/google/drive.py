"""Google Drive tools: search + read."""

from __future__ import annotations

import asyncio
import io
from typing import Any

from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from pydantic import BaseModel, Field

from ..base import Tool, ToolContext, ToolError
from ..registry import default_registry
from ._client import drive_service


class DriveSearchInput(BaseModel):
    query: str = Field(
        description=(
            "Free-text. Translated into Drive `q=fullText contains 'X'` plus optional "
            "mime-type filters from `mime_type`."
        )
    )
    mime_type: str | None = Field(
        default=None,
        description="Optional Drive MIME type filter, e.g. application/vnd.google-apps.spreadsheet",
    )
    page_size: int = Field(default=10, le=50)


class DriveFile(BaseModel):
    id: str
    name: str
    mime_type: str
    modified_time: str | None = None
    web_view_link: str | None = None


class DriveSearchOutput(BaseModel):
    files: list[DriveFile]


class DriveSearch(Tool[DriveSearchInput, DriveSearchOutput]):
    name = "google.drive.search"
    description = "Search the user's Google Drive by full-text query and optional MIME type."
    provider = "google"
    Input = DriveSearchInput
    Output = DriveSearchOutput

    async def run(self, ctx: ToolContext, args: DriveSearchInput) -> DriveSearchOutput:
        creds = await ctx.creds_lookup("google")
        svc = drive_service(creds)

        q_parts: list[str] = [f"fullText contains '{args.query.replace(chr(39), '')}'"]
        if args.mime_type:
            q_parts.append(f"mimeType = '{args.mime_type}'")
        q_parts.append("trashed = false")
        q = " and ".join(q_parts)

        def _call() -> dict[str, Any]:
            return (
                svc.files()
                .list(
                    q=q,
                    pageSize=args.page_size,
                    fields="files(id,name,mimeType,modifiedTime,webViewLink)",
                )
                .execute()
            )

        try:
            resp = await asyncio.to_thread(_call)
        except HttpError as exc:
            raise ToolError(f"Drive search failed: {exc}") from exc

        files = [
            DriveFile(
                id=f["id"],
                name=f["name"],
                mime_type=f["mimeType"],
                modified_time=f.get("modifiedTime"),
                web_view_link=f.get("webViewLink"),
            )
            for f in resp.get("files", [])
        ]
        return DriveSearchOutput(files=files)


# -----------------------------------------------------------------------------


class DriveReadInput(BaseModel):
    file_id: str
    export_mime: str | None = Field(
        default=None,
        description=(
            "For Google-native docs (Doc/Sheet/Slide), the export MIME type. "
            "Default text/plain for Docs, text/csv for Sheets."
        ),
    )


class DriveReadOutput(BaseModel):
    file_id: str
    mime_type: str
    name: str
    content_text: str
    truncated: bool = False


class DriveRead(Tool[DriveReadInput, DriveReadOutput]):
    name = "google.drive.read"
    description = (
        "Download a Drive file by id. Google-native files are exported to text/plain or text/csv."
    )
    provider = "google"
    Input = DriveReadInput
    Output = DriveReadOutput

    _MAX_BYTES = 1_000_000  # 1 MB safety cap to avoid blowing context

    async def run(self, ctx: ToolContext, args: DriveReadInput) -> DriveReadOutput:
        creds = await ctx.creds_lookup("google")
        svc = drive_service(creds)

        def _meta() -> dict[str, Any]:
            return svc.files().get(fileId=args.file_id, fields="id,name,mimeType").execute()

        meta = await asyncio.to_thread(_meta)
        mime = meta["mimeType"]

        export_map = {
            "application/vnd.google-apps.document": "text/plain",
            "application/vnd.google-apps.spreadsheet": "text/csv",
            "application/vnd.google-apps.presentation": "text/plain",
        }

        def _download() -> bytes:
            buf = io.BytesIO()
            if mime in export_map:
                req = svc.files().export_media(
                    fileId=args.file_id, mimeType=args.export_mime or export_map[mime]
                )
            else:
                req = svc.files().get_media(fileId=args.file_id)
            downloader = MediaIoBaseDownload(buf, req, chunksize=256 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()
                if buf.tell() > DriveRead._MAX_BYTES:
                    break
            return buf.getvalue()

        try:
            raw = await asyncio.to_thread(_download)
        except HttpError as exc:
            raise ToolError(f"Drive read failed: {exc}") from exc

        truncated = len(raw) > DriveRead._MAX_BYTES
        if truncated:
            raw = raw[: DriveRead._MAX_BYTES]
        text = raw.decode("utf-8", errors="replace")

        return DriveReadOutput(
            file_id=args.file_id,
            mime_type=mime,
            name=meta["name"],
            content_text=text,
            truncated=truncated,
        )


default_registry.register(DriveSearch())
default_registry.register(DriveRead())
