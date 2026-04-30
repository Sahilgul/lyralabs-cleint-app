"""Google Docs tools — minimal: create from a body of text/markdown."""

from __future__ import annotations

import asyncio
from typing import Any

from googleapiclient.errors import HttpError
from pydantic import BaseModel, Field

from ..base import Tool, ToolContext, ToolError
from ..registry import default_registry
from ._client import docs_service, drive_service


class DocsCreateInput(BaseModel):
    title: str
    body_text: str = Field(description="Plain text or simple markdown. Inserted as the doc body.")
    folder_id: str | None = Field(
        default=None,
        description="Optional Drive folder id to place the doc in. Otherwise: My Drive root.",
    )


class DocsCreateOutput(BaseModel):
    document_id: str
    title: str
    web_view_link: str


class DocsCreate(Tool[DocsCreateInput, DocsCreateOutput]):
    name = "google.docs.create"
    description = "Create a new Google Doc with the supplied title and body text."
    provider = "google"
    requires_approval = True
    Input = DocsCreateInput
    Output = DocsCreateOutput

    async def run(self, ctx: ToolContext, args: DocsCreateInput) -> DocsCreateOutput:
        creds = await ctx.creds_lookup("google")
        docs = docs_service(creds)
        drive = drive_service(creds)

        if ctx.dry_run:
            return DocsCreateOutput(
                document_id="dry-run",
                title=args.title,
                web_view_link="https://docs.google.com/document/d/dry-run/edit",
            )

        def _create_doc() -> dict[str, Any]:
            return docs.documents().create(body={"title": args.title}).execute()

        def _insert_body(doc_id: str) -> None:
            docs.documents().batchUpdate(
                documentId=doc_id,
                body={
                    "requests": [
                        {"insertText": {"location": {"index": 1}, "text": args.body_text}}
                    ]
                },
            ).execute()

        def _move(doc_id: str, folder_id: str) -> None:
            file = drive.files().get(fileId=doc_id, fields="parents").execute()
            prev = ",".join(file.get("parents", []))
            drive.files().update(
                fileId=doc_id, addParents=folder_id, removeParents=prev, fields="id, parents"
            ).execute()

        try:
            doc = await asyncio.to_thread(_create_doc)
            await asyncio.to_thread(_insert_body, doc["documentId"])
            if args.folder_id:
                await asyncio.to_thread(_move, doc["documentId"], args.folder_id)
        except HttpError as exc:
            raise ToolError(f"Docs create failed: {exc}") from exc

        return DocsCreateOutput(
            document_id=doc["documentId"],
            title=doc["title"],
            web_view_link=f"https://docs.google.com/document/d/{doc['documentId']}/edit",
        )


default_registry.register(DocsCreate())
