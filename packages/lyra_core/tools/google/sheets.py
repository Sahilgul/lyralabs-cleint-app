"""Google Sheets tools."""

from __future__ import annotations

import asyncio
from typing import Any

from googleapiclient.errors import HttpError
from pydantic import BaseModel, Field

from ..base import Tool, ToolContext, ToolError
from ..registry import default_registry
from ._client import sheets_service


class SheetsReadInput(BaseModel):
    spreadsheet_id: str
    range_a1: str = Field(
        description="A1 range, e.g. 'Sheet1!A1:F100' or 'Leads' for the whole 'Leads' sheet."
    )


class SheetsReadOutput(BaseModel):
    spreadsheet_id: str
    range: str
    values: list[list[Any]]
    n_rows: int
    n_cols: int


class SheetsRead(Tool[SheetsReadInput, SheetsReadOutput]):
    name = "google.sheets.read"
    description = "Read a range from a Google Sheet. Returns rows as list-of-lists."
    provider = "google"
    Input = SheetsReadInput
    Output = SheetsReadOutput

    async def run(self, ctx: ToolContext, args: SheetsReadInput) -> SheetsReadOutput:
        creds = await ctx.creds_lookup("google")
        svc = sheets_service(creds)

        def _call() -> dict[str, Any]:
            return (
                svc.spreadsheets()
                .values()
                .get(spreadsheetId=args.spreadsheet_id, range=args.range_a1)
                .execute()
            )

        try:
            resp = await asyncio.to_thread(_call)
        except HttpError as exc:
            raise ToolError(f"Sheets read failed: {exc}") from exc

        values = resp.get("values", [])
        n_rows = len(values)
        n_cols = max((len(r) for r in values), default=0)
        return SheetsReadOutput(
            spreadsheet_id=args.spreadsheet_id,
            range=resp.get("range", args.range_a1),
            values=values,
            n_rows=n_rows,
            n_cols=n_cols,
        )


# -----------------------------------------------------------------------------


class SheetsAppendInput(BaseModel):
    spreadsheet_id: str
    range_a1: str = Field(description="Anchor range, e.g. 'Sheet1!A1' or 'Leads'.")
    rows: list[list[Any]] = Field(description="Rows to append. Each row is a list of cell values.")
    value_input_option: str = Field(default="USER_ENTERED")


class SheetsAppendOutput(BaseModel):
    updated_range: str
    updated_rows: int


class SheetsAppend(Tool[SheetsAppendInput, SheetsAppendOutput]):
    name = "google.sheets.append"
    description = "Append rows to the bottom of a Google Sheet range."
    provider = "google"
    requires_approval = True
    Input = SheetsAppendInput
    Output = SheetsAppendOutput

    async def run(self, ctx: ToolContext, args: SheetsAppendInput) -> SheetsAppendOutput:
        creds = await ctx.creds_lookup("google")
        svc = sheets_service(creds)

        if ctx.dry_run:
            return SheetsAppendOutput(
                updated_range=f"{args.range_a1} (dry-run)",
                updated_rows=len(args.rows),
            )

        def _call() -> dict[str, Any]:
            return (
                svc.spreadsheets()
                .values()
                .append(
                    spreadsheetId=args.spreadsheet_id,
                    range=args.range_a1,
                    valueInputOption=args.value_input_option,
                    body={"values": args.rows},
                )
                .execute()
            )

        try:
            resp = await asyncio.to_thread(_call)
        except HttpError as exc:
            raise ToolError(f"Sheets append failed: {exc}") from exc

        updates = resp.get("updates", {})
        return SheetsAppendOutput(
            updated_range=updates.get("updatedRange", ""),
            updated_rows=int(updates.get("updatedRows", 0)),
        )


default_registry.register(SheetsRead())
default_registry.register(SheetsAppend())
