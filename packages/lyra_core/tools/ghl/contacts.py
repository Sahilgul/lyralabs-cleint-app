"""GHL Contacts: search + create."""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field

from ..base import Tool, ToolContext
from ..registry import default_registry
from .client import GhlClient


class GhlContact(BaseModel):
    id: str
    name: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    phone: str | None = None
    tags: list[str] = []


# --- Search ------------------------------------------------------------------


class GhlContactsSearchInput(BaseModel):
    query: str = Field(description="Free-text query: name, email, or phone.")
    limit: int = Field(default=10, le=50)


class GhlContactsSearchOutput(BaseModel):
    contacts: list[GhlContact]
    count: int


class GhlContactsSearch(Tool[GhlContactsSearchInput, GhlContactsSearchOutput]):
    name = "ghl.contacts.search"
    description = "Search GoHighLevel contacts by name/email/phone. Read-only."
    provider = "ghl"
    Input = GhlContactsSearchInput
    Output = GhlContactsSearchOutput

    async def run(self, ctx: ToolContext, args: GhlContactsSearchInput) -> GhlContactsSearchOutput:
        creds = await ctx.creds_lookup("ghl")
        client = GhlClient(creds)

        body = {
            "locationId": client.location_id,
            "query": args.query,
            "pageLimit": args.limit,
        }
        data = await client.request("POST", "/contacts/search", json=body)

        contacts: list[GhlContact] = []
        for c in data.get("contacts", [])[: args.limit]:
            contacts.append(
                GhlContact(
                    id=c.get("id", ""),
                    name=c.get("contactName") or c.get("name"),
                    first_name=c.get("firstName") or c.get("firstNameLowerCase"),
                    last_name=c.get("lastName") or c.get("lastNameLowerCase"),
                    email=c.get("email"),
                    phone=c.get("phone"),
                    tags=c.get("tags", []) or [],
                )
            )
        return GhlContactsSearchOutput(contacts=contacts, count=len(contacts))


# --- Create ------------------------------------------------------------------


class GhlContactsCreateInput(BaseModel):
    first_name: str
    last_name: str | None = None
    email: EmailStr | None = None
    phone: str | None = None
    tags: list[str] = Field(default_factory=list)
    source: str | None = "Coworker AI"


class GhlContactsCreateOutput(BaseModel):
    id: str
    email: str | None = None


class GhlContactsCreate(Tool[GhlContactsCreateInput, GhlContactsCreateOutput]):
    name = "ghl.contacts.create"
    description = (
        "Create a new contact in GoHighLevel. Required: first_name. At least one of email/phone."
    )
    provider = "ghl"
    requires_approval = True
    Input = GhlContactsCreateInput
    Output = GhlContactsCreateOutput

    async def run(self, ctx: ToolContext, args: GhlContactsCreateInput) -> GhlContactsCreateOutput:
        if not args.email and not args.phone:
            from ..base import ToolError

            raise ToolError("at least one of email or phone is required")

        creds = await ctx.creds_lookup("ghl")
        client = GhlClient(creds)
        if ctx.dry_run:
            return GhlContactsCreateOutput(
                id="dry-run", email=str(args.email) if args.email else None
            )

        body = {
            "locationId": client.location_id,
            "firstName": args.first_name,
            "lastName": args.last_name,
            "email": str(args.email) if args.email else None,
            "phone": args.phone,
            "tags": args.tags,
            "source": args.source,
        }
        body = {k: v for k, v in body.items() if v is not None}

        data = await client.request("POST", "/contacts/", json=body)
        contact = data.get("contact", data)
        return GhlContactsCreateOutput(id=contact.get("id", ""), email=contact.get("email"))


default_registry.register(GhlContactsSearch())
default_registry.register(GhlContactsCreate())
