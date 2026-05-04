"""Tool abstraction.

Each Tool is:
  - Identified by a stable string `name` (used by the planner).
  - Typed by a Pydantic input + Pydantic output model.
  - Marked `requires_approval=True` for any write that mutates external state.
  - Async, side-effect-bearing, never reads global state.

The Tool's `run` is called by the executor with a ToolContext, NOT raw env.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field


class ToolError(Exception):
    """Raised by a tool when execution fails after retries."""


class ApprovalRequired(Exception):  # noqa: N818 intentional
    """Raised by an executor when the tool needs approval before mutating."""

    def __init__(self, preview: dict[str, Any]):
        super().__init__("approval required")
        self.preview = preview


class TrustTier(StrEnum):
    LOW = "low"  # read-only; auto-approved, no interrupt
    MEDIUM = "medium"  # write with limited blast radius; Approve/Reject button
    HIGH = "high"  # irreversible or bulk; requires "I confirm" text confirmation


@dataclass
class RiskProfile:
    tier: TrustTier
    reversibility: Literal["reversible", "irreversible"]
    blast_radius: Literal["single", "batch", "bulk"]
    cost_estimate_usd: float = 0.0
    rationale: str = ""


class ToolContext(BaseModel):
    """Per-call execution context. Carries the tenant id and a getter for
    integration credentials, so tools never touch the DB or env directly."""

    tenant_id: str
    job_id: str | None = None
    user_id: str | None = None
    dry_run: bool = False
    client_id: str | None = None
    # Per-(tenant, client, server_key) auth headers for MCP servers.
    # McpToolAdapter builds these from creds_lookup if not provided.
    mcp_server_headers: dict[str, dict[str, str]] | None = None
    # When True, tools should return a description of what they would do
    # without performing actual side-effects. Used by the rehearsal engine.
    simulation_mode: bool = False
    creds_lookup: Any = Field(default=None, description="callable: provider -> ProviderCredentials")
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class ToolResult[OutT: BaseModel](BaseModel):
    ok: bool
    data: OutT | None = None
    error: str | None = None
    cost_usd: float = 0.0


class Tool[InT: BaseModel, OutT: BaseModel](ABC):
    """Subclass and override `name`, `description`, `Input`, `Output`, `run`."""

    name: ClassVar[str]
    description: ClassVar[str]
    requires_approval: ClassVar[bool] = False
    provider: ClassVar[str] = ""  # "google" | "ghl" | "" (no creds needed)
    trust_tier: ClassVar[TrustTier] = TrustTier.MEDIUM
    blast_radius: ClassVar[Literal["single", "batch", "bulk"]] = "single"

    Input: ClassVar[type[BaseModel]]
    Output: ClassVar[type[BaseModel]]

    @abstractmethod
    async def run(self, ctx: ToolContext, args: InT) -> OutT:
        """Implement the side-effecting work."""

    def validate_args(self, args: dict[str, Any]) -> str | None:
        """Type-check `args` against this tool's input schema.

        Returns an error message if `args` would be rejected at execution
        time (missing required fields, wrong types). Returns None on success.

        Default implementation validates against `self.Input`. MCP-discovered
        tools override this because their per-instance JSON schema lives on
        the wrapped LangChain BaseTool, not on the shared `Input` model.
        """
        try:
            self.Input(**args)
        except Exception as exc:
            return str(exc)
        return None

    async def simulate(self, ctx: ToolContext, args: InT) -> str:
        """Return a human-readable preview of what run() would do.

        Uses the CHEAP LLM tier to describe the action in plain English.
        Override only when you want to fetch live data for a richer preview.
        """
        from ..common.llm import ModelTier, chat

        try:
            args_text = args.model_dump_json(indent=2)
            response = await chat(
                tier=ModelTier.CHEAP,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Tool: {self.description}\n"
                            f"Arguments:\n{args_text}\n\n"
                            "In one short plain-English sentence, describe exactly what this will do. "
                            "Use specific names/values from the arguments. No technical terms, no JSON, no code."
                        ),
                    }
                ],
                max_tokens=80,
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return ""

    async def safe_run(self, ctx: ToolContext, args: InT) -> ToolResult[OutT]:
        """Wrap `run` with error handling. Always returns a ToolResult."""
        try:
            data = await self.run(ctx, args)
            return ToolResult(ok=True, data=data)
        except ApprovalRequired:
            raise
        except ToolError as exc:
            return ToolResult(ok=False, error=str(exc))
        except Exception as exc:
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")

    def to_openai_schema(self) -> dict[str, Any]:
        """Return the OpenAI-format function-calling schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.Input.model_json_schema(),
            },
        }


CredentialsLookup = Callable[[str], Awaitable[Any]]
