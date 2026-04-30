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
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, Field


class ToolError(Exception):
    """Raised by a tool when execution fails after retries."""


class ApprovalRequired(Exception):
    """Raised by an executor when the tool needs approval before mutating."""

    def __init__(self, preview: dict[str, Any]):
        super().__init__("approval required")
        self.preview = preview


class ToolContext(BaseModel):
    """Per-call execution context. Carries the tenant id and a getter for
    integration credentials, so tools never touch the DB or env directly."""

    tenant_id: str
    job_id: str | None = None
    user_id: str | None = None
    dry_run: bool = False
    creds_lookup: Any = Field(default=None, description="callable: provider -> ProviderCredentials")
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


InT = TypeVar("InT", bound=BaseModel)
OutT = TypeVar("OutT", bound=BaseModel)


class ToolResult(BaseModel, Generic[OutT]):
    ok: bool
    data: OutT | None = None
    error: str | None = None
    cost_usd: float = 0.0


class Tool(ABC, Generic[InT, OutT]):
    """Subclass and override `name`, `description`, `Input`, `Output`, `run`."""

    name: ClassVar[str]
    description: ClassVar[str]
    requires_approval: ClassVar[bool] = False
    provider: ClassVar[str] = ""  # "google" | "ghl" | "" (no creds needed)

    Input: ClassVar[type[BaseModel]]
    Output: ClassVar[type[BaseModel]]

    @abstractmethod
    async def run(self, ctx: ToolContext, args: InT) -> OutT:
        """Implement the side-effecting work."""

    async def safe_run(self, ctx: ToolContext, args: InT) -> ToolResult[OutT]:
        """Wrap `run` with error handling. Always returns a ToolResult."""
        try:
            data = await self.run(ctx, args)
            return ToolResult(ok=True, data=data)
        except ApprovalRequired:
            raise
        except ToolError as exc:
            return ToolResult(ok=False, error=str(exc))
        except Exception as exc:  # noqa: BLE001
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
