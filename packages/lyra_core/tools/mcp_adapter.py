"""MCP tool adapter.

Wraps a LangChain-discovered MCP BaseTool in our Tool[InT, OutT] safety interface
so approval gates, trust tiers, and audit logging remain intact for all MCP tools.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel

from ..common.logging import get_logger
from .base import RiskProfile, Tool, ToolContext, ToolError, TrustTier

log = get_logger(__name__)


def _build_provider_headers(provider: str, creds: Any) -> dict[str, str]:
    """Build per-provider auth headers from a ProviderCredentials object."""
    if provider == "ghl":
        return {
            "Authorization": f"Bearer {creds.access_token}",
            "locationId": creds.external_account_id,
        }
    return {"Authorization": f"Bearer {creds.access_token}"}


class McpInput(BaseModel):
    """Fallback input model used only when an MCP tool has no discoverable
    args_schema. Real MCP tools get their per-tool Pydantic model from
    LangChain's `args_schema` and bypass this entirely.

    Configured with `extra='allow'` so a stray submission against this fallback
    keeps fields rather than silently dropping them — the previous default
    (`extra='ignore'`) was the root cause of every MCP write call sending
    `{}` to the upstream API regardless of what the LLM emitted.
    """

    model_config = {"extra": "allow"}


class McpOutput(BaseModel):
    result: Any = None


class McpToolAdapter(Tool[BaseModel, McpOutput]):
    """Wraps a discovered MCP tool in the Tool safety interface.

    Instances are created by `_make_mcp_tool_adapter` which builds a dynamic
    subclass whose `Input` ClassVar is the underlying MCP tool's args_schema
    (a Pydantic model LangChain built from the JSON schema the MCP server
    returned during discovery). That makes validation and execution use the
    real per-tool schema instead of a generic `arguments: dict` envelope —
    the latter silently dropped every populated field on Pydantic v2.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    requires_approval: ClassVar[bool] = False
    trust_tier: ClassVar[TrustTier] = TrustTier.LOW
    blast_radius: ClassVar[Literal["single", "batch", "bulk"]] = "single"
    provider: ClassVar[str] = ""
    Input = McpInput
    Output = McpOutput

    def __init__(self, lc_tool: Any, server_key: str, mcp_tool_name: str) -> None:
        self._lc_tool = lc_tool
        self._server_key = server_key
        self._mcp_tool_name = mcp_tool_name

    async def run(self, ctx: ToolContext, args: BaseModel) -> McpOutput:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        from .mcp_registry import MCP_SERVER_CONFIGS

        # Get fresh credentials on every call so we never use an expired token.
        if self.provider and ctx.creds_lookup:
            creds = await ctx.creds_lookup(self.provider)
            headers = _build_provider_headers(self.provider, creds)
        else:
            headers = (ctx.mcp_server_headers or {}).get(self._server_key, {})

        config = MCP_SERVER_CONFIGS[self._server_key]
        mc = MultiServerMCPClient(
            {
                self._server_key: {  # type: ignore[dict-item, misc]
                    "url": config.url,
                    "transport": config.transport,
                    "headers": headers,
                }
            }
        )
        lc_tools = await mc.get_tools()
        tool = next((t for t in lc_tools if t.name == self._mcp_tool_name), None)
        if tool is None:
            raise ToolError(
                f"MCP tool {self._mcp_tool_name!r} not found on server {self._server_key!r}"
            )
        # Pass every populated field through to the MCP tool. model_dump
        # excludes unset fields when the schema has optional+default fields,
        # which prevents us from sending `null` for things the MCP server
        # treats as "not provided".
        payload = args.model_dump(exclude_unset=True, exclude_none=True)
        result = await tool.ainvoke(payload)
        return McpOutput(result=result)

    async def simulate(self, ctx: ToolContext, args: BaseModel) -> str:
        import json

        payload = args.model_dump(exclude_unset=True, exclude_none=True)
        return f"Will call `{self.name}` with:\n{json.dumps(payload, indent=2, default=str)}"


def _resolve_input_schema(lc_tool: Any) -> type[BaseModel]:
    """Pull the real per-tool args schema off a LangChain BaseTool, falling
    back to McpInput if it isn't a Pydantic model.

    LangChain's MCP adapter builds `args_schema` from each MCP tool's
    inputSchema (JSON Schema) — that's what carries the required-fields
    list (email, firstName, etc.) and types we need both for LLM-facing
    function specs AND for validating step.args before execution.
    """
    schema = getattr(lc_tool, "args_schema", None)
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema
    return McpInput


def _make_mcp_tool_adapter(
    *,
    lc_tool: Any,
    server_key: str,
    mcp_tool_name: str,
    profile: RiskProfile,
    provider: str,
) -> McpToolAdapter:
    """Build a McpToolAdapter subclass with the correct ClassVars for one MCP tool.

    Critically: each adapter's `Input` is the per-tool Pydantic schema
    discovered from the MCP server, NOT the generic `McpInput` envelope.
    This is what makes:
      - the LLM's function-calling schema reflect the real fields (so the
        agent populates email/firstName/lastName instead of guessing);
      - the executor's `tool.Input(**step.args)` actually carry those
        fields through to the upstream call instead of silently dropping
        them (Pydantic v2 default `extra='ignore'`);
      - the validator catch missing required fields before the user is
        asked to approve an unexecutable plan.
    """
    cls = type(
        f"Mcp_{mcp_tool_name.replace('-', '_').replace('.', '_')}",
        (McpToolAdapter,),
        {
            "name": mcp_tool_name,
            "description": getattr(lc_tool, "description", mcp_tool_name) or mcp_tool_name,
            "requires_approval": profile.tier != TrustTier.LOW,
            "trust_tier": profile.tier,
            "blast_radius": profile.blast_radius,
            "provider": provider,
            "Input": _resolve_input_schema(lc_tool),
            "Output": McpOutput,
        },
    )
    return cls(lc_tool=lc_tool, server_key=server_key, mcp_tool_name=mcp_tool_name)
