"""MCP tool adapter.

Wraps a LangChain-discovered MCP BaseTool in our Tool[InT, OutT] safety interface
so approval gates, trust tiers, and audit logging remain intact for all MCP tools.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

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
    arguments: dict[str, Any] = Field(default_factory=dict)


class McpOutput(BaseModel):
    result: Any = None


class McpToolAdapter(Tool[McpInput, McpOutput]):
    """Wraps a discovered MCP tool in the Tool safety interface.

    Instances are created by `_make_mcp_tool_adapter` which builds a dynamic
    subclass carrying the correct ClassVars (name, description, trust_tier, etc.)
    so the registry and approval gate see them as normal Tool subclasses.
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

    async def run(self, ctx: ToolContext, args: McpInput) -> McpOutput:
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
            {self._server_key: {"url": config.url, "transport": config.transport, "headers": headers}}
        )
        lc_tools = await mc.get_tools()
        tool = next((t for t in lc_tools if t.name == self._mcp_tool_name), None)
        if tool is None:
            raise ToolError(
                f"MCP tool {self._mcp_tool_name!r} not found on server {self._server_key!r}"
            )
        result = await tool.ainvoke(args.arguments)
        return McpOutput(result=result)

    async def simulate(self, ctx: ToolContext, args: McpInput) -> str:
        import json

        return f"Will call `{self.name}` with:\n{json.dumps(args.arguments, indent=2)}"


def _make_mcp_tool_adapter(
    *,
    lc_tool: Any,
    server_key: str,
    mcp_tool_name: str,
    profile: RiskProfile,
    provider: str,
) -> McpToolAdapter:
    """Build a McpToolAdapter subclass with the correct ClassVars for one MCP tool."""
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
            "Input": McpInput,
            "Output": McpOutput,
        },
    )
    return cls(lc_tool=lc_tool, server_key=server_key, mcp_tool_name=mcp_tool_name)
