"""MCP server configs, tool classification, and discovery.

`discover_and_register_tools` is the single entry point for the worker:
called once at job start with fresh credentials, it populates the
default_registry with McpToolAdapter instances for every tool the
MCP server exposes. Results are cached for 24 hours per
(tenant_id, client_id, server_key) to avoid hammering the MCP servers.

Trust classification defaults to LOW (read-only, no approval).
Explicit allow-lists drive MEDIUM and HIGH — anything not on a list
stays LOW, which means the write-tool guard in tool_node.py will let
the agent call it freely. If you're unsure whether a tool is a write,
check the 'unclassified_write_candidate' warning in the logs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..common.logging import get_logger
from .base import RiskProfile, TrustTier

if TYPE_CHECKING:
    from .registry import ToolRegistry

log = get_logger(__name__)

# Tokens in tool names that suggest a write operation.
# Used only for logging candidates, not for classification.
_WRITE_VERB_HINTS = frozenset(
    {"create", "update", "upsert", "add", "remove", "delete", "send", "post", "book", "edit", "set"}
)


@dataclass
class McpServerConfig:
    url: str
    transport: str = "http"
    provider: str = ""
    # Tool names on this list → MEDIUM (Approve/Reject button)
    write_tools: frozenset[str] = field(default_factory=frozenset)
    # Tool names on this list → HIGH ("I confirm" text gate)
    high_risk_tools: frozenset[str] = field(default_factory=frozenset)


MCP_SERVER_CONFIGS: dict[str, McpServerConfig] = {
    "ghl": McpServerConfig(
        url="https://services.leadconnectorhq.com/mcp/",
        transport="http",
        provider="ghl",
        write_tools=frozenset(
            {
                "contacts_create-contact",
                "contacts_update-contact",
                "contacts_upsert-contact",
                "contacts_add-tags",
                "contacts_remove-tags",
                "calendars_book-appointment",
                "opportunities_update-opportunity",
                "social-media-posting_create-post",
                "social-media-posting_edit-post",
                "blogs_create-blog-post",
                "blogs_update-blog-post",
                "emails_create-template",
            }
        ),
        high_risk_tools=frozenset(
            {
                "conversations_send-a-new-message",  # sends SMS/email to real people
            }
        ),
    ),
    "slack": McpServerConfig(
        url="https://mcp.slack.com/mcp",
        transport="http",
        provider="slack",
        write_tools=frozenset({"send_message", "create_canvas", "update_canvas"}),
    ),
}


def _classify_mcp_tool(mcp_tool_name: str, config: McpServerConfig) -> RiskProfile:
    """Default LOW. Explicit allow-lists for MEDIUM and HIGH only."""
    if mcp_tool_name in config.high_risk_tools:
        return RiskProfile(tier=TrustTier.HIGH, reversibility="irreversible", blast_radius="bulk")
    if mcp_tool_name in config.write_tools:
        return RiskProfile(tier=TrustTier.MEDIUM, reversibility="reversible", blast_radius="single")
    return RiskProfile(tier=TrustTier.LOW, reversibility="reversible", blast_radius="single")


# 24-hour discovery cache: (tenant_id, client_id, server_key) → (tool_names, expires_at)
_DISCOVERY_CACHE: dict[tuple, tuple] = {}


async def discover_and_register_tools(
    server_key: str,
    tenant_id: str,
    client_id: str | None,
    headers: dict[str, str],
    registry: "ToolRegistry",
) -> list[str]:
    """Discover MCP tools and register them in the registry. Idempotent; 24h cache.

    Logs 'unclassified_write_candidate' for tools that look like writes but are
    not on the explicit allow-lists — helps the operator keep the lists current.
    """
    cache_key = (tenant_id, client_id, server_key)
    cached = _DISCOVERY_CACHE.get(cache_key)
    if cached and cached[1] > time.monotonic():
        return list(cached[0])

    config = MCP_SERVER_CONFIGS.get(server_key)
    if config is None:
        raise ValueError(f"Unknown MCP server: {server_key!r}")

    from langchain_mcp_adapters.client import MultiServerMCPClient

    from .mcp_adapter import _make_mcp_tool_adapter

    mc = MultiServerMCPClient(
        {server_key: {"url": config.url, "transport": config.transport, "headers": headers}}
    )
    lc_tools = await mc.get_tools()
    registered: list[str] = []

    for lc_tool in lc_tools:
        mcp_name = lc_tool.name
        profile = _classify_mcp_tool(mcp_name, config)

        # Warn about tools that look like writes but aren't classified.
        if profile.tier == TrustTier.LOW:
            parts = set(mcp_name.replace("-", "_").split("_"))
            if parts & _WRITE_VERB_HINTS:
                log.warning(
                    "mcp.unclassified_write_candidate",
                    tool=mcp_name,
                    server=server_key,
                    hint="add to write_tools or high_risk_tools if this mutates state",
                )

        adapter = _make_mcp_tool_adapter(
            lc_tool=lc_tool,
            server_key=server_key,
            mcp_tool_name=mcp_name,
            profile=profile,
            provider=config.provider,
        )
        registry.register_or_update(adapter)
        registered.append(mcp_name)

    _DISCOVERY_CACHE[cache_key] = (registered, time.monotonic() + 86400)
    log.info(
        "mcp.tools_discovered",
        server=server_key,
        count=len(registered),
        tenant=tenant_id,
        client=client_id,
    )
    return registered
