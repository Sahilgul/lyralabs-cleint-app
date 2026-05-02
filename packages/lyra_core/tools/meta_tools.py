"""Built-in meta-tools always available to the agent.

`discover_tools` — keyword-scores the registry and returns matching tool
schemas so the agent can find tools without seeing all 50+ upfront.
Upgrade to embedding similarity in v2 when registry grows large enough
to make keyword overlap ambiguous.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from ..common.logging import get_logger
from .base import Tool, ToolContext, TrustTier
from .registry import default_registry

log = get_logger(__name__)

_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "to", "and", "or", "for", "of", "in", "with",
        "i", "want", "need", "please", "help", "can", "you", "how", "do",
        "get", "use", "using", "by", "from", "is", "it", "that", "this",
    }
)


class DiscoverToolsInput(BaseModel):
    intent: str = Field(
        description=(
            "Natural-language description of what you want to do, e.g. "
            "'send SMS to contact', 'list pipeline deals', 'search Slack messages'."
        )
    )
    limit: int = Field(default=10, ge=1, le=30)


class DiscoverToolsOutput(BaseModel):
    tools: list[dict[str, Any]] = Field(
        description="OpenAI function schemas for the top matching tools"
    )


class DiscoverToolsTool(Tool[DiscoverToolsInput, DiscoverToolsOutput]):
    name: ClassVar[str] = "discover_tools"
    description: ClassVar[str] = (
        "Search the tool registry by intent. Call this before any task to find "
        "relevant tools and their exact argument schemas. Do NOT guess tool names "
        "— always call discover_tools first if you are unsure what is available."
    )
    requires_approval: ClassVar[bool] = False
    trust_tier: ClassVar[TrustTier] = TrustTier.LOW
    provider: ClassVar[str] = ""
    Input = DiscoverToolsInput
    Output = DiscoverToolsOutput

    async def run(self, ctx: ToolContext, args: DiscoverToolsInput) -> DiscoverToolsOutput:
        query_tokens = {
            w.lower()
            for w in args.intent.replace("_", " ").replace("-", " ").split()
            if w.lower() not in _STOP_WORDS and len(w) > 1
        }
        scored: list[tuple[int, Any]] = []
        for tool in default_registry.all():
            if tool.name == self.name:
                continue
            target = (tool.name + " " + tool.description).lower()
            score = sum(1 for tok in query_tokens if tok in target)
            if score > 0:
                scored.append((score, tool))

        scored.sort(key=lambda x: -x[0])
        top = [t for _, t in scored[: args.limit]]
        schemas = [t.to_openai_schema() for t in top]
        log.info("discover_tools.result", intent=args.intent[:80], matched=len(top))
        return DiscoverToolsOutput(tools=schemas)


# Register at import time so it's always in the registry.
discover_tools = DiscoverToolsTool()
default_registry.register(discover_tools)
