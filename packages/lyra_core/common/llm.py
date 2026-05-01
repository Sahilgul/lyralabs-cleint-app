"""LLM call surface, multi-provider via LiteLLM.

Two tiers (callers pick one):
  - PRIMARY: agent reasoning, critic, executor reasoning.
  - CHEAP:   reserved for cheap helper calls (e.g. cheap-pass critic).

Per-tier model + credentials are resolved at call time by
`lyra_core.llm.router.resolve(tier)`. The router reads the active
`(provider, model_id, api_key, api_base, extra_config)` from Postgres
(populated by the super-admin via the admin UI), with a 30-second
in-process cache and an env-var fallback for fresh deployments.

This means the agent can be hot-switched between Qwen / DeepSeek /
OpenAI / Anthropic / Gemini / Moonshot / MiniMax / Z.AI from the admin
panel without any redeploy. See `packages/lyra_core/llm/catalog.py`
for the catalog of supported providers and known model IDs.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from litellm import acompletion
from litellm.types.utils import ModelResponse

from ..llm.router import resolve
from .logging import get_logger

logger = get_logger(__name__)


class ModelTier(StrEnum):
    PRIMARY = "primary"
    CHEAP = "cheap"


async def chat(
    *,
    tier: ModelTier,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    metadata: dict[str, Any] | None = None,
) -> ModelResponse:
    """Single chat completion. Caller passes OpenAI-format messages.

    Resolves the active (model, api_key, api_base, ...) from the runtime
    router so admin-panel switches take effect on the next call (modulo
    the router's 30s cache TTL).

    `api_key` and `api_base` are passed *per call* rather than via env
    vars -- this is the prefork-safe pattern: no shared mutable state
    between Celery workers, and no surprises if one tier is on
    OpenAI-compat (Z.AI, Moonshot, MiniMax) while another is on a
    direct provider (Anthropic, Gemini).
    """
    resolved = await resolve(tier.value)  # type: ignore[arg-type]

    kwargs: dict[str, Any] = {
        "model": resolved.model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "metadata": metadata or {},
    }
    if resolved.api_key:
        kwargs["api_key"] = resolved.api_key
    if resolved.api_base:
        kwargs["api_base"] = resolved.api_base
    for k, v in resolved.extra_kwargs.items():
        # Don't let extra_config stomp the canonical model/messages args.
        if k not in {"model", "messages"}:
            kwargs[k] = v
    if tools:
        kwargs["tools"] = tools
    if response_format:
        kwargs["response_format"] = response_format

    logger.debug(
        "llm.call",
        model=resolved.model_id,
        provider=resolved.provider_key,
        source=resolved.source,
        n_messages=len(messages),
        n_tools=len(tools or []),
    )
    return await acompletion(**kwargs)


def estimate_cost(response: ModelResponse) -> float:
    """LiteLLM tracks usage cost on the response object."""
    try:
        return float(response._hidden_params.get("response_cost") or 0.0)  # type: ignore[attr-defined]
    except Exception:
        return 0.0
