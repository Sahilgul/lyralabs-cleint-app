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

import re
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


# DeepSeek (and a few other providers) reject tool names that don't match
# ^[a-zA-Z0-9_-]+$. Our internal tools use dots/slashes in their names
# (`slack.conversations.history`, `ghl.contacts.search`). We sanitize the
# names sent to the LLM and reverse-map any tool_calls in the response so
# the rest of the agent code keeps using the canonical names.
_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_tool_name(name: str) -> str:
    return _SAFE_NAME_RE.sub("_", name)


def _sanitize_tools(
    tools: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Return (sanitized_tools, sanitized_to_original_name_map)."""
    name_map: dict[str, str] = {}
    sanitized: list[dict[str, Any]] = []
    for t in tools:
        fn = t.get("function") or {}
        original = fn.get("name", "")
        safe = _sanitize_tool_name(original)
        if safe != original:
            t = {**t, "function": {**fn, "name": safe}}
            name_map[safe] = original
        sanitized.append(t)
    return sanitized, name_map


def _restore_tool_call_names(response: ModelResponse, name_map: dict[str, str]) -> None:
    """Mutate response.choices[*].message.tool_calls[*].function.name back to canonical."""
    if not name_map:
        return
    try:
        for choice in getattr(response, "choices", []) or []:
            msg = getattr(choice, "message", None)
            for tc in getattr(msg, "tool_calls", None) or []:
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                current = getattr(fn, "name", None)
                if current in name_map:
                    fn.name = name_map[current]
    except Exception as exc:  # noqa: BLE001
        logger.warning("llm.tool_name_restore_failed", error=str(exc))


def _filter_reasoning_fields(
    messages: list[dict[str, Any]], provider_key: str
) -> list[dict[str, Any]]:
    keep_reasoning = provider_key == "deepseek"
    keep_thinking = provider_key == "anthropic"
    if keep_reasoning and keep_thinking:
        return messages
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") != "assistant" or (
            "reasoning_content" not in m and "thinking_blocks" not in m
        ):
            out.append(m)
            continue
        m_copy = dict(m)
        if not keep_reasoning:
            m_copy.pop("reasoning_content", None)
        if not keep_thinking:
            m_copy.pop("thinking_blocks", None)
        out.append(m_copy)
    return out


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

    # Provider-specific reasoning echo: DeepSeek's thinking mode requires the
    # prior assistant `reasoning_content` to be passed back; Anthropic uses
    # `thinking_blocks`. Other providers don't know these fields and may
    # reject them, so strip them when targeting anything else.
    messages = _filter_reasoning_fields(messages, resolved.provider_key)

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
    name_map: dict[str, str] = {}
    if tools:
        sanitized_tools, name_map = _sanitize_tools(tools)
        kwargs["tools"] = sanitized_tools
        if name_map:
            # If any tool name was sanitized, also rewrite the same names in the
            # message history (assistant tool_calls + tool result messages) so the
            # model sees consistent names this turn. We deep-copy so we don't
            # mutate the caller's state (which keeps the canonical names for
            # downstream nodes / persistence).
            original_to_safe = {o: s for s, o in name_map.items()}
            rewritten: list[dict[str, Any]] = []
            for m in kwargs["messages"]:
                m_copy = dict(m)
                tcs = m_copy.get("tool_calls")
                if tcs:
                    new_tcs = []
                    for tc in tcs:
                        tc_copy = dict(tc)
                        fn = dict(tc_copy.get("function") or {})
                        if fn.get("name") in original_to_safe:
                            fn["name"] = original_to_safe[fn["name"]]
                        tc_copy["function"] = fn
                        new_tcs.append(tc_copy)
                    m_copy["tool_calls"] = new_tcs
                if m_copy.get("role") == "tool" and m_copy.get("name") in original_to_safe:
                    m_copy["name"] = original_to_safe[m_copy["name"]]
                rewritten.append(m_copy)
            kwargs["messages"] = rewritten
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
    response = await acompletion(**kwargs)
    _restore_tool_call_names(response, name_map)
    return response


def estimate_cost(response: ModelResponse) -> float:
    """LiteLLM tracks usage cost on the response object."""
    try:
        return float(response._hidden_params.get("response_cost") or 0.0)  # type: ignore[attr-defined]
    except Exception:
        return 0.0
