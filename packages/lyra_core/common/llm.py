"""LLM call surface, multi-provider via LiteLLM.

Three tiers (callers pick one):
  - PRIMARY: agent orchestrator (DeepSeek V4-Pro by default).
  - CRITIC:  final user-facing summary writer (MiniMax M2.7 by default).
             Faster than PRIMARY, perfect voice score in eval — ideal for
             the critic node which writes directly to the user in Slack.
  - CHEAP:   fast helper calls e.g. tool pre-routing filter.

Per-tier model + credentials are resolved at call time by
`lyra_core.llm.router.resolve(tier)`. The router reads the active
`(provider, model_id, api_key, api_base, extra_config)` from Postgres
(populated by the super-admin via the admin UI), with a 30-second
in-process cache and an env-var fallback for fresh deployments.

Three-tier fallback chain
-------------------------
``chat_with_fallback(quality="pro"|"flash")`` tries providers in order:
  Primary (DeepSeek) → Secondary (MiniMax) → Tertiary (Kimi)

Fallback is triggered by transient provider errors: timeout, rate limit,
connection failure, service unavailable, or auth errors. Non-retryable
errors (400 BadRequest, 404 NotFound) propagate immediately — a fallback
won't fix a malformed request.

See `packages/lyra_core/llm/catalog.py` for the provider catalog and
`packages/lyra_core/llm/router.py` for chain construction.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Literal

import litellm
from litellm import acompletion
from litellm.types.utils import ModelResponse

from ..llm.router import Quality, ResolvedModel, build_env_fallback_chain, resolve
from .logging import get_logger

logger = get_logger(__name__)


class ModelTier(StrEnum):
    PRIMARY = "primary"
    CRITIC = "critic"
    CHEAP = "cheap"


# Errors that indicate the *provider* is unavailable — safe to retry on the
# next provider in the fallback chain. Everything else (BadRequest, NotFound,
# content policy violations) is not retryable and propagates immediately.
_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    litellm.Timeout,
    litellm.APIConnectionError,
    litellm.RateLimitError,
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.AuthenticationError,  # bad key on this provider → try next
)

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
    except Exception as exc:
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


def _build_kwargs(
    resolved: ResolvedModel,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    response_format: dict[str, Any] | None,
    max_tokens: int,
    temperature: float,
    metadata: dict[str, Any],
    timeout_s: float | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Assemble the kwargs dict for ``litellm.acompletion`` and return
    ``(kwargs, name_map)`` where ``name_map`` maps sanitized → original
    tool names (empty when no sanitization was needed).
    """
    messages = _filter_reasoning_fields(messages, resolved.provider_key)

    # Kimi / Moonshot only accepts temperature=1. Clamp here so callers don't
    # need to know about this provider quirk, and the fallback chain never
    # hits a 400 when routing to Kimi.
    if resolved.provider_key == "moonshot":
        temperature = 1.0

    kwargs: dict[str, Any] = {
        "model": resolved.model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "metadata": metadata,
    }
    if timeout_s is not None:
        kwargs["timeout"] = timeout_s
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
            # Rewrite the same tool names inside the *message history* so the
            # model sees consistent names this turn. Deep-copied so we don't
            # mutate the caller's state (which keeps canonical names for
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

    return kwargs, name_map


async def _call_resolved(
    resolved: ResolvedModel,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    metadata: dict[str, Any] | None = None,
    timeout_s: float | None = None,
) -> ModelResponse:
    """Call a single resolved model. No fallback logic — raises on any error."""
    kwargs, name_map = _build_kwargs(
        resolved,
        messages,
        tools,
        response_format,
        max_tokens,
        temperature,
        metadata or {},
        timeout_s,
    )
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
    """Single chat completion against the DB-configured tier model.

    Resolves the active (model, api_key, api_base, ...) from the runtime
    router so admin-panel switches take effect on the next call (modulo
    the router's 30s cache TTL).

    `api_key` and `api_base` are passed *per call* rather than via env
    vars -- this is the prefork-safe pattern: no shared mutable state
    between Celery workers, and no surprises if one tier is on
    OpenAI-compat (Z.AI, Moonshot, MiniMax) while another is on a
    direct provider (Anthropic, Gemini).

    Use ``chat_with_fallback()`` when you want automatic provider failover.
    """
    resolved = await resolve(tier.value)
    return await _call_resolved(
        resolved,
        messages=messages,
        tools=tools,
        response_format=response_format,
        max_tokens=max_tokens,
        temperature=temperature,
        metadata=metadata,
    )


async def chat_with_fallback(
    *,
    quality: Quality,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    metadata: dict[str, Any] | None = None,
    timeout_s: float = 60.0,
) -> ModelResponse:
    """Chat with automatic Primary → Secondary → Tertiary provider fallback.

    ``quality="pro"``   tries: DeepSeek V4-Pro → MiniMax M2.7 → Kimi K2.6
    ``quality="flash"`` tries: DeepSeek V4-Flash → MiniMax M2.5 → Kimi K2.5

    Each attempt has a ``timeout_s`` deadline. Fallback is triggered by:
      - Timeout / connection error
      - Rate limit (429)
      - Service unavailable (503) / internal server error (5xx)
      - Auth error (misconfigured key on this provider)

    Non-retryable errors (400 BadRequest, 404 NotFound, content policy
    violations) propagate immediately — a different provider won't fix them.

    Raises the last retryable error if every provider in the chain fails.
    """
    chain = build_env_fallback_chain(quality)
    if not chain:
        raise RuntimeError(
            f"LLM fallback chain for quality={quality!r} is empty. "
            "Check DEEPSEEK_API_KEY / MINIMAX_API_KEY / KIMI_API_KEY in env."
        )

    last_exc: BaseException | None = None
    for idx, resolved in enumerate(chain):
        is_last = idx == len(chain) - 1
        try:
            logger.info(
                "llm.fallback.attempt",
                quality=quality,
                attempt=idx + 1,
                of=len(chain),
                model=resolved.model_id,
                provider=resolved.provider_key,
            )
            response = await _call_resolved(
                resolved,
                messages=messages,
                tools=tools,
                response_format=response_format,
                max_tokens=max_tokens,
                temperature=temperature,
                metadata=metadata,
                timeout_s=timeout_s,
            )
            if idx > 0:
                logger.info(
                    "llm.fallback.succeeded_on_fallback",
                    quality=quality,
                    attempt=idx + 1,
                    model=resolved.model_id,
                    provider=resolved.provider_key,
                )
            return response

        except _RETRYABLE_ERRORS as exc:
            last_exc = exc
            logger.warning(
                "llm.fallback.provider_failed",
                quality=quality,
                attempt=idx + 1,
                of=len(chain),
                model=resolved.model_id,
                provider=resolved.provider_key,
                error=type(exc).__name__,
                detail=str(exc)[:300],
                falling_back=not is_last,
            )
            if is_last:
                raise

        except Exception:
            # Non-retryable — propagate immediately, don't burn other providers.
            raise

    # Unreachable: the loop always either returns or raises in the is_last branch.
    raise RuntimeError("fallback chain exhausted")  # pragma: no cover


def estimate_cost(response: ModelResponse) -> float:
    """LiteLLM tracks usage cost on the response object."""
    try:
        return float(response._hidden_params.get("response_cost") or 0.0)
    except Exception:
        return 0.0
