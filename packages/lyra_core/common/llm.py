"""LLM router using LiteLLM.

Two tiers:
  - PRIMARY (Claude Sonnet 4.5 by default): planner, executor reasoning, critic-on-error.
  - CHEAP (Gemini Flash by default): classifier, critic-pass, simple summarizers.

All call sites must pick a tier explicitly. Tier names are stable;
underlying model can change via env without touching code.

Supported providers (set via LLM_PRIMARY_MODEL / LLM_CHEAP_MODEL):
  - Anthropic:  anthropic/claude-sonnet-4-5, anthropic/claude-haiku-4
  - OpenAI:     openai/gpt-4o, openai/gpt-4o-mini
  - Gemini:     gemini/gemini-2.5-flash, gemini/gemini-2.5-pro
  - Qwen:       dashscope/qwen-max, dashscope/qwen-plus, dashscope/qwen-turbo,
                dashscope/qwen2.5-72b-instruct, dashscope/qwen2.5-coder-32b-instruct
  - DeepSeek:   deepseek/deepseek-chat (V3, general), deepseek/deepseek-reasoner (R1)
"""

from __future__ import annotations

import os
from enum import StrEnum
from typing import Any

import litellm
from litellm import acompletion
from litellm.types.utils import ModelResponse

from .config import get_settings
from .logging import get_logger

logger = get_logger(__name__)


class ModelTier(StrEnum):
    PRIMARY = "primary"
    CHEAP = "cheap"


def _model_for_tier(tier: ModelTier) -> str:
    settings = get_settings()
    if tier is ModelTier.PRIMARY:
        return settings.llm_primary_model
    return settings.llm_cheap_model


def _configure_keys() -> None:
    settings = get_settings()
    if settings.anthropic_api_key:
        litellm.anthropic_key = settings.anthropic_api_key
    if settings.openai_api_key:
        litellm.openai_key = settings.openai_api_key
    if settings.google_api_key:
        litellm.gemini_key = settings.google_api_key
    # Qwen (Alibaba DashScope). LiteLLM's dashscope/* provider reads the
    # standard DashScope env vars, so we just mirror our typed setting into
    # the env. This keeps a single source of truth (.env) and lets users
    # override the regional endpoint via QWEN_API_BASE.
    if settings.qwen_api_key:
        os.environ["DASHSCOPE_API_KEY"] = settings.qwen_api_key
        if settings.qwen_api_base:
            os.environ["DASHSCOPE_API_BASE"] = settings.qwen_api_base
    # DeepSeek (V3 / R1). LiteLLM's deepseek/* provider reads DEEPSEEK_API_KEY.
    if settings.deepseek_api_key:
        os.environ["DEEPSEEK_API_KEY"] = settings.deepseek_api_key


_configure_keys()


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

    `metadata` is passed to LiteLLM for cost-tracking / logging hooks.
    """
    model = _model_for_tier(tier)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "metadata": metadata or {},
    }
    if tools:
        kwargs["tools"] = tools
    if response_format:
        kwargs["response_format"] = response_format

    logger.debug("llm.call", model=model, n_messages=len(messages), n_tools=len(tools or []))
    resp = await acompletion(**kwargs)
    return resp


def estimate_cost(response: ModelResponse) -> float:
    """LiteLLM tracks usage cost on the response object."""
    try:
        return float(response._hidden_params.get("response_cost") or 0.0)  # type: ignore[attr-defined]
    except Exception:
        return 0.0
