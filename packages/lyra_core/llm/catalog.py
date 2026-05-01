"""Static catalog of LLM providers and known model IDs.

This is the *only* file you edit when a new provider releases (GLM 5,
Kimi K3, a fresh DeepSeek checkpoint). The DB and admin endpoints are
provider-agnostic -- they just look up entries here to render the UI
dropdowns and to know which LiteLLM identifier to send.

Model IDs follow LiteLLM's `<provider>/<model>` convention so we can
hand them straight to `litellm.acompletion(model=...)` without any
translation. See https://docs.litellm.ai/docs/providers for the full
list and any provider-specific kwargs.

Tier hints:
  - "primary": meant for planning/critic/executor reasoning.
  - "cheap":   meant for cheap helper calls (e.g. cheap-pass critic).
  - "both":    fine for either; admin picks.
  - "embedding": embedding model only (do not assign to chat tiers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TierHint = Literal["primary", "cheap", "both", "embedding"]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    context_window: int
    tier_hint: TierHint
    notes: str = ""


@dataclass(frozen=True)
class ProviderSpec:
    """One LLM vendor (Qwen, DeepSeek, OpenAI, ...).

    `litellm_prefix` is what comes before the slash in `model=` strings
    (e.g. `dashscope/qwen-max` => prefix is `dashscope`). It's only used
    for documentation / sanity checks; the canonical identifier is each
    `ModelSpec.id` which already includes the prefix.

    `default_api_base` is the official endpoint. The admin can override
    per-deployment (e.g. China-region DashScope, Azure OpenAI) without a
    code change by setting `api_base` on the LlmProvider DB row.

    `extra_config_keys` lists optional JSON keys the admin can fill in
    via the UI -- for providers like Azure OpenAI that need an
    `api_version` and `deployment_id`, or Bedrock that needs
    `aws_region_name`. Persisted in `LlmProvider.extra_config` JSONB
    and forwarded to `litellm.acompletion` as kwargs.
    """

    key: str
    display_name: str
    litellm_prefix: str
    default_api_base: str | None
    docs_url: str
    known_models: list[ModelSpec]
    requires_api_key: bool = True
    extra_config_keys: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# The catalog
# -----------------------------------------------------------------------------
#
# Entries are populated for providers we have keys for today (qwen, deepseek)
# and pre-registered for the rest so flipping them on later is "drop in an API
# key in the admin UI" -- no code change.

PROVIDERS: dict[str, ProviderSpec] = {
    "qwen": ProviderSpec(
        key="qwen",
        display_name="Alibaba Qwen (DashScope)",
        litellm_prefix="dashscope",
        default_api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        docs_url="https://help.aliyun.com/zh/dashscope/developer-reference/api-details",
        known_models=[
            ModelSpec(
                "dashscope/qwen-max",
                "Qwen Max",
                32768,
                "primary",
                "Flagship reasoning model.",
            ),
            ModelSpec(
                "dashscope/qwen-plus",
                "Qwen Plus",
                131072,
                "both",
                "Balanced cost/quality, very long context.",
            ),
            ModelSpec(
                "dashscope/qwen-turbo",
                "Qwen Turbo",
                8192,
                "cheap",
                "Fastest + cheapest tier.",
            ),
        ],
    ),
    "deepseek": ProviderSpec(
        key="deepseek",
        display_name="DeepSeek",
        litellm_prefix="deepseek",
        default_api_base="https://api.deepseek.com/v1",
        docs_url="https://api-docs.deepseek.com/",
        known_models=[
            ModelSpec(
                "deepseek/deepseek-chat",
                "DeepSeek V3",
                65536,
                "both",
                "General chat / coding. Strong + cheap.",
            ),
            ModelSpec(
                "deepseek/deepseek-reasoner",
                "DeepSeek R1",
                65536,
                "primary",
                "Reasoning model; higher latency.",
            ),
        ],
    ),
    "openai": ProviderSpec(
        key="openai",
        display_name="OpenAI",
        litellm_prefix="openai",
        default_api_base=None,
        docs_url="https://platform.openai.com/docs/models",
        known_models=[
            ModelSpec("openai/gpt-4o", "GPT-4o", 128000, "primary"),
            ModelSpec("openai/gpt-4o-mini", "GPT-4o mini", 128000, "cheap"),
            ModelSpec("openai/gpt-4.1", "GPT-4.1", 1047576, "primary"),
            ModelSpec("openai/gpt-4.1-mini", "GPT-4.1 mini", 1047576, "both"),
            ModelSpec(
                "openai/text-embedding-3-small",
                "text-embedding-3-small",
                8192,
                "embedding",
            ),
            ModelSpec(
                "openai/text-embedding-3-large",
                "text-embedding-3-large",
                8192,
                "embedding",
            ),
        ],
        extra_config_keys=["organization", "project"],
    ),
    "anthropic": ProviderSpec(
        key="anthropic",
        display_name="Anthropic",
        litellm_prefix="anthropic",
        default_api_base=None,
        docs_url="https://docs.anthropic.com/en/docs/about-claude/models",
        known_models=[
            ModelSpec(
                "anthropic/claude-sonnet-4-5",
                "Claude Sonnet 4.5",
                200000,
                "primary",
            ),
            ModelSpec(
                "anthropic/claude-opus-4-1",
                "Claude Opus 4.1",
                200000,
                "primary",
            ),
            ModelSpec(
                "anthropic/claude-haiku-4-5",
                "Claude Haiku 4.5",
                200000,
                "cheap",
            ),
        ],
    ),
    "gemini": ProviderSpec(
        key="gemini",
        display_name="Google Gemini",
        litellm_prefix="gemini",
        default_api_base=None,
        docs_url="https://ai.google.dev/gemini-api/docs/models",
        known_models=[
            ModelSpec("gemini/gemini-2.5-pro", "Gemini 2.5 Pro", 2000000, "primary"),
            ModelSpec("gemini/gemini-2.5-flash", "Gemini 2.5 Flash", 1000000, "both"),
            ModelSpec(
                "gemini/gemini-2.5-flash-lite",
                "Gemini 2.5 Flash Lite",
                1000000,
                "cheap",
            ),
            ModelSpec(
                "gemini/text-embedding-004",
                "Gemini text-embedding-004",
                2048,
                "embedding",
            ),
        ],
    ),
    "moonshot": ProviderSpec(
        key="moonshot",
        display_name="Moonshot AI (Kimi)",
        litellm_prefix="openai",  # OpenAI-compatible endpoint
        default_api_base="https://api.moonshot.ai/v1",
        docs_url="https://platform.moonshot.ai/docs/api-reference",
        known_models=[
            ModelSpec(
                "openai/kimi-k2-0905-preview",
                "Kimi K2 (preview)",
                256000,
                "primary",
                "Latest K2 release. Use OpenAI-compatible endpoint.",
            ),
            ModelSpec(
                "openai/moonshot-v1-128k",
                "Moonshot v1 128k",
                128000,
                "both",
            ),
        ],
        extra_config_keys=[],
    ),
    "minimax": ProviderSpec(
        key="minimax",
        display_name="MiniMax",
        litellm_prefix="openai",  # OpenAI-compatible endpoint
        default_api_base="https://api.minimax.io/v1",
        docs_url="https://www.minimax.io/platform/document/ChatCompletion%20v2",
        known_models=[
            ModelSpec(
                "openai/MiniMax-M2",
                "MiniMax M2",
                204800,
                "primary",
                "Agent-grade flagship.",
            ),
            ModelSpec(
                "openai/MiniMax-Text-01",
                "MiniMax Text-01",
                1000000,
                "both",
                "Long-context text model.",
            ),
        ],
    ),
    "zai": ProviderSpec(
        key="zai",
        display_name="Z.AI (GLM)",
        litellm_prefix="openai",  # OpenAI-compatible endpoint
        default_api_base="https://api.z.ai/api/paas/v4",
        docs_url="https://docs.z.ai/api-reference",
        known_models=[
            ModelSpec(
                "openai/glm-4.6",
                "GLM-4.6",
                200000,
                "primary",
                "Latest GLM flagship.",
            ),
            ModelSpec(
                "openai/glm-4.5",
                "GLM-4.5",
                128000,
                "both",
            ),
            ModelSpec(
                "openai/glm-4-air",
                "GLM-4 Air",
                128000,
                "cheap",
            ),
        ],
    ),
}


def model_spec(model_id: str) -> ModelSpec | None:
    """Look up a ModelSpec by its full LiteLLM id, across all providers."""
    for prov in PROVIDERS.values():
        for m in prov.known_models:
            if m.id == model_id:
                return m
    return None


def provider_for_model(model_id: str) -> ProviderSpec | None:
    """Find the provider that owns this model id."""
    for prov in PROVIDERS.values():
        if any(m.id == model_id for m in prov.known_models):
            return prov
    return None
