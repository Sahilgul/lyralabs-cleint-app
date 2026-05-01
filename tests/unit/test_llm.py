"""lyra_core.common.llm — chat() now resolves model + creds via the router."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from lyra_core.common import llm
from lyra_core.common.llm import ModelTier, chat, estimate_cost
from lyra_core.llm.router import ResolvedModel


def _resolved(
    *,
    tier: str = "cheap",
    provider_key: str = "qwen",
    model_id: str = "dashscope/qwen-turbo",
    api_key: str | None = "sk-test",
    api_base: str | None = "https://example.com/v1",
    extra_kwargs: dict | None = None,
    source: str = "db",
) -> ResolvedModel:
    return ResolvedModel(
        tier=tier,
        provider_key=provider_key,
        model_id=model_id,
        api_key=api_key,
        api_base=api_base,
        extra_kwargs=extra_kwargs or {},
        source=source,  # type: ignore[arg-type]
    )


def _patch_resolve(monkeypatch, resolved: ResolvedModel) -> AsyncMock:
    mock = AsyncMock(return_value=resolved)
    monkeypatch.setattr(llm, "resolve", mock)
    return mock


class TestChat:
    @pytest.mark.asyncio
    async def test_chat_uses_router_model(self, monkeypatch, mock_litellm_response) -> None:
        _patch_resolve(monkeypatch, _resolved(model_id="dashscope/qwen-max"))
        mock_acompletion = AsyncMock(return_value=mock_litellm_response("hi"))
        monkeypatch.setattr(llm, "acompletion", mock_acompletion)

        await chat(
            tier=ModelTier.PRIMARY,
            messages=[{"role": "user", "content": "hello"}],
        )

        kwargs = mock_acompletion.call_args.kwargs
        assert kwargs["model"] == "dashscope/qwen-max"
        assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
        assert kwargs["max_tokens"] == 4096
        assert kwargs["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_chat_passes_api_key_and_base_per_call(
        self, monkeypatch, mock_litellm_response
    ) -> None:
        """Per-call api_key/api_base is the prefork-safe pattern -- no env mutation."""
        _patch_resolve(
            monkeypatch,
            _resolved(
                api_key="sk-real",
                api_base="https://api.deepseek.com/v1",
                provider_key="deepseek",
                model_id="deepseek/deepseek-chat",
            ),
        )
        mock_acompletion = AsyncMock(return_value=mock_litellm_response("ok"))
        monkeypatch.setattr(llm, "acompletion", mock_acompletion)

        await chat(tier=ModelTier.PRIMARY, messages=[{"role": "user", "content": "h"}])

        kwargs = mock_acompletion.call_args.kwargs
        assert kwargs["model"] == "deepseek/deepseek-chat"
        assert kwargs["api_key"] == "sk-real"
        assert kwargs["api_base"] == "https://api.deepseek.com/v1"

    @pytest.mark.asyncio
    async def test_chat_omits_api_key_when_router_returns_none(
        self, monkeypatch, mock_litellm_response
    ) -> None:
        """When the resolved model has no key (e.g. local Ollama), don't send a
        bogus api_key= that LiteLLM might reject."""
        _patch_resolve(monkeypatch, _resolved(api_key=None, api_base=None))
        mock_acompletion = AsyncMock(return_value=mock_litellm_response("ok"))
        monkeypatch.setattr(llm, "acompletion", mock_acompletion)

        await chat(tier=ModelTier.CHEAP, messages=[{"role": "user", "content": "h"}])

        kwargs = mock_acompletion.call_args.kwargs
        assert "api_key" not in kwargs
        assert "api_base" not in kwargs

    @pytest.mark.asyncio
    async def test_chat_forwards_extra_config(
        self, monkeypatch, mock_litellm_response
    ) -> None:
        """Provider-specific config (e.g. OpenAI org, Azure deployment_id) flows
        through to LiteLLM as kwargs."""
        _patch_resolve(
            monkeypatch,
            _resolved(extra_kwargs={"organization": "org-123", "project": "proj-x"}),
        )
        mock_acompletion = AsyncMock(return_value=mock_litellm_response("ok"))
        monkeypatch.setattr(llm, "acompletion", mock_acompletion)

        await chat(tier=ModelTier.CHEAP, messages=[{"role": "user", "content": "h"}])

        kwargs = mock_acompletion.call_args.kwargs
        assert kwargs["organization"] == "org-123"
        assert kwargs["project"] == "proj-x"

    @pytest.mark.asyncio
    async def test_chat_extra_config_cannot_override_canonical_args(
        self, monkeypatch, mock_litellm_response
    ) -> None:
        """A misconfigured extra_config must NOT be able to swap out the model
        or messages -- those are caller-owned."""
        _patch_resolve(
            monkeypatch,
            _resolved(
                model_id="dashscope/qwen-turbo",
                extra_kwargs={"model": "evil/model", "messages": "evil"},
            ),
        )
        mock_acompletion = AsyncMock(return_value=mock_litellm_response("ok"))
        monkeypatch.setattr(llm, "acompletion", mock_acompletion)

        await chat(tier=ModelTier.CHEAP, messages=[{"role": "user", "content": "real"}])

        kwargs = mock_acompletion.call_args.kwargs
        assert kwargs["model"] == "dashscope/qwen-turbo"
        assert kwargs["messages"] == [{"role": "user", "content": "real"}]

    @pytest.mark.asyncio
    async def test_chat_passes_tools_and_response_format_and_metadata(
        self, monkeypatch, mock_litellm_response
    ) -> None:
        _patch_resolve(monkeypatch, _resolved())
        mock_acompletion = AsyncMock(return_value=mock_litellm_response("ok"))
        monkeypatch.setattr(llm, "acompletion", mock_acompletion)
        tools = [{"type": "function", "function": {"name": "x"}}]

        await chat(
            tier=ModelTier.CHEAP,
            messages=[{"role": "user", "content": "h"}],
            tools=tools,
            response_format={"type": "json_object"},
            metadata={"job_id": "j-1"},
        )

        kwargs = mock_acompletion.call_args.kwargs
        assert kwargs["tools"] == tools
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["metadata"] == {"job_id": "j-1"}


class TestEstimateCost:
    def test_returns_zero_when_no_hidden_params(self) -> None:
        resp = MagicMock()
        resp._hidden_params = {}
        assert estimate_cost(resp) == 0.0

    def test_returns_response_cost_value(self) -> None:
        resp = MagicMock()
        resp._hidden_params = {"response_cost": 0.0042}
        assert estimate_cost(resp) == pytest.approx(0.0042)

    def test_handles_missing_attribute_gracefully(self) -> None:
        resp = object()
        assert estimate_cost(resp) == 0.0
