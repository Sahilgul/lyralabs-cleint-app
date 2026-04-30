"""lyra_core.common.llm."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from lyra_core.common import llm
from lyra_core.common.config import Settings
from lyra_core.common.llm import ModelTier, _configure_keys, _model_for_tier, chat, estimate_cost


class TestModelForTier:
    def test_primary_returns_settings_primary(self) -> None:
        assert _model_for_tier(ModelTier.PRIMARY) == llm.get_settings().llm_primary_model

    def test_cheap_returns_settings_cheap(self) -> None:
        assert _model_for_tier(ModelTier.CHEAP) == llm.get_settings().llm_cheap_model

    def test_unknown_tier_falls_through_to_cheap(self) -> None:
        # The current implementation: anything not PRIMARY -> CHEAP. Verifies that.
        # Build a fake enum value via cast to confirm the else branch.
        class Fake:
            value = "fake"

        assert _model_for_tier(Fake()) == llm.get_settings().llm_cheap_model  # type: ignore[arg-type]


class TestChat:
    @pytest.mark.asyncio
    async def test_chat_calls_acompletion_with_primary_model(
        self, monkeypatch, mock_litellm_response
    ) -> None:
        mock_acompletion = AsyncMock(return_value=mock_litellm_response("hi", cost=0.001))
        monkeypatch.setattr(llm, "acompletion", mock_acompletion)

        resp = await chat(
            tier=ModelTier.PRIMARY,
            messages=[{"role": "user", "content": "hello"}],
        )

        assert resp.choices[0].message.content == "hi"
        kwargs = mock_acompletion.call_args.kwargs
        assert kwargs["model"] == llm.get_settings().llm_primary_model
        assert kwargs["max_tokens"] == 4096
        assert kwargs["temperature"] == 0.2
        assert kwargs["messages"] == [{"role": "user", "content": "hello"}]

    @pytest.mark.asyncio
    async def test_chat_includes_tools_when_passed(
        self, monkeypatch, mock_litellm_response
    ) -> None:
        mock_acompletion = AsyncMock(return_value=mock_litellm_response("ok"))
        monkeypatch.setattr(llm, "acompletion", mock_acompletion)
        tools = [{"type": "function", "function": {"name": "x"}}]

        await chat(tier=ModelTier.CHEAP, messages=[{"role": "user", "content": "h"}], tools=tools)

        assert mock_acompletion.call_args.kwargs["tools"] == tools

    @pytest.mark.asyncio
    async def test_chat_includes_response_format_when_passed(
        self, monkeypatch, mock_litellm_response
    ) -> None:
        mock_acompletion = AsyncMock(return_value=mock_litellm_response("ok"))
        monkeypatch.setattr(llm, "acompletion", mock_acompletion)

        await chat(
            tier=ModelTier.CHEAP,
            messages=[{"role": "user", "content": "h"}],
            response_format={"type": "json_object"},
        )

        assert mock_acompletion.call_args.kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_chat_passes_metadata_for_cost_tracking(
        self, monkeypatch, mock_litellm_response
    ) -> None:
        mock_acompletion = AsyncMock(return_value=mock_litellm_response("ok"))
        monkeypatch.setattr(llm, "acompletion", mock_acompletion)

        await chat(
            tier=ModelTier.CHEAP,
            messages=[{"role": "user", "content": "h"}],
            metadata={"job_id": "j-1", "tenant_id": "t-1"},
        )

        assert mock_acompletion.call_args.kwargs["metadata"] == {
            "job_id": "j-1",
            "tenant_id": "t-1",
        }

    @pytest.mark.asyncio
    async def test_chat_default_metadata_empty(self, monkeypatch, mock_litellm_response) -> None:
        mock_acompletion = AsyncMock(return_value=mock_litellm_response("ok"))
        monkeypatch.setattr(llm, "acompletion", mock_acompletion)

        await chat(tier=ModelTier.CHEAP, messages=[{"role": "user", "content": "h"}])

        assert mock_acompletion.call_args.kwargs["metadata"] == {}


class TestConfigureKeys:
    """The _configure_keys() side-effect — registers provider keys with LiteLLM
    or mirrors them into the env vars LiteLLM expects (DashScope/Qwen)."""

    def test_qwen_key_mirrors_into_dashscope_env(self, monkeypatch) -> None:
        # Start clean — strip any leakage from previous tests.
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        monkeypatch.delenv("DASHSCOPE_API_BASE", raising=False)

        fake_settings = Settings(
            database_url="postgresql+asyncpg://x:x@x/x",
            database_url_sync="postgresql+psycopg://x:x@x/x",
            master_encryption_key="0123456789abcdef0123456789abcdef0123456789ab=",
            qwen_api_key="sk-dashscope-test-123",
            qwen_api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        monkeypatch.setattr(llm, "get_settings", lambda: fake_settings)

        _configure_keys()

        assert os.environ["DASHSCOPE_API_KEY"] == "sk-dashscope-test-123"
        assert (
            os.environ["DASHSCOPE_API_BASE"]
            == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )

    def test_qwen_key_unset_does_not_pollute_env(self, monkeypatch) -> None:
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        monkeypatch.delenv("DASHSCOPE_API_BASE", raising=False)

        fake_settings = Settings(
            database_url="postgresql+asyncpg://x:x@x/x",
            database_url_sync="postgresql+psycopg://x:x@x/x",
            master_encryption_key="0123456789abcdef0123456789abcdef0123456789ab=",
            qwen_api_key="",  # not configured
        )
        monkeypatch.setattr(llm, "get_settings", lambda: fake_settings)

        _configure_keys()

        assert "DASHSCOPE_API_KEY" not in os.environ

    @pytest.mark.asyncio
    async def test_chat_routes_dashscope_qwen_model_through_acompletion(
        self, monkeypatch, mock_litellm_response
    ) -> None:
        """Sanity check: a `dashscope/qwen-max` tier model is forwarded to
        LiteLLM verbatim — LiteLLM owns provider routing from there."""
        fake_settings = Settings(
            database_url="postgresql+asyncpg://x:x@x/x",
            database_url_sync="postgresql+psycopg://x:x@x/x",
            master_encryption_key="0123456789abcdef0123456789abcdef0123456789ab=",
            llm_primary_model="dashscope/qwen-max",
        )
        monkeypatch.setattr(llm, "get_settings", lambda: fake_settings)
        mock_acompletion = AsyncMock(return_value=mock_litellm_response("salaam"))
        monkeypatch.setattr(llm, "acompletion", mock_acompletion)

        await chat(tier=ModelTier.PRIMARY, messages=[{"role": "user", "content": "hi"}])

        assert mock_acompletion.call_args.kwargs["model"] == "dashscope/qwen-max"


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
        resp = object()  # no _hidden_params
        assert estimate_cost(resp) == 0.0
