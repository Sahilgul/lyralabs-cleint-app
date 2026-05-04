"""lyra_core.common.llm — chat() now resolves model + creds via the router."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import litellm
import pytest
from lyra_core.common import llm
from lyra_core.common.llm import ModelTier, chat, chat_with_fallback, estimate_cost
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
    async def test_chat_forwards_extra_config(self, monkeypatch, mock_litellm_response) -> None:
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


class TestModelTierCritic:
    @pytest.mark.asyncio
    async def test_chat_critic_tier_passes_critic_to_resolve(
        self, monkeypatch, mock_litellm_response
    ) -> None:
        """ModelTier.CRITIC must forward the string "critic" to resolve() so the
        router returns LLM_CRITIC_MODEL (MiniMax M2.7) not the primary model."""
        resolve_mock = AsyncMock(
            return_value=_resolved(
                tier="critic", provider_key="minimax", model_id="openai/MiniMax-M2.7"
            )
        )
        monkeypatch.setattr(llm, "resolve", resolve_mock)
        monkeypatch.setattr(llm, "acompletion", AsyncMock(return_value=mock_litellm_response("ok")))

        await chat(tier=ModelTier.CRITIC, messages=[{"role": "user", "content": "summarise"}])

        resolve_mock.assert_awaited_once_with("critic")

    @pytest.mark.asyncio
    async def test_chat_critic_uses_minimax_endpoint(
        self, monkeypatch, mock_litellm_response
    ) -> None:
        """The resolved MiniMax endpoint (api.minimax.io/v1) must be forwarded
        per-call so Celery workers don't share mutable env-var state."""
        _patch_resolve(
            monkeypatch,
            _resolved(
                tier="critic",
                provider_key="minimax",
                model_id="openai/MiniMax-M2.7",
                api_key="sk-minimax",
                api_base="https://api.minimax.io/v1",
            ),
        )
        mock_acompletion = AsyncMock(return_value=mock_litellm_response("ok"))
        monkeypatch.setattr(llm, "acompletion", mock_acompletion)

        await chat(tier=ModelTier.CRITIC, messages=[{"role": "user", "content": "hi"}])

        kwargs = mock_acompletion.call_args.kwargs
        assert kwargs["model"] == "openai/MiniMax-M2.7"
        assert kwargs["api_key"] == "sk-minimax"
        assert kwargs["api_base"] == "https://api.minimax.io/v1"


class TestKimiTemperatureClamp:
    @pytest.mark.asyncio
    async def test_moonshot_temperature_clamped_to_one(
        self, monkeypatch, mock_litellm_response
    ) -> None:
        """Kimi / Moonshot only accepts temperature=1.0. Any caller value must be
        silently clamped so the fallback chain never hits a 400 BadRequest."""
        _patch_resolve(
            monkeypatch,
            _resolved(provider_key="moonshot", model_id="openai/kimi-k2.6"),
        )
        mock_acompletion = AsyncMock(return_value=mock_litellm_response("ok"))
        monkeypatch.setattr(llm, "acompletion", mock_acompletion)

        await chat(
            tier=ModelTier.PRIMARY,
            messages=[{"role": "user", "content": "h"}],
            temperature=0.2,
        )

        assert mock_acompletion.call_args.kwargs["temperature"] == 1.0

    @pytest.mark.asyncio
    async def test_non_moonshot_temperature_passed_through(
        self, monkeypatch, mock_litellm_response
    ) -> None:
        """Non-Kimi providers must receive the caller's temperature unchanged."""
        _patch_resolve(
            monkeypatch,
            _resolved(provider_key="deepseek", model_id="deepseek/deepseek-v4-pro"),
        )
        mock_acompletion = AsyncMock(return_value=mock_litellm_response("ok"))
        monkeypatch.setattr(llm, "acompletion", mock_acompletion)

        await chat(
            tier=ModelTier.PRIMARY,
            messages=[{"role": "user", "content": "h"}],
            temperature=0.5,
        )

        assert mock_acompletion.call_args.kwargs["temperature"] == 0.5


def _chain(*provider_models: tuple[str, str]) -> list[ResolvedModel]:
    """Build a fake fallback chain from (provider_key, model_id) pairs."""
    return [_resolved(tier="env_pro", provider_key=p, model_id=m) for p, m in provider_models]


_PRO_CHAIN = [
    ("deepseek", "deepseek/deepseek-v4-pro"),
    ("minimax", "openai/MiniMax-M2.7"),
    ("moonshot", "openai/kimi-k2.6"),
]


class TestChatWithFallback:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_provider(self, monkeypatch, mock_litellm_response) -> None:
        monkeypatch.setattr(llm, "build_env_fallback_chain", lambda _: _chain(*_PRO_CHAIN))
        monkeypatch.setattr(llm, "acompletion", AsyncMock(return_value=mock_litellm_response("ok")))

        await chat_with_fallback(quality="pro", messages=[{"role": "user", "content": "h"}])

        assert llm.acompletion.call_count == 1
        assert llm.acompletion.call_args.kwargs["model"] == "deepseek/deepseek-v4-pro"

    @pytest.mark.asyncio
    async def test_falls_back_to_second_on_rate_limit(
        self, monkeypatch, mock_litellm_response
    ) -> None:
        """A 429 from DeepSeek must transparently retry on MiniMax."""
        monkeypatch.setattr(llm, "build_env_fallback_chain", lambda _: _chain(*_PRO_CHAIN))

        call_n = {"n": 0}

        async def _acompletion(**kwargs):
            call_n["n"] += 1
            if call_n["n"] == 1:
                raise litellm.RateLimitError(
                    "rate limited", llm_provider="deepseek", model="deepseek/deepseek-v4-pro"
                )
            return mock_litellm_response("ok from minimax")

        monkeypatch.setattr(llm, "acompletion", _acompletion)

        await chat_with_fallback(quality="pro", messages=[{"role": "user", "content": "h"}])

        assert call_n["n"] == 2

    @pytest.mark.asyncio
    async def test_raises_when_all_providers_fail(self, monkeypatch) -> None:
        monkeypatch.setattr(llm, "build_env_fallback_chain", lambda _: _chain(*_PRO_CHAIN))

        async def _acompletion(**kwargs):
            raise litellm.ServiceUnavailableError("down", llm_provider="x", model="x")

        monkeypatch.setattr(llm, "acompletion", _acompletion)

        with pytest.raises(litellm.ServiceUnavailableError):
            await chat_with_fallback(quality="pro", messages=[{"role": "user", "content": "h"}])

    @pytest.mark.asyncio
    async def test_non_retryable_error_propagates_without_fallback(self, monkeypatch) -> None:
        """A 400 BadRequest means the request itself is broken — trying other
        providers won't fix it. Must raise immediately after the first attempt."""
        monkeypatch.setattr(llm, "build_env_fallback_chain", lambda _: _chain(*_PRO_CHAIN))

        call_n = {"n": 0}

        async def _acompletion(**kwargs):
            call_n["n"] += 1
            raise litellm.BadRequestError("bad prompt", llm_provider="deepseek", model="x")

        monkeypatch.setattr(llm, "acompletion", _acompletion)

        with pytest.raises(litellm.BadRequestError):
            await chat_with_fallback(quality="pro", messages=[{"role": "user", "content": "h"}])

        assert call_n["n"] == 1  # only first provider tried

    @pytest.mark.asyncio
    async def test_empty_chain_raises_runtime_error(self, monkeypatch) -> None:
        monkeypatch.setattr(llm, "build_env_fallback_chain", lambda _: [])

        with pytest.raises(RuntimeError, match="empty"):
            await chat_with_fallback(quality="pro", messages=[{"role": "user", "content": "h"}])

    @pytest.mark.asyncio
    async def test_flash_quality_routes_to_flash_chain(
        self, monkeypatch, mock_litellm_response
    ) -> None:
        """quality='flash' must pass 'flash' to build_env_fallback_chain."""
        received = {"quality": None}

        def _build(q):
            received["quality"] = q
            return _chain(("deepseek", "deepseek/deepseek-v4-flash"))

        monkeypatch.setattr(llm, "build_env_fallback_chain", _build)
        monkeypatch.setattr(llm, "acompletion", AsyncMock(return_value=mock_litellm_response("ok")))

        await chat_with_fallback(quality="flash", messages=[{"role": "user", "content": "h"}])

        assert received["quality"] == "flash"


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
