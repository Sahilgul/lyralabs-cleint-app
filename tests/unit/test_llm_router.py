"""lyra_core.llm.router -- runtime resolution + cache + env fallback."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from lyra_core.common.crypto import encrypt_platform
from lyra_core.db.models import LlmModelAssignment, LlmProvider
from lyra_core.llm import router as router_mod


@pytest.fixture(autouse=True)
def _reset_cache():
    """Each test starts with a clean cache, otherwise cross-test contamination
    masks the cache-invalidation logic we're trying to verify."""
    router_mod.invalidate_router_cache()
    yield
    router_mod.invalidate_router_cache()


class _FakeSession:
    """An async_session() drop-in that returns canned scalar lists."""

    def __init__(self, providers=None, assignments=None):
        self._providers = providers or []
        self._assignments = assignments or []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def execute(self, stmt):
        text = str(stmt)
        result = MagicMock()
        if "llm_providers" in text:
            scalars = MagicMock()
            scalars.all = MagicMock(return_value=self._providers)
            scalars.scalar_one_or_none = MagicMock(
                return_value=self._providers[0] if self._providers else None
            )
            result.scalars = MagicMock(return_value=scalars)
            result.scalar_one_or_none = MagicMock(
                return_value=self._providers[0] if self._providers else None
            )
        elif "llm_model_assignments" in text:
            scalars = MagicMock()
            scalars.all = MagicMock(return_value=self._assignments)
            result.scalars = MagicMock(return_value=scalars)
        else:
            scalars = MagicMock()
            scalars.all = MagicMock(return_value=[])
            result.scalars = MagicMock(return_value=scalars)
            result.scalar_one_or_none = MagicMock(return_value=None)
        return result

    async def commit(self):
        return None


def _provider(
    key: str = "qwen", api_key: str | None = "sk-test", enabled: bool = True
) -> LlmProvider:
    p = LlmProvider(
        provider_key=key,
        api_key_encrypted=encrypt_platform(api_key) if api_key else None,
        api_base=None,
        extra_config={},
        enabled=enabled,
    )
    return p


def _assignment(
    tier: str = "cheap",
    provider_key: str = "qwen",
    model_id: str = "dashscope/qwen-turbo",
) -> LlmModelAssignment:
    a = LlmModelAssignment(
        tier=tier,
        provider_key=provider_key,
        model_id=model_id,
    )
    a.updated_at = datetime.now(UTC)
    return a


# --- env-var fallback ---------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_falls_back_to_env_when_no_db_assignment(monkeypatch) -> None:
    """Fresh deploy: no llm_model_assignments rows. Router uses env settings
    so the agent stays alive until the operator configures the admin UI."""
    monkeypatch.setattr(
        router_mod,
        "async_session",
        lambda: _FakeSession(providers=[], assignments=[]),
    )

    resolved = await router_mod.resolve("primary")

    assert resolved.source == "env"
    assert resolved.model_id  # whatever LLM_PRIMARY_MODEL is in the test env


@pytest.mark.asyncio
async def test_resolve_uses_db_assignment_when_present(monkeypatch) -> None:
    monkeypatch.setattr(
        router_mod,
        "async_session",
        lambda: _FakeSession(
            providers=[_provider(key="qwen", api_key="real-key")],
            assignments=[_assignment(tier="cheap", model_id="dashscope/qwen-turbo")],
        ),
    )

    resolved = await router_mod.resolve("cheap")

    assert resolved.source == "db"
    assert resolved.model_id == "dashscope/qwen-turbo"
    assert resolved.api_key == "real-key"  # decrypted


@pytest.mark.asyncio
async def test_resolve_skips_assignment_for_disabled_provider(monkeypatch) -> None:
    """A disabled provider must not be returned as the active resolution --
    the operator turned it off, falling back to env is the right thing."""
    monkeypatch.setattr(
        router_mod,
        "async_session",
        lambda: _FakeSession(
            providers=[_provider(key="qwen", enabled=False)],
            assignments=[_assignment(tier="cheap")],
        ),
    )

    resolved = await router_mod.resolve("cheap")

    assert resolved.source == "env"


@pytest.mark.asyncio
async def test_resolve_skips_assignment_with_missing_provider_row(monkeypatch) -> None:
    """Operator deleted the credentials row but left the assignment behind.
    Router treats it the same as 'no assignment' rather than blowing up."""
    monkeypatch.setattr(
        router_mod,
        "async_session",
        lambda: _FakeSession(
            providers=[],  # no provider configured
            assignments=[_assignment(tier="primary", provider_key="anthropic")],
        ),
    )

    resolved = await router_mod.resolve("primary")

    assert resolved.source == "env"


# --- caching ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_caches_within_ttl(monkeypatch) -> None:
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return _FakeSession(
            providers=[_provider(key="qwen")],
            assignments=[_assignment(tier="cheap")],
        )

    monkeypatch.setattr(router_mod, "async_session", factory)

    await router_mod.resolve("cheap")
    await router_mod.resolve("cheap")
    await router_mod.resolve("cheap")

    # One DB load, three resolves -- the cache absorbed the rest.
    # (factory() builds two sessions per load: providers + assignments)
    assert calls["n"] <= 2


@pytest.mark.asyncio
async def test_invalidate_cache_forces_db_reload(monkeypatch) -> None:
    sessions = {"n": 0}

    def factory():
        sessions["n"] += 1
        return _FakeSession(
            providers=[_provider(key="qwen")],
            assignments=[_assignment(tier="cheap")],
        )

    monkeypatch.setattr(router_mod, "async_session", factory)

    await router_mod.resolve("cheap")
    after_first = sessions["n"]

    router_mod.invalidate_router_cache()
    await router_mod.resolve("cheap")

    # Cache invalidation forced another load, so session count went up.
    assert sessions["n"] > after_first


@pytest.mark.asyncio
async def test_resolve_critic_tier_falls_back_to_env(monkeypatch) -> None:
    """resolve('critic') must use env settings (LLM_CRITIC_MODEL) when no DB
    assignment exists — same bootstrap behaviour as primary/cheap."""
    monkeypatch.setattr(
        router_mod,
        "async_session",
        lambda: _FakeSession(providers=[], assignments=[]),
    )

    resolved = await router_mod.resolve("critic")

    assert resolved.source == "env"
    assert resolved.tier == "critic"
    assert resolved.model_id  # whatever LLM_CRITIC_MODEL is in the test env


@pytest.mark.asyncio
async def test_resolve_critic_tier_uses_db_when_present(monkeypatch) -> None:
    """An operator can override the critic model via the admin UI."""
    monkeypatch.setattr(
        router_mod,
        "async_session",
        lambda: _FakeSession(
            providers=[_provider(key="minimax", api_key="sk-minimax")],
            assignments=[
                _assignment(
                    tier="critic",
                    provider_key="minimax",
                    model_id="openai/MiniMax-M2.7",
                )
            ],
        ),
    )

    resolved = await router_mod.resolve("critic")

    assert resolved.source == "db"
    assert resolved.model_id == "openai/MiniMax-M2.7"
    assert resolved.api_key == "sk-minimax"


# --- build_env_fallback_chain -------------------------------------------------


def test_build_env_fallback_chain_pro_has_three_providers(monkeypatch) -> None:
    """Pro chain: DeepSeek → MiniMax → Kimi, in that order."""
    monkeypatch.setattr(router_mod, "_env_api_key_for", lambda spec: "sk-fake")

    chain = router_mod.build_env_fallback_chain("pro")

    assert len(chain) == 3
    provider_keys = [m.provider_key for m in chain]
    assert provider_keys[0] == "deepseek"
    assert provider_keys[1] == "minimax"
    assert provider_keys[2] == "moonshot"


def test_build_env_fallback_chain_flash_has_three_providers(monkeypatch) -> None:
    """Flash chain has the same provider order as pro, different model IDs."""
    monkeypatch.setattr(router_mod, "_env_api_key_for", lambda spec: "sk-fake")

    chain = router_mod.build_env_fallback_chain("flash")

    assert len(chain) == 3
    assert chain[0].provider_key == "deepseek"
    assert chain[1].provider_key == "minimax"
    assert chain[2].provider_key == "moonshot"


def test_build_env_fallback_chain_skips_provider_without_key(monkeypatch) -> None:
    """A provider with no API key is silently dropped so the remaining
    providers are still attempted."""

    def _key_for(spec):
        return None if spec.key == "minimax" else "sk-fake"

    monkeypatch.setattr(router_mod, "_env_api_key_for", _key_for)

    chain = router_mod.build_env_fallback_chain("pro")
    provider_keys = [m.provider_key for m in chain]

    assert "minimax" not in provider_keys
    assert "deepseek" in provider_keys
    assert "moonshot" in provider_keys


def test_build_env_fallback_chain_empty_when_no_keys(monkeypatch) -> None:
    """All keys missing → empty list. chat_with_fallback raises RuntimeError."""
    monkeypatch.setattr(router_mod, "_env_api_key_for", lambda spec: None)

    chain = router_mod.build_env_fallback_chain("pro")

    assert chain == []


def test_build_env_fallback_chain_source_is_env(monkeypatch) -> None:
    """Every entry in the chain must have source='env' (built from env vars,
    not from a DB assignment lookup)."""
    monkeypatch.setattr(router_mod, "_env_api_key_for", lambda spec: "sk-fake")

    chain = router_mod.build_env_fallback_chain("pro")

    assert all(m.source == "env" for m in chain)


# --- db failure ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_survives_db_failure(monkeypatch, caplog) -> None:
    """If Postgres is briefly unreachable, resolve() must fall back to env --
    breaking the agent because the admin DB hiccuped is unacceptable."""

    def boom():
        class _Dead:
            async def __aenter__(self):
                raise RuntimeError("db unreachable")

            async def __aexit__(self, *_):
                return False

        return _Dead()

    monkeypatch.setattr(router_mod, "async_session", boom)

    resolved = await router_mod.resolve("cheap")

    assert resolved.source == "env"
