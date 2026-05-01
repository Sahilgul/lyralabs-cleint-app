"""lyra_core.llm.catalog -- the static provider/model registry.

These tests are the regression net for the "easy to add a new provider"
property: a misshapen entry (wrong prefix, model id that doesn't match
the provider, broken tier hint) blows up here, not in production.
"""

from __future__ import annotations

import pytest

from lyra_core.llm.catalog import PROVIDERS, model_spec, provider_for_model

_VALID_TIER_HINTS = {"primary", "cheap", "both", "embedding"}


class TestCatalogShape:
    @pytest.mark.parametrize("provider_key", list(PROVIDERS))
    def test_provider_key_matches_dict_key(self, provider_key: str) -> None:
        assert PROVIDERS[provider_key].key == provider_key

    @pytest.mark.parametrize("provider_key", list(PROVIDERS))
    def test_provider_has_at_least_one_model(self, provider_key: str) -> None:
        assert len(PROVIDERS[provider_key].known_models) >= 1

    @pytest.mark.parametrize("provider_key", list(PROVIDERS))
    def test_model_ids_use_litellm_provider_prefix(self, provider_key: str) -> None:
        spec = PROVIDERS[provider_key]
        for m in spec.known_models:
            assert m.id.startswith(spec.litellm_prefix + "/"), (
                f"{m.id} doesn't start with {spec.litellm_prefix}/ for "
                f"provider {provider_key} -- LiteLLM will route it elsewhere"
            )

    @pytest.mark.parametrize("provider_key", list(PROVIDERS))
    def test_model_tier_hints_are_valid(self, provider_key: str) -> None:
        for m in PROVIDERS[provider_key].known_models:
            assert m.tier_hint in _VALID_TIER_HINTS

    @pytest.mark.parametrize("provider_key", list(PROVIDERS))
    def test_model_context_window_is_positive(self, provider_key: str) -> None:
        for m in PROVIDERS[provider_key].known_models:
            assert m.context_window > 0

    def test_no_duplicate_model_ids_across_providers(self) -> None:
        """A model id must uniquely identify a provider -- otherwise the
        router can't reverse-lookup `provider_for_model()`."""
        seen: dict[str, str] = {}
        for prov_key, spec in PROVIDERS.items():
            for m in spec.known_models:
                if m.id in seen and seen[m.id] != prov_key:
                    raise AssertionError(
                        f"model id {m.id} is registered under both "
                        f"{seen[m.id]} and {prov_key}"
                    )
                seen[m.id] = prov_key


class TestRequiredProviders:
    """Concrete providers we ship today must stay in the catalog."""

    def test_qwen_present(self) -> None:
        assert "qwen" in PROVIDERS
        assert any(
            m.id == "dashscope/qwen-turbo" for m in PROVIDERS["qwen"].known_models
        )

    def test_deepseek_present(self) -> None:
        assert "deepseek" in PROVIDERS
        assert any(
            m.id == "deepseek/deepseek-chat" for m in PROVIDERS["deepseek"].known_models
        )


class TestLookupHelpers:
    def test_model_spec_finds_by_id(self) -> None:
        m = model_spec("dashscope/qwen-max")
        assert m is not None
        assert m.display_name == "Qwen Max"

    def test_model_spec_returns_none_for_unknown(self) -> None:
        assert model_spec("acme/nonexistent") is None

    def test_provider_for_model_resolves(self) -> None:
        prov = provider_for_model("deepseek/deepseek-reasoner")
        assert prov is not None
        assert prov.key == "deepseek"

    def test_provider_for_model_unknown_returns_none(self) -> None:
        assert provider_for_model("acme/nope") is None
