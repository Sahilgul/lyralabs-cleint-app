"""lyra_core.common.config.Settings."""

from __future__ import annotations

import base64

import pytest
from lyra_core.common.config import Settings, get_settings
from pydantic import ValidationError


def test_get_settings_is_cached() -> None:
    a = get_settings()
    b = get_settings()
    assert a is b


def test_settings_loads_from_env() -> None:
    s = get_settings()
    assert s.app_env == "test"
    assert s.master_encryption_key
    assert s.database_url.startswith("postgresql+asyncpg://")
    assert s.database_url_sync.startswith("postgresql+psycopg://")


def test_is_prod_false_for_test_env() -> None:
    assert get_settings().is_prod is False


def test_master_key_validator_rejects_placeholder() -> None:
    with pytest.raises(ValueError, match="MASTER_ENCRYPTION_KEY"):
        Settings(
            master_encryption_key="replace-me-with-a-fernet-key",
            database_url="x",
            database_url_sync="y",
        )  # type: ignore[call-arg]


def test_master_key_validator_rejects_empty() -> None:
    with pytest.raises(ValueError, match="MASTER_ENCRYPTION_KEY"):
        Settings(
            master_encryption_key="",
            database_url="x",
            database_url_sync="y",
        )  # type: ignore[call-arg]


def test_slack_scopes_list_splits_csv() -> None:
    s = Settings(
        master_encryption_key=base64.urlsafe_b64encode(b"a" * 32).decode(),
        database_url="x",
        database_url_sync="y",
        slack_scopes="a,b , c,",
    )  # type: ignore[call-arg]
    assert s.slack_scopes_list == ["a", "b", "c"]


def test_google_scopes_list_splits_csv_and_strips() -> None:
    s = Settings(
        master_encryption_key=base64.urlsafe_b64encode(b"a" * 32).decode(),
        database_url="x",
        database_url_sync="y",
        google_oauth_scopes=" https://x , https://y, ",
    )  # type: ignore[call-arg]
    assert s.google_scopes_list == ["https://x", "https://y"]


def test_ghl_scopes_list_splits_on_spaces() -> None:
    s = Settings(
        master_encryption_key=base64.urlsafe_b64encode(b"a" * 32).decode(),
        database_url="x",
        database_url_sync="y",
        ghl_scopes="contacts.readonly  contacts.write opportunities.readonly",
    )  # type: ignore[call-arg]
    assert s.ghl_scopes_list == [
        "contacts.readonly",
        "contacts.write",
        "opportunities.readonly",
    ]


def test_app_env_literal_only_accepts_known() -> None:
    with pytest.raises(ValidationError):
        Settings(
            app_env="garbage",  # type: ignore[arg-type]
            master_encryption_key=base64.urlsafe_b64encode(b"a" * 32).decode(),
            database_url="x",
            database_url_sync="y",
        )  # type: ignore[call-arg]


def test_is_prod_true_when_app_env_production() -> None:
    s = Settings(
        app_env="production",
        master_encryption_key=base64.urlsafe_b64encode(b"a" * 32).decode(),
        database_url="x",
        database_url_sync="y",
    )  # type: ignore[call-arg]
    assert s.is_prod is True


def test_default_models_set() -> None:
    """Both tiers ship with a non-empty <provider>/<model> default. Provider-agnostic
    so swapping the active vendor (Anthropic / Qwen / OpenAI / ...) doesn't break."""
    s = get_settings()
    for value in (s.llm_primary_model, s.llm_cheap_model):
        assert value, "model must be non-empty"
        assert "/" in value, f"expected '<provider>/<model>' format, got {value!r}"
