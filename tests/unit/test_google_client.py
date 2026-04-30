"""lyra_core.tools.google._client."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from google.oauth2.credentials import Credentials

from lyra_core.tools.credentials import ProviderCredentials
from lyra_core.tools.google import _client as client_mod


def _make(scopes: str = "scope.a scope.b"):
    return ProviderCredentials(
        provider="google",
        access_token="AT",
        refresh_token="RT",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        external_account_id="acct",
        scopes=scopes,
    )


def test_google_creds_returns_oauth_credentials() -> None:
    pc = _make()
    creds = client_mod.google_creds(pc)
    assert isinstance(creds, Credentials)
    assert creds.token == "AT"
    assert creds.refresh_token == "RT"
    assert creds.scopes == ["scope.a", "scope.b"]


def test_google_creds_handles_empty_scopes() -> None:
    creds = client_mod.google_creds(_make(scopes=""))
    assert creds.scopes is None


def test_service_builders_call_build_with_correct_args(monkeypatch) -> None:
    captured = []

    def fake_build(api: str, version: str, credentials, cache_discovery=False):
        captured.append((api, version, cache_discovery))
        return f"svc:{api}:{version}"

    monkeypatch.setattr(client_mod, "build", fake_build)

    pc = _make()
    assert client_mod.drive_service(pc) == "svc:drive:v3"
    assert client_mod.docs_service(pc) == "svc:docs:v1"
    assert client_mod.sheets_service(pc) == "svc:sheets:v4"
    assert client_mod.calendar_service(pc) == "svc:calendar:v3"
    assert client_mod.slides_service(pc) == "svc:slides:v1"

    assert captured == [
        ("drive", "v3", False),
        ("docs", "v1", False),
        ("sheets", "v4", False),
        ("calendar", "v3", False),
        ("slides", "v1", False),
    ]
