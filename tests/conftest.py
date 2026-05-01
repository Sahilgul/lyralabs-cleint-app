"""Pytest config + shared fixtures.

ENV is set BEFORE any first import of `lyra_core`, otherwise pydantic-settings
will raise validation errors. Tests then import freely.

Heavy infra (postgres, redis, slack, livestreaming LLM calls) is mocked in unit
tests. For tests that need a real Postgres, place them under tests/integration/.
"""

from __future__ import annotations

import base64
import os
import sys
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

# --- env shim (must run before pydantic-settings binds) -----------------------

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault(
    "MASTER_ENCRYPTION_KEY", base64.urlsafe_b64encode(b"a" * 32).decode("ascii")
)
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"
)
os.environ.setdefault(
    "DATABASE_URL_SYNC", "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
)
os.environ.setdefault(
    "ADMIN_JWT_SECRET",
    "test-admin-jwt-secret-with-enough-length-to-satisfy-rfc7518-32bytes",
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "packages"))
sys.path.insert(0, ROOT)

# --- now safe to import test deps and source -----------------------------------
import pytest  # noqa: E402

# Eagerly import tool packages so default_registry has all of them. Tools
# self-register at import time. Without this, tests that rely on the registry
# would only see whatever was imported transitively.
import lyra_core.tools.artifacts  # noqa: E402, F401
import lyra_core.tools.ghl  # noqa: E402, F401
import lyra_core.tools.google  # noqa: E402, F401


# --- shared fixtures -----------------------------------------------------------


@pytest.fixture
def freeze_now() -> datetime:
    """Frozen reference time used across tests that need stable clocks."""
    return datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def fake_creds():
    """A `ProviderCredentials` test double for any tool needing OAuth tokens."""
    from lyra_core.tools.credentials import ProviderCredentials

    return ProviderCredentials(
        provider="google",
        access_token="ya29.test-token",
        refresh_token="1//rt-test",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        external_account_id="test@example.com",
        scopes="https://www.googleapis.com/auth/drive",
        metadata={"location_id": "loc-test-1"},
    )


@pytest.fixture
def ghl_creds():
    """A `ProviderCredentials` for GHL with a real-shaped location_id."""
    from lyra_core.tools.credentials import ProviderCredentials

    return ProviderCredentials(
        provider="ghl",
        access_token="ghl-access-token",
        refresh_token="ghl-refresh-token",
        expires_at=datetime.now(UTC) + timedelta(hours=23),
        external_account_id="loc-acme-1",
        scopes="contacts.readonly contacts.write",
        metadata={"location_id": "loc-acme-1"},
    )


@pytest.fixture
def make_ctx(fake_creds) -> Callable[..., Any]:
    """Build a ToolContext that returns the supplied creds for every provider."""
    from lyra_core.tools.base import ToolContext

    def _factory(creds=None, dry_run: bool = False, **extra: Any):
        creds = creds or fake_creds

        async def lookup(provider: str):
            return creds

        return ToolContext(
            tenant_id="tenant-test-1",
            job_id="job-test-1",
            user_id="user-test-1",
            dry_run=dry_run,
            creds_lookup=lookup,
            extra=extra,
        )

    return _factory


@pytest.fixture
def mock_session() -> AsyncMock:
    """An AsyncMock pretending to be an `AsyncSession`."""
    s = AsyncMock()
    s.add = MagicMock()  # add() is sync on AsyncSession
    s.flush = AsyncMock()
    s.commit = AsyncMock()
    s.rollback = AsyncMock()
    s.refresh = AsyncMock()
    s.delete = AsyncMock()
    s.execute = AsyncMock()
    return s


@pytest.fixture
def mock_session_cm(mock_session: AsyncMock):
    """An `async_session()` factory whose context manager yields `mock_session`."""

    class _CM:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *exc):
            return False

    def _factory(*_args, **_kwargs):
        return _CM()

    return _factory


@pytest.fixture
def mock_litellm_response():
    """Build a fake `litellm.acompletion` response object."""

    def _factory(content: str, cost: float = 0.0):
        choice = MagicMock()
        choice.message.content = content
        resp = MagicMock()
        resp.choices = [choice]
        resp._hidden_params = {"response_cost": cost}
        return resp

    return _factory


@pytest.fixture
def patch_chat(monkeypatch, mock_litellm_response):
    """Patch `common.llm.chat` with a controllable AsyncMock; returns the mock."""
    from lyra_core.common import llm

    mock = AsyncMock()
    monkeypatch.setattr(llm, "chat", mock)
    # Also patch wherever the import has been already bound in node modules.
    for mod_name in [
        "lyra_core.agent.nodes.critic",
        "lyra_core.agent.nodes.agent",
    ]:
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "chat"):
            monkeypatch.setattr(mod, "chat", mock)

    mock.factory = mock_litellm_response  # type: ignore[attr-defined]
    return mock
