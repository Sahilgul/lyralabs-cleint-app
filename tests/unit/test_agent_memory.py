"""lyra_core.agent.memory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from lyra_core.agent import memory as memory_mod
from lyra_core.agent.memory import (
    collection_for,
    ensure_tenant_collection,
    get_workspace_facts,
    upsert_workspace_fact,
)
from lyra_core.db.models import Tenant


def test_collection_for_replaces_dashes() -> None:
    assert collection_for("abc-123-def") == "tenant_abc_123_def"


def test_collection_for_no_dashes_unchanged() -> None:
    assert collection_for("abcdef") == "tenant_abcdef"


@pytest.mark.asyncio
async def test_get_workspace_facts_returns_facts(monkeypatch) -> None:
    tenant = Tenant(external_team_id="T", channel="slack", name="A")
    tenant.id = "t1"
    tenant.settings = {"facts": {"team_slug": "acme"}}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _stmt):
            r = MagicMock()
            r.scalar_one.return_value = tenant
            return r

    monkeypatch.setattr(memory_mod, "async_session", FakeSession)
    facts = await get_workspace_facts("t1")
    assert facts == {"team_slug": "acme"}


@pytest.mark.asyncio
async def test_get_workspace_facts_returns_empty_when_no_facts(monkeypatch) -> None:
    tenant = Tenant(external_team_id="T", channel="slack", name="A")
    tenant.id = "t1"
    tenant.settings = {}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _stmt):
            r = MagicMock()
            r.scalar_one.return_value = tenant
            return r

    monkeypatch.setattr(memory_mod, "async_session", FakeSession)
    facts = await get_workspace_facts("t1")
    assert facts == {}


@pytest.mark.asyncio
async def test_upsert_workspace_fact_writes_facts(monkeypatch) -> None:
    tenant = Tenant(external_team_id="T", channel="slack", name="A")
    tenant.id = "t1"
    tenant.settings = {"facts": {"a": 1}}

    captured = {}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _stmt):
            r = MagicMock()
            r.scalar_one.return_value = tenant
            return r

        async def commit(self):
            captured["committed"] = True

    monkeypatch.setattr(memory_mod, "async_session", FakeSession)

    await upsert_workspace_fact("t1", "b", 2)
    assert tenant.settings == {"facts": {"a": 1, "b": 2}}
    assert captured["committed"] is True


@pytest.mark.asyncio
async def test_upsert_workspace_fact_initializes_facts(monkeypatch) -> None:
    """Settings with no 'facts' key gets one created."""
    tenant = Tenant(external_team_id="T", channel="slack", name="A")
    tenant.id = "t1"
    tenant.settings = {}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _stmt):
            r = MagicMock()
            r.scalar_one.return_value = tenant
            return r

        async def commit(self):
            return None

    monkeypatch.setattr(memory_mod, "async_session", FakeSession)

    await upsert_workspace_fact("t1", "k", "v")
    assert tenant.settings["facts"] == {"k": "v"}


@pytest.mark.asyncio
async def test_ensure_tenant_collection_creates_when_missing(monkeypatch) -> None:
    """Mock the qdrant_client to verify create_collection is called once when collection is missing."""
    import sys
    import types

    # Build a fake qdrant_client + qdrant_client.models
    fake_root = types.ModuleType("qdrant_client")

    class FakeAsyncQdrantClient:
        def __init__(self, url=None, api_key=None):
            self.url = url

        async def get_collections(self):
            return types.SimpleNamespace(collections=[])

        async def create_collection(self, collection_name, vectors_config):
            FakeAsyncQdrantClient.last_create = (collection_name, vectors_config)
            return None

    fake_root.AsyncQdrantClient = FakeAsyncQdrantClient

    fake_models = types.ModuleType("qdrant_client.models")

    class Distance:
        COSINE = "cosine"

    class VectorParams:
        def __init__(self, size, distance):
            self.size = size
            self.distance = distance

    fake_models.Distance = Distance
    fake_models.VectorParams = VectorParams

    monkeypatch.setitem(sys.modules, "qdrant_client", fake_root)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", fake_models)

    await ensure_tenant_collection("abc-123")
    assert FakeAsyncQdrantClient.last_create[0] == "tenant_abc_123"


@pytest.mark.asyncio
async def test_ensure_tenant_collection_no_op_when_exists(monkeypatch) -> None:
    import sys
    import types

    fake_root = types.ModuleType("qdrant_client")
    created = []

    class FakeAsyncQdrantClient:
        def __init__(self, url=None, api_key=None):
            pass

        async def get_collections(self):
            return types.SimpleNamespace(
                collections=[types.SimpleNamespace(name="tenant_abc")]
            )

        async def create_collection(self, **kw):
            created.append(kw)

    fake_root.AsyncQdrantClient = FakeAsyncQdrantClient
    fake_models = types.ModuleType("qdrant_client.models")
    fake_models.Distance = type("D", (), {"COSINE": "cosine"})
    fake_models.VectorParams = lambda size, distance: None

    monkeypatch.setitem(sys.modules, "qdrant_client", fake_root)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", fake_models)

    await ensure_tenant_collection("abc")
    assert created == []
