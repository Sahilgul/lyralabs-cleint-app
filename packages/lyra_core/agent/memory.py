"""4-tier memory.

  1. Working memory   - LangGraph state for the current turn (volatile)
  2. Session memory   - LangGraph checkpointer keyed by thread_id (Postgres)
  3. Workspace memory - durable per-tenant Postgres + Qdrant facts
  4. Semantic memory  - Qdrant vector index for RAG over tenant docs

For MVP we ship 1+2 fully (already wired) and stub 3+4. Workspace facts
land in `tenants.settings` JSONB until we need a dedicated table.
"""

from __future__ import annotations

import time
from typing import Any

from sqlalchemy import select

from ..common.config import get_settings
from ..common.logging import get_logger
from ..db.models import Tenant
from ..db.session import async_session

log = get_logger(__name__)


# Workspace facts only mutate via the admin panel (rare) and are read on
# EVERY agent turn. Fetching from Tokyo Postgres on every turn was adding
# ~200ms per loop iteration. Cache for 30s -- short enough that admin-panel
# edits propagate quickly, long enough to absorb a multi-iteration agent
# run for a single user message.
_FACTS_TTL_SECONDS = 30.0
_facts_cache: dict[str, tuple[float, dict[str, Any]]] = {}


async def get_workspace_facts(tenant_id: str) -> dict[str, Any]:
    """Return durable per-tenant facts (e.g. team slug, default Drive folder)."""
    cached = _facts_cache.get(tenant_id)
    if cached is not None:
        cached_at, facts = cached
        if time.monotonic() - cached_at < _FACTS_TTL_SECONDS:
            return facts

    async with async_session() as s:
        t = (await s.execute(select(Tenant).where(Tenant.id == tenant_id))).scalar_one()
        facts = (t.settings or {}).get("facts", {})

    _facts_cache[tenant_id] = (time.monotonic(), facts)
    return facts


def invalidate_workspace_facts_cache(tenant_id: str | None = None) -> None:
    """Drop the cached facts (call after admin edits a tenant's settings)."""
    if tenant_id is None:
        _facts_cache.clear()
    else:
        _facts_cache.pop(tenant_id, None)


async def upsert_workspace_fact(tenant_id: str, key: str, value: Any) -> None:
    async with async_session() as s:
        t = (await s.execute(select(Tenant).where(Tenant.id == tenant_id))).scalar_one()
        settings_dict = dict(t.settings or {})
        facts = dict(settings_dict.get("facts", {}))
        facts[key] = value
        settings_dict["facts"] = facts
        t.settings = settings_dict
        await s.commit()
    invalidate_workspace_facts_cache(tenant_id)


# --- Semantic / RAG (Qdrant) -------------------------------------------------


def collection_for(tenant_id: str) -> str:
    return f"tenant_{tenant_id.replace('-', '_')}"


async def ensure_tenant_collection(tenant_id: str, vector_size: int = 1536) -> None:
    """Create the per-tenant Qdrant collection if missing."""
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Distance, VectorParams

    settings = get_settings()
    client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    name = collection_for(tenant_id)
    existing = await client.get_collections()
    if name not in {c.name for c in existing.collections}:
        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        log.info("qdrant.collection.created", tenant_id=tenant_id, name=name)
