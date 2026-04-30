"""4-tier memory.

  1. Working memory   - LangGraph state for the current turn (volatile)
  2. Session memory   - LangGraph checkpointer keyed by thread_id (Postgres)
  3. Workspace memory - durable per-tenant Postgres + Qdrant facts
  4. Semantic memory  - Qdrant vector index for RAG over tenant docs

For MVP we ship 1+2 fully (already wired) and stub 3+4. Workspace facts
land in `tenants.settings` JSONB until we need a dedicated table.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from ..common.config import get_settings
from ..common.logging import get_logger
from ..db.models import Tenant
from ..db.session import async_session

log = get_logger(__name__)


async def get_workspace_facts(tenant_id: str) -> dict[str, Any]:
    """Return durable per-tenant facts (e.g. team slug, default Drive folder)."""
    async with async_session() as s:
        t = (await s.execute(select(Tenant).where(Tenant.id == tenant_id))).scalar_one()
        return (t.settings or {}).get("facts", {})


async def upsert_workspace_fact(tenant_id: str, key: str, value: Any) -> None:
    async with async_session() as s:
        t = (await s.execute(select(Tenant).where(Tenant.id == tenant_id))).scalar_one()
        settings_dict = dict(t.settings or {})
        facts = dict(settings_dict.get("facts", {}))
        facts[key] = value
        settings_dict["facts"] = facts
        t.settings = settings_dict
        await s.commit()


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
