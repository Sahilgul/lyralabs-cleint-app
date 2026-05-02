"""Living Artifact: per-thread JSONB workspace state distilled after each job.

One row in `workspace_artifacts` per (tenant_id, client_id, thread_id).
The artifact carries durable facts the agent has learned about this
conversation — e.g. "last_campaign_sent", "client_timezone", "pipeline_name".
It's injected into the agent system prompt on every turn so ARLO never
asks the user to repeat stable facts.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from ..db.models import WorkspaceArtifact
from ..db.session import async_session


async def load_artifact(
    tenant_id: str, client_id: str | None, thread_id: str
) -> dict[str, Any]:
    """Load the artifact body. Returns {} when not yet created."""
    async with async_session() as s:
        row = (
            await s.execute(
                select(WorkspaceArtifact).where(
                    WorkspaceArtifact.tenant_id == tenant_id,
                    WorkspaceArtifact.client_id == client_id,
                    WorkspaceArtifact.thread_id == thread_id,
                )
            )
        ).scalar_one_or_none()
    return row.body if row else {}


async def upsert_artifact(
    tenant_id: str,
    client_id: str | None,
    thread_id: str,
    body: dict[str, Any],
) -> None:
    """Insert or update the artifact. Merges body on top of any existing row."""
    async with async_session() as s:
        row = (
            await s.execute(
                select(WorkspaceArtifact).where(
                    WorkspaceArtifact.tenant_id == tenant_id,
                    WorkspaceArtifact.client_id == client_id,
                    WorkspaceArtifact.thread_id == thread_id,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            row = WorkspaceArtifact(
                id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                client_id=client_id,
                thread_id=thread_id,
                body=body,
            )
            s.add(row)
        else:
            row.body = body
            row.updated_at = datetime.now(UTC)
        await s.commit()


def format_artifact_for_prompt(body: dict[str, Any]) -> str:
    """Format the artifact for injection into the agent system prompt."""
    if not body:
        return "(no prior context for this conversation)"
    return "\n".join(f"  - {k}: {v}" for k, v in body.items())
