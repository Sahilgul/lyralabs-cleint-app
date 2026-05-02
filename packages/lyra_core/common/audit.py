"""Audit log helpers.

Every tool call and every approval decision is recorded. Args are stored
as a hash (not the raw payload) by default to avoid leaking PII; raw
args are stored only when `store_raw=True` and the tenant has opted in.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import AuditEvent


def _hash_args(args: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(args, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


async def record_event(
    session: AsyncSession,
    *,
    tenant_id: str,
    actor_user_id: str | None = None,
    job_id: str | None,
    event_type: str,
    client_id: str | None = None,
    tool_name: str | None = None,
    args: dict[str, Any] | None = None,
    result_status: str = "ok",
    model_used: str | None = None,
    cost_usd: float = 0.0,
    extra: dict[str, Any] | None = None,
    store_raw_args: bool = False,
) -> AuditEvent:
    args_hash = _hash_args(args) if args is not None else None
    event = AuditEvent(
        tenant_id=tenant_id,
        client_id=client_id,
        actor_user_id=actor_user_id,
        job_id=job_id,
        event_type=event_type,
        tool_name=tool_name,
        args_hash=args_hash,
        raw_args=args if store_raw_args else None,
        result_status=result_status,
        model_used=model_used,
        cost_usd=cost_usd,
        extra=extra or {},
        ts=datetime.now(UTC),
    )
    session.add(event)
    await session.flush()
    return event
