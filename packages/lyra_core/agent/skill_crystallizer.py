"""Skill Crystallizer: mine audit_events for repeated tool-call sequences.

A "skill" is a tool sequence that appears >= MINE_FREQUENCY_THRESHOLD times
across jobs for the same (tenant_id, client_id) in the last 30 days.
Sequences are hashed on (tool_name, arg_schema_shape) — structure only,
not values — so "send SMS to contact X" and "send SMS to contact Y" hash
to the same sequence and count toward the same skill.

Promoted skills surface in the agent's system prompt as shortcuts so the
user can say "run the morning pipeline review" without enumerating every step.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any

from sqlalchemy import text

from ..common.logging import get_logger
from ..db.models import Skill
from ..db.session import async_session

log = get_logger(__name__)

MINE_FREQUENCY_THRESHOLD = 5

# 5-minute in-process cache: (tenant_id, client_id) → (skills_list, expires_at)
_SKILLS_CACHE: dict[tuple, tuple] = {}


def _arg_schema_shape(raw_args: dict | None) -> list[tuple[str, str]]:
    """Sorted [(key, type_name)] — captures argument structure, not values."""
    if not raw_args:
        return []
    return sorted((k, type(v).__name__) for k, v in raw_args.items())


def _sequence_hash(sequence: list[tuple[str, list[tuple[str, str]]]]) -> str:
    serialized = json.dumps(sequence, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


async def mine_and_promote_skills(
    tenant_id: str, client_id: str | None
) -> list[str]:
    """Mine audit_events and promote qualifying sequences to the skills table.

    Returns a list of newly promoted skill slugs (existing skills are updated
    with a fresh usage_count but not re-reported as new).
    """
    promoted: list[str] = []
    client_filter = "AND client_id = :client_id" if client_id is not None else "AND client_id IS NULL"

    async with async_session() as s:
        rows = (
            await s.execute(
                text(f"""
                    SELECT job_id,
                           array_agg(tool_name ORDER BY ts)   AS tool_names,
                           array_agg(raw_args   ORDER BY ts)  AS raw_args_list
                    FROM audit_events
                    WHERE tenant_id = :tenant_id
                      {client_filter}
                      AND event_type = 'tool_call'
                      AND result_status = 'ok'
                      AND ts >= now() - interval '30 days'
                    GROUP BY job_id
                    HAVING count(*) >= 2
                """),
                {"tenant_id": tenant_id, "client_id": client_id},
            )
        ).fetchall()

        seq_counts: dict[str, tuple[int, list]] = {}
        for row in rows:
            tool_names = row[1] or []
            raw_args_list = row[2] or []
            sequence = [
                (name, _arg_schema_shape(args))
                for name, args in zip(tool_names, raw_args_list)
            ]
            h = _sequence_hash(sequence)
            count, _ = seq_counts.get(h, (0, sequence))
            seq_counts[h] = (count + 1, sequence)

        for h, (count, sequence) in seq_counts.items():
            if count < MINE_FREQUENCY_THRESHOLD:
                continue
            slug = f"seq_{h}"
            tool_sequence = [{"tool_name": t, "arg_schema_shape": s} for t, s in sequence]

            existing = (
                await s.execute(
                    text(
                        "SELECT id FROM skills "
                        "WHERE tenant_id = :tid "
                        f"{client_filter.replace('client_id', 'skills.client_id')} "
                        "AND slug = :slug"
                    ),
                    {"tenant_id": tenant_id, "client_id": client_id, "slug": slug},
                )
            ).fetchone()

            if existing:
                await s.execute(
                    text("UPDATE skills SET usage_count = :uc, updated_at = now() WHERE id = :id"),
                    {"uc": count, "id": existing[0]},
                )
            else:
                s.add(
                    Skill(
                        id=str(uuid.uuid4()),
                        tenant_id=tenant_id,
                        client_id=client_id,
                        name=f"Sequence {h[:6]}",
                        slug=slug,
                        tool_sequence=tool_sequence,
                        usage_count=count,
                    )
                )
                promoted.append(slug)
                log.info("skill.promoted", slug=slug, count=count, tenant=tenant_id)

        await s.commit()
    return promoted


async def load_active_skills(
    tenant_id: str, client_id: str | None
) -> list[dict[str, Any]]:
    """Return promoted skills for this (tenant, client). 5-minute in-process cache."""
    key = (tenant_id, client_id)
    cached = _SKILLS_CACHE.get(key)
    if cached and cached[1] > time.monotonic():
        return list(cached[0])

    client_filter = "AND client_id = :client_id" if client_id is not None else "AND client_id IS NULL"
    async with async_session() as s:
        rows = (
            await s.execute(
                text(f"""
                    SELECT slug, name, description, tool_sequence, usage_count
                    FROM skills
                    WHERE tenant_id = :tenant_id {client_filter}
                    ORDER BY usage_count DESC
                    LIMIT 20
                """),
                {"tenant_id": tenant_id, "client_id": client_id},
            )
        ).fetchall()

    skills = [
        {
            "slug": r[0],
            "name": r[1],
            "description": r[2],
            "tool_sequence": r[3],
            "usage_count": r[4],
        }
        for r in rows
    ]
    _SKILLS_CACHE[key] = (skills, time.monotonic() + 300)
    return skills
