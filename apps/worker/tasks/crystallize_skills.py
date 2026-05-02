"""arq cron task: nightly skill mining (3 AM UTC).

Scans audit_events from the last 30 days across all active (tenant, client)
pairs and promotes repeated tool sequences to the skills table.
"""

from __future__ import annotations

from sqlalchemy import text

from lyra_core.agent.skill_crystallizer import mine_and_promote_skills
from lyra_core.common.logging import get_logger
from lyra_core.db.session import async_session

log = get_logger(__name__)


async def crystallize_skills(ctx: dict) -> dict:
    return await _run()


async def _run() -> dict:
    async with async_session() as s:
        pairs = (
            await s.execute(
                text("""
                    SELECT DISTINCT tenant_id, client_id
                    FROM audit_events
                    WHERE event_type = 'tool_call'
                      AND result_status = 'ok'
                      AND ts >= now() - interval '30 days'
                """)
            )
        ).fetchall()

    total_promoted = 0
    for row in pairs:
        tenant_id, client_id = row
        try:
            promoted = await mine_and_promote_skills(tenant_id, client_id)
            total_promoted += len(promoted)
        except Exception as exc:
            log.warning(
                "crystallize_skills.pair_error",
                tenant=tenant_id,
                client=client_id,
                error=str(exc),
            )

    log.info(
        "crystallize_skills.done",
        pairs_checked=len(pairs),
        total_promoted=total_promoted,
    )
    return {"pairs_checked": len(pairs), "total_promoted": total_promoted}
