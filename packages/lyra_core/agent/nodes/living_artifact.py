"""Living Artifact distillation node — runs after critic, before END.

Calls a cheap LLM to distill new durable facts from the just-completed job
into the per-thread artifact. Errors are swallowed and logged so a
distillation failure never blocks the user from getting their reply.
"""

from __future__ import annotations

import json
from typing import Any

from ...common.audit import record_event
from ...common.llm import ModelTier, chat
from ...common.logging import get_logger
from ...db.session import async_session
from ..living_artifact import format_artifact_for_prompt, upsert_artifact
from ..state import AgentState

log = get_logger(__name__)

_DISTILL_SYSTEM = "You distill workspace facts from an agent job into a JSON object. Be concise."

_DISTILL_USER = """\
Current artifact (existing durable facts):
{artifact}

Just-completed job:
  Goal: {goal}
  Steps completed: {results_summary}
  Final summary: {final_summary}

Update the artifact with any NEW durable facts learned in this job.
Merge with existing facts; only drop a fact if clearly superseded.
Return ONLY a flat JSON object, e.g.:
{{"last_campaign_sent": "2026-05-02 email to 200 contacts", "primary_pipeline": "Sales"}}
"""


async def living_artifact_node(state: AgentState) -> dict[str, Any]:
    """Distill job results into the living artifact. Never raises."""
    tenant_id = state["tenant_id"]
    client_id = state.get("client_id")
    thread_id = state.get("thread_id", "")
    current_body = state.get("living_artifact") or {}

    try:
        plan_dict = state.get("plan") or {}
        goal = plan_dict.get("goal", state.get("user_request", ""))
        results_summary = json.dumps(
            [
                {"step": r.get("step_id"), "ok": r.get("ok")}
                for r in (state.get("step_results") or [])
            ],
            default=str,
        )[:500]

        messages = [
            {"role": "system", "content": _DISTILL_SYSTEM},
            {
                "role": "user",
                "content": _DISTILL_USER.format(
                    artifact=(
                        json.dumps(current_body, indent=2) if current_body else "(empty)"
                    ),
                    goal=goal,
                    results_summary=results_summary,
                    final_summary=state.get("final_summary") or "",
                ),
            },
        ]
        resp = await chat(tier=ModelTier.CHEAP, messages=messages, max_tokens=512, temperature=0.1)
        raw = (resp.choices[0].message.content or "{}").strip()
        # Strip markdown fences if the model wraps in ```json ... ```.
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip() if len(parts) > 1 else "{}"
        new_facts: dict[str, Any] = json.loads(raw)
        merged = {**current_body, **new_facts}
        await upsert_artifact(tenant_id, client_id, thread_id, merged)
        log.info(
            "living_artifact.updated",
            new_facts=len(new_facts),
            total_facts=len(merged),
            thread=thread_id,
        )
        return {"living_artifact": merged}
    except Exception as exc:
        log.warning("living_artifact.update_failed", error=str(exc))
        try:
            async with async_session() as s:
                await record_event(
                    s,
                    tenant_id=tenant_id,
                    client_id=client_id,
                    job_id=state.get("job_id"),
                    event_type="living_artifact_update_failed",
                    extra={"error": str(exc)},
                )
                await s.commit()
        except Exception:
            pass
        return {}
