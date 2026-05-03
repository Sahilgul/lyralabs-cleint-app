"""Executor node.

Walks the Plan in dependency order and runs each tool. Each step's args
may reference earlier step outputs via simple `{{ step_1.field }}`
template substitution before being passed to the tool.

Records each tool call in the audit log.
"""

from __future__ import annotations

import re
from typing import Any

from ...common.audit import record_event
from ...common.logging import get_logger
from ...db.session import async_session
from ...tools.credentials import get_credentials
from ...tools.registry import default_registry
from ..state import AgentState, Plan, StepResult

log = get_logger(__name__)

_TEMPLATE_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_.]+)\s*\}\}")


def _resolve_args(args: dict, prior: dict[str, dict]) -> dict:
    """Replace {{ step_X.path.to.value }} placeholders with values from prior results."""

    def _resolve_value(v: Any) -> Any:
        if isinstance(v, str):

            def repl(m: re.Match[str]) -> str:
                path = m.group(1).split(".")
                step = path[0]
                if step not in prior:
                    return m.group(0)
                cur: Any = prior[step]
                for p in path[1:]:
                    if isinstance(cur, dict):
                        cur = cur.get(p)
                    elif isinstance(cur, list) and p.isdigit():
                        cur = cur[int(p)] if int(p) < len(cur) else None
                    else:
                        cur = None
                    if cur is None:
                        break
                return str(cur) if not isinstance(cur, str) else cur

            return _TEMPLATE_RE.sub(repl, v)
        if isinstance(v, dict):
            return {k: _resolve_value(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_resolve_value(x) for x in v]
        return v

    return {k: _resolve_value(v) for k, v in args.items()}


async def executor_node(state: AgentState) -> dict[str, Any]:
    plan = Plan.model_validate(state["plan"])
    tenant_id = state["tenant_id"]
    job_id = state["job_id"]

    async def creds_lookup(provider: str):
        return await get_credentials(tenant_id, provider)

    from ...tools.base import ToolContext

    ctx = ToolContext(
        tenant_id=tenant_id,
        job_id=job_id,
        user_id=state.get("user_id"),
        creds_lookup=creds_lookup,
    )

    prior: dict[str, dict] = {}
    results: list[dict[str, Any]] = []

    for step in plan.steps:
        # Wait for any depends_on (linear order makes this trivial).
        try:
            tool = default_registry.get(step.tool_name)
        except KeyError:
            res = StepResult(
                step_id=step.id, tool_name=step.tool_name, ok=False, error="unknown tool"
            )
            results.append(res.model_dump())
            continue

        try:
            resolved_args = _resolve_args(step.args, prior)
            args_obj = tool.Input(**resolved_args)
        except Exception as exc:
            res = StepResult(
                step_id=step.id,
                tool_name=step.tool_name,
                ok=False,
                error=f"arg validation: {exc}",
            )
            results.append(res.model_dump())
            continue

        try:
            tr = await tool.safe_run(ctx, args_obj)
        except Exception as exc:
            log.exception("executor.tool_crash", step=step.id, tool=step.tool_name)
            results.append(
                StepResult(
                    step_id=step.id,
                    tool_name=step.tool_name,
                    ok=False,
                    error=f"tool crash: {exc}",
                ).model_dump()
            )
            continue

        if tr.ok and tr.data is not None:
            prior[step.id] = tr.data.model_dump()

        res = StepResult(
            step_id=step.id,
            tool_name=step.tool_name,
            ok=tr.ok,
            data=tr.data.model_dump() if tr.data else None,
            error=tr.error,
            cost_usd=tr.cost_usd,
        )
        results.append(res.model_dump())

        async with async_session() as s:
            await record_event(
                s,
                tenant_id=tenant_id,
                actor_user_id=state.get("user_id"),
                job_id=job_id,
                event_type="tool_call",
                tool_name=step.tool_name,
                args=resolved_args,
                result_status="ok" if tr.ok else "error",
                cost_usd=tr.cost_usd,
                extra={"step_id": step.id, "error": tr.error} if tr.error else {"step_id": step.id},
            )
            await s.commit()

        if not tr.ok:
            log.warning("executor.step_failed", step=step.id, error=tr.error)
            # For MVP: stop on first failure. Later: ask critic to retry/repair.
            break

    # Lift any artifacts the tools produced (PDFs, charts) onto state.
    new_artifacts = list(state.get("artifacts", []))
    new_artifacts.extend(ctx.extra.get("artifacts", []))

    return {"step_results": results, "artifacts": new_artifacts}
