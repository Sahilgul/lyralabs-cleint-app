"""Microsoft Teams adapter (Phase 2 - skeleton).

Mirrors the Slack adapter pattern: translate inbound Bot Framework
activities into our channel-agnostic InboundMessage and enqueue the same
`run_agent` Celery task. The agent runtime never imports Bot Framework.

Wire this in by:
  1. `pip install '.[teams]'`
  2. Create a Bot Channels Registration in Azure
  3. Add `TEAMS_APP_ID`, `TEAMS_APP_PASSWORD` to env
  4. In apps/api/main.py uncomment the teams_routes mount
"""

from __future__ import annotations

from typing import Any

from ...common.config import get_settings
from ...common.logging import get_logger
from ..schema import InboundMessage, Surface

log = get_logger(__name__)


def build_teams_app() -> Any:
    """Return a configured BotFrameworkAdapter + handler.

    Implementation deferred to Phase 2; placeholder raises if invoked
    without the optional `teams` extra installed.
    """
    settings = get_settings()
    try:
        from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings
        from botbuilder.schema import Activity
    except ImportError as exc:
        raise RuntimeError(
            "botbuilder-python not installed; run: pip install '.[teams]'"
        ) from exc

    adapter_settings = BotFrameworkAdapterSettings(
        app_id=settings.teams_app_id, app_password=settings.teams_app_password
    )
    adapter = BotFrameworkAdapter(adapter_settings)

    async def on_turn(turn_context):  # noqa: ANN001
        """Handle an inbound Teams activity."""
        from apps.worker.tasks.run_agent import run_agent  # noqa: PLC0415

        activity: Activity = turn_context.activity
        if activity.type != "message" or not activity.text:
            return

        msg = InboundMessage(
            surface=Surface.TEAMS,
            tenant_external_id=str(activity.channel_data.get("tenant", {}).get("id", ""))
            if activity.channel_data
            else "",
            channel_id=str(activity.conversation.id) if activity.conversation else "",
            thread_id=str(activity.conversation.id) if activity.conversation else "",
            user_id=str(activity.from_property.id) if activity.from_property else "",
            user_display_name=activity.from_property.name if activity.from_property else None,
            text=activity.text.strip(),
            raw=activity.serialize() if hasattr(activity, "serialize") else {},
        )
        run_agent.delay(message_json=msg.model_dump_json())

    return adapter, on_turn
