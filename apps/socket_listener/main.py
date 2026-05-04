"""Socket Mode listener entrypoint.

Why Socket Mode (vs. the HTTPS /slack/events webhook):

  * No 3-second-ack cliff. HTTP delivery retries an event up to 3x if
    the endpoint doesn't 200 within 3s. On Cloud Run cold starts this
    routinely produced retry storms (4 run_agent tasks for one DM).
    Socket Mode has no such retry path -- events are streamed over a
    persistent WebSocket and acked over the same channel.
  * No public ingress. The VM dials OUT to Slack's Events backbone, so
    we don't need to expose port 443 to the internet for inbound events.
    OAuth callbacks still need HTTPS, but those land on Cloud Run via
    the existing /oauth/slack/* routes.
  * No cold-start tax. The connection lives in this always-on VM, so
    inbound events skip the entire "wake the API container" prelude.

Multi-tenant note: the App-Level Token is a property of the SLACK APP,
not of any one workspace. One Socket Mode connection serves every
workspace that installed ARLO; per-event authorization is handled by
the same `PostgresInstallationStore` the HTTPS path uses.

Configuration: the runner exits cleanly with status 0 and a warning
if SLACK_APP_TOKEN isn't set, so an absent token doesn't crash-loop
the container under docker's restart policy. To enable, set
SLACK_APP_TOKEN to an `xapp-...` token with `connections:write` scope.
"""

from __future__ import annotations

import asyncio
import signal
import sys

from lyra_core.channels.slack.adapter import build_socket_mode_app
from lyra_core.common.config import get_settings
from lyra_core.common.logging import configure_logging, get_logger

# Importing the tools modules here too so that any code path the Bolt
# handlers eventually reach (e.g. via run_agent) sees the registered
# tools. Celery's worker process is a separate container and registers
# them on its own; doing it here as well costs nothing and keeps the
# socket listener self-contained.
from lyra_core.tools import artifacts as _artifacts  # noqa: F401
from lyra_core.tools import google as _google  # noqa: F401
from lyra_core.tools import slack as _slack  # noqa: F401
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

log = get_logger(__name__)


async def _run() -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.is_prod)

    if not settings.slack_app_token:
        log.warning(
            "socket_listener.disabled",
            reason="SLACK_APP_TOKEN is empty; staying idle so the container "
            "doesn't crash-loop. Set the env var to start the WebSocket "
            "connection.",
        )
        # Sleep forever instead of exiting -- avoids a docker restart loop
        # while making it obvious to the operator that nothing is listening.
        # Configurable via SIGTERM; docker-compose `down` interrupts cleanly.
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
        await stop.wait()
        return

    if not settings.slack_app_token.startswith("xapp-"):
        log.error(
            "socket_listener.bad_token",
            reason="SLACK_APP_TOKEN must start with 'xapp-'. Bot tokens "
            "(xoxb-) and user tokens (xoxp-) won't work for Socket Mode.",
        )
        sys.exit(1)

    app = build_socket_mode_app()
    handler = AsyncSocketModeHandler(app, settings.slack_app_token)

    log.info(
        "socket_listener.starting",
        # Don't log the token; just enough to confirm config plumbed through.
        token_prefix=settings.slack_app_token[:7] + "...",
        signing_secret_set=bool(settings.slack_signing_secret),
        env=settings.app_env,
    )

    # `start_async()` returns once the socket disconnects (e.g. graceful
    # shutdown). It auto-reconnects internally on transient drops; we only
    # exit when the loop is cancelled.
    try:
        await handler.start_async()
    except asyncio.CancelledError:
        log.info("socket_listener.cancelled")
        raise
    finally:
        log.info("socket_listener.shutdown")


def main() -> None:  # pragma: no cover -- entry point
    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    main()
