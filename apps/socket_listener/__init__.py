"""Socket Mode listener.

A long-running process that maintains a persistent WebSocket connection
to Slack via the App-Level Token (`xapp-...`) and dispatches inbound
events into the same Celery `run_agent` task that the HTTPS webhook
uses. Replaces the Cloud Run /slack/events path as the primary inbound
transport when SLACK_APP_TOKEN is configured.
"""
