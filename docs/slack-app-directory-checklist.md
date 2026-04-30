# Slack App Directory submission checklist

Required before anyone outside your dev workspace can install. Plan 2–3 review cycles (1–2 weeks each).

## Required surface

- [ ] Bot user with all scopes in `SLACK_SCOPES` (no extras — Slack reviewers reject over-scoping)
- [ ] At least one slash command (`/arlo`)
- [ ] At least one event subscription (`app_mention`, `message.im`)
- [ ] OAuth install URL (`/oauth/slack/install`) — must be publicly reachable HTTPS
- [ ] Redirect URL (`/oauth/slack/callback`)
- [ ] Interactivity request URL (for the Approve/Reject buttons) → `/slack/events`
- [ ] App home tab populated with onboarding text + "Connect Google" / "Connect GHL" buttons that deep-link to your admin panel

## Required listing assets

- [ ] App icon (512×512 PNG, transparent bg)
- [ ] Background color hex
- [ ] Short description (≤140 chars)
- [ ] Long description (≤4000 chars) — explain what the bot does, what data it accesses, why
- [ ] 3–5 screenshots showing the bot in action (1024×768)
- [ ] Demo YouTube video (≤2 minutes) — same one you submit to Google
- [ ] Privacy policy URL
- [ ] Terms of service URL
- [ ] Support email + URL

## Security review questions you'll get

- Where are bot tokens stored? → Postgres, encrypted at rest with per-tenant Fernet keys derived from a master KMS-managed key
- How do you handle uninstall? → On `app_uninstalled` event, mark `slack_installations.bot_token_encrypted = NULL` and revoke any active jobs
- How do you handle data deletion requests? → DELETE `/admin/me/data` endpoint that cascades through tenant_id

## Tracker

- [ ] App created in api.slack.com/apps
- [ ] Manifest exported and committed to `infra/slack-app-manifest.yml`
- [ ] All listing assets uploaded
- [ ] Submitted for review (date: ___)
- [ ] Review feedback round 1 addressed
- [ ] Listed in App Directory (date: ___)
