# Beta launch runbook (weeks 11-12)

End-to-end checklist to get the first 5-10 paying workspaces live.

## T-2 weeks

- [ ] All 4 demo scripts work locally (see [demo-scenarios.md](demo-scenarios.md))
- [ ] Production Cloud Run services deployed (`lyralabs-api`, `lyralabs-worker`, `lyralabs-admin`)
- [ ] Custom domain (`api.<yourdomain>`, `app.<yourdomain>`) with TLS
- [ ] Supabase Postgres + Upstash Redis + Qdrant Cloud (or self-hosted) provisioned
- [ ] Stripe account live mode enabled, $50/mo `Team` price created, `STRIPE_PRICE_ID_TEAM_MONTHLY` set
- [ ] Sentry / better-stack / Logtail configured (read JSON logs from Cloud Run)
- [ ] Slack App in "distributed" install mode in api.slack.com (still test-mode for review)

## T-1 week

- [ ] Submit the Slack App Directory listing (see [slack-app-directory-checklist.md](slack-app-directory-checklist.md))
- [ ] Confirm Google OAuth verification status — if still pending, ensure your 100 test-user slots are filled by your design partners' Google emails
- [ ] Privacy policy and ToS published at `<yourdomain>/privacy` and `<yourdomain>/terms`
- [ ] Status page (`status.<yourdomain>`) wired to Cloud Run uptime checks
- [ ] Run the smoke tests against staging: `pytest tests/smoke/`

## Day 0

- [ ] Email design partners with: Slack install link, Google connect link, GHL connect link, 3-5 sample prompts
- [ ] Schedule a 30-min onboarding call with each in their first 24 hours
- [ ] Open a shared Slack Connect channel with each design partner for support

## Daily during beta

- [ ] Tail audit log: `select * from audit_events where ts > now() - interval '24 hours' order by ts desc`
- [ ] Check `jobs.status='failed'` count — investigate every failure
- [ ] Track cost per workspace (`/admin/cost`) — anyone burning >$10/day in trial credit needs intervention
- [ ] Read every job's `user_request` — these are your roadmap

## Iteration cadence

- Daily: prompt tweaks, tool arg-schema fixes (no deploy needed for prompts; ship system prompts via `tenants.settings`)
- Weekly: tool additions (one new tool per week max)
- Bi-weekly: feature flag rollout to one design partner at a time

## Hard cut-offs to enforce

- Plan changes (pricing, scope) only between cohorts of 10 workspaces
- No new integration providers until weeks 13+ regardless of customer asks
- No web chat UI until 50 paying workspaces

## What to track per design partner

| Metric | Target by week 4 |
|---|---|
| Days from install -> first agent task | < 2 |
| Tasks per workspace per week | > 5 |
| Approval-rejection rate | < 20% |
| LLM cost per workspace per week | < $8 |
| Trial -> paid conversion | > 40% |
