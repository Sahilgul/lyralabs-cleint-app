# Phase 2 roadmap (weeks 13+)

After the MVP cohort is paying. Each item is its own focused sprint; do not try in parallel.

## 13-14: Microsoft Teams adapter

Goal: parity with Slack so enterprise prospects on Teams can buy.

- Skeleton already exists at [`packages/lyra_core/channels/teams/`](../packages/lyra_core/channels/teams/).
- Tasks:
  - [ ] Register a Bot in Azure Bot Service
  - [ ] Create Teams app manifest, sideload to a test tenant
  - [ ] Implement an `InstallationStore` parallel to Postgres Slack store
  - [ ] Wire `/teams/messages` endpoint in `apps/api/main.py`
  - [ ] Adaptive Card preview in place of Slack Block Kit
  - [ ] Submit to Teams App Store (~3 weeks review)

## 15: Stripe-as-a-tool

Goal: agent can pull MRR / list subscriptions / draft invoice notes.

- Pattern: register `StripeListSubscriptions`, `StripeMrrSummary`, `StripeCustomerSearch` under `packages/lyra_core/tools/stripe/` using the same `Tool` base.
- Re-use existing `IntegrationConnection` table (`provider='stripe_external'` to distinguish from our own billing Stripe customer).
- All tools are READ-ONLY in v1. Charges/refunds gated behind a separate human-in-loop confirmation flow.

## 16-17: Facebook Business Manager + Meta Ads

Goal: pull campaign metrics, post weekly performance digest.

- OAuth flow: Facebook Login + System User token for long-lived access.
- Tools: `meta.ads.list_campaigns`, `meta.insights.get_metrics`, `meta.pages.list_posts`.
- Caveat: Meta App Review is significantly stricter than Google's. Plan 6-8 weeks lead time.

## 18+: Voice via LiveKit (your wheelhouse)

Goal: phone the bot for hands-free agent interaction (drive-time exec).

- Architecture: LiveKit Agents framework, room per session, STT (Deepgram) -> agent (existing LangGraph) -> TTS (Inworld/OpenAI).
- This unlocks a real differentiator vs Viktor (text-only) for execs who want to talk to the bot from a car.
- Reuses your AI Drive-Thru pattern.

## 19+: Hard integrations (Canva, Capcut, Fanbasis)

Per the original plan: these have no clean OAuth APIs. Strategy:

- Build a separate `tools/browser/` package that wraps Playwright in a sandboxed container.
- Each provider becomes a "skill" rather than a `Tool` — it composes a sequence of browser actions guided by a small LLM.
- Run these in dedicated worker containers (CPU + memory hungry, can crash the regular worker).
