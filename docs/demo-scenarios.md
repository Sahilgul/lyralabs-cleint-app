# Demo scenarios — paste these into Slack to verify each phase

## Phase 1 — Foundation (week 2)

In Slack, mention the bot:

> @ARLO hello

Expected: friendly direct reply from the `agent` node within 5 seconds (no tool calls). Job appears in `/admin/jobs` with status=done.

---

## Phase 2 — Google read-only (week 4)

> @ARLO pull the last 10 rows from the "Leads" tab of sheet `1AbCdEfGhIjKlMnOpQrStUv` and tell me how many came from Facebook.

Expected:
- Plan has 1 step: `google.sheets.read`. No approval needed (read-only).
- Reply: a short summary with the count + a sample of the rows.
- Job cost < $0.05.

---

## Phase 3 — GHL multi-step (week 6)

> @ARLO list the opportunities in pipeline `xyz123` that have been stuck in stage for more than 14 days, and draft a follow-up SMS for each — preview before sending.

Expected:
- Plan has 2 steps:
  1. `ghl.pipelines.opportunities` (read-only, with `stuck_for_days=14`)
  2. `ghl.conversations.send_message` (write, requires_approval=true)
- Bot posts a Block Kit preview card with Approve / Reject buttons.
- Click Approve: messages send, summary posted in thread.
- Click Reject: bot says "Got it - rejected".

---

## Phase 4 — Approval + artifacts (week 8)

> @ARLO generate this week's agency client report for ACME from sheet `1AbCdEfGhIjKlMnOpQrStUv` and a bar chart of the per-source spend, then save it as a PDF.

Expected:
- Plan has 3 steps: `google.sheets.read` -> `artifact.chart.bar` -> `artifact.pdf.from_markdown`.
- PDF + chart upload to the Slack thread as files.
- Critic posts a one-paragraph summary.

---

## Phase 5 — Full SaaS path (week 10)

1. Install Slack app from the App Directory test link.
2. Open the admin panel, click "Connect Google", complete OAuth.
3. Click "Connect GoHighLevel", complete OAuth.
4. Run any of the above demos.
5. Click "Subscribe — $50/mo" -> Stripe Checkout (test mode card 4242 4242 4242 4242).
6. Tenant `plan` flips to `team` after webhook lands.
