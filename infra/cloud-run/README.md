# Cloud Run deployment guide

Two services from the same repo + same Dockerfile, different runtime command.

```mermaid
graph LR
    Slack[Slack/Teams webhooks] -->|HTTPS| API["lyralabs-api Cloud Run service"]
    Vercel[lyralabs.vercel.app admin UI] -->|HTTPS CORS| API
    API -->|"run_agent.delay()"| Redis[Upstash Redis db=0]
    Redis -->|"Celery poll"| Worker["lyralabs-worker Cloud Run service"]
    Worker -->|checkpoints| PG[(Supabase Postgres)]
    Worker -->|chat.postMessage| Slack
    Worker -->|LLM calls| Qwen[Qwen / DashScope]
```

| Service           | Currently named  | Inbound traffic | Runs                                                   |
| ----------------- | ---------------- | --------------- | ------------------------------------------------------ |
| `lyralabs-api`    | `lyralabs-app` * | public          | `uvicorn apps.api.main:app` (Dockerfile default CMD)  |
| `lyralabs-worker` | _not deployed_   | internal only   | `celery -A apps.worker.celery_app:celery worker ...`  |

\* The current API service is named `lyralabs-app` — historical artifact from the first deploy. Sprint 2 renames it to `lyralabs-api`.

## Both services share

- The **same root [Dockerfile](../../Dockerfile)** — built by Cloud Build via the GitHub trigger you set up for `lyralabs-app`. The image is the same; only the runtime command differs.
- The **same env vars** — see the [shared env vars](#shared-env-vars) section below. (Eventually moves to Secret Manager in Sprint 2.)
- The **same region** — `us-east1`, picked when `lyralabs-app` was first deployed.

## Step 1 — Add two missing env vars to lyralabs-app

The current `lyralabs-app` service is missing `CELERY_BROKER_URL` and `CELERY_RESULT_BACKEND`, which is why DMs to ARLO crash with `Connection refused: localhost:6379`. Fix this before deploying the worker.

In the GCP Console:

1. Open **Cloud Run → lyralabs-app → Edit & deploy new revision**
2. Scroll to **Variables & Secrets**
3. Add two environment variables:
   - **`CELERY_BROKER_URL`** = `rediss://default:UPSTASH_REDIS_TOKEN_REDACTED@secure-zebra-111186.upstash.io:6379/0`
   - **`CELERY_RESULT_BACKEND`** = `rediss://default:UPSTASH_REDIS_TOKEN_REDACTED@secure-zebra-111186.upstash.io:6379/0`
4. Click **Deploy**

Both point at Upstash db=0 (Upstash free tier supports only one db; Celery uses prefixed keys to avoid collisions with general `REDIS_URL` use).

## Step 2 — Create the lyralabs-worker service

Same flow you used for `lyralabs-app`, with three differences:

1. Different service name (`lyralabs-worker`)
2. Container command override (run Celery instead of uvicorn)
3. Different scaling/resource knobs

In the GCP Console:

### 2.1 Service basics

- **Cloud Run → Create service**
- **Service name:** `lyralabs-worker`
- **Region:** `us-east1` (must match `lyralabs-app`)
- **Deploy from:** **Continuously deploy from a repository (source or function)**
  - Repository: same GitHub repo as `lyralabs-app`
  - Branch: `main` (or whichever branch you push to)
  - Build type: **Dockerfile**
  - Dockerfile path: `Dockerfile` (repo root — same one the API uses)

### 2.2 Container, command, and arguments

This is the key step that makes the same image behave as a worker.

In the **Container(s)** section, expand **Container settings**:

- **Container command** → click **+** and set: `celery`
- **Container arguments** → add these 7 args, **one per line**:
  ```
  -A
  apps.worker.celery_app:celery
  worker
  -Q
  agent
  --concurrency=4
  --loglevel=INFO
  ```

This overrides the Dockerfile's default `CMD exec uvicorn ...`.

### 2.3 Resources & scaling

- **CPU:** 2
- **Memory:** 2 GiB
- **CPU is always allocated** → **YES** (toggle on — **critical**, the worker needs CPU between Celery polls or it freezes)
- **Execution environment:** Second generation (gen2)
- **Request timeout:** 900 seconds (15 minutes — agent run ceiling)
- **Maximum concurrent requests per instance:** 1
- **Min instances:** 1
- **Max instances:** 10

### 2.4 Networking

- **Ingress control:** **Internal only** (worker has no public surface)
- **Authentication:** **Require authentication** (no public invokers)

### 2.5 Variables & Secrets

Add all the env vars listed in [Shared env vars](#shared-env-vars) below. The fastest way: open `lyralabs-app` in another tab, copy each var, paste into `lyralabs-worker`.

The worker doesn't strictly need the Slack signing/client secrets (it doesn't receive webhooks), but copying the full set is safer and easier than curating.

### 2.6 Service account

Use the same service account as `lyralabs-app` for now. Sprint 2 will create a dedicated `lyralabs-worker` SA with narrower IAM scopes.

### 2.7 Deploy

Click **Create** and wait for the first build. The first deploy takes ~3-5 minutes (Cloud Build builds the image from scratch). Subsequent deploys are faster.

## Step 3 — Smoke test

1. Open Cloud Run logs for `lyralabs-worker`. Within 30 seconds of the deploy completing, you should see:

   ```
   celery@lyralabs-worker-... ready.
   ```

2. In Slack, DM `@ARLO`:

   ```
   hello
   ```

3. Watch the logs in two tabs:

   **`lyralabs-app` log:**
   ```
   slack.event.enqueue tenant=T0A9B4DPW2W ...
   ```

   **`lyralabs-worker` log:**
   ```
   Received task apps.worker.tasks.run_agent.run_agent[<uuid>]
   Task apps.worker.tasks.run_agent.run_agent[<uuid>] succeeded
   ```

4. Back in Slack, ARLO should reply within 5-10 seconds.

If the worker logs show `Received task` but no reply appears in Slack, check:
- The worker can decrypt the bot token (`MASTER_ENCRYPTION_KEY` matches what was used at install time)
- The worker can reach the Slack API (no egress restrictions)
- The agent graph isn't crashing — look for `run_agent.crash` in worker logs

## Shared env vars

These must be set on **both** `lyralabs-api` and `lyralabs-worker`. Pull values from your local `.env` (cloud-specific values noted inline).

### App
| Var               | Value                                                                          |
| ----------------- | ------------------------------------------------------------------------------ |
| `APP_ENV`         | `production`                                                                   |
| `LOG_LEVEL`       | `INFO`                                                                         |
| `APP_BASE_URL`    | `https://lyralabs-app-876710803394.us-east1.run.app` (cloud value)             |
| `ADMIN_BASE_URL`  | `https://lyralabs.vercel.app` (cloud value, must match Vercel deploy)          |

### Storage
| Var                     | Value                                                                                           |
| ----------------------- | ----------------------------------------------------------------------------------------------- |
| `DATABASE_URL`          | from `.env` — Supabase pooler asyncpg URL on port 6543                                          |
| `DATABASE_URL_SYNC`     | from `.env` — Supabase psycopg URL on port 5432                                                 |
| `REDIS_URL`             | from `.env` — Upstash `rediss://...@secure-zebra-111186.upstash.io:6379/0`                      |
| `CELERY_BROKER_URL`     | **same as `REDIS_URL`** (Upstash free tier is single-db, db=0 shared via Celery key prefixes)   |
| `CELERY_RESULT_BACKEND` | **same as `REDIS_URL`**                                                                         |

### Crypto
| Var                     | Value                                                                                           |
| ----------------------- | ----------------------------------------------------------------------------------------------- |
| `MASTER_ENCRYPTION_KEY` | from `.env` — used to decrypt tenant Slack bot tokens (must match across services)              |
| `ADMIN_JWT_SECRET`      | from `.env` — signs admin panel JWTs and OAuth state                                            |
| `ADMIN_JWT_ISSUER`      | `lyralabs-admin`                                                                                |

### LLM
| Var                  | Value                                                            |
| -------------------- | ---------------------------------------------------------------- |
| `LLM_PRIMARY_MODEL`  | `dashscope/qwen-max`                                             |
| `LLM_CHEAP_MODEL`    | `dashscope/qwen-turbo`                                           |
| `QWEN_API_KEY`       | from `.env`                                                      |
| `QWEN_API_BASE`      | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`         |

### Slack (api needs all; worker only needs the bot token, which it reads from Postgres — Slack secrets here are belt-and-suspenders)
| Var                          | Value                                                                                              |
| ---------------------------- | -------------------------------------------------------------------------------------------------- |
| `SLACK_CLIENT_ID`            | from `.env`                                                                                        |
| `SLACK_CLIENT_SECRET`        | from `.env`                                                                                        |
| `SLACK_SIGNING_SECRET`       | from `.env`                                                                                        |
| `SLACK_SCOPES`               | `app_mentions:read,channels:history,...` (full CSV from `.env`)                                    |
| `SLACK_INSTALL_REDIRECT_URL` | `https://lyralabs-app-876710803394.us-east1.run.app/oauth/slack/callback` (must match Slack manifest) |

### Tool integrations (fill in when you wire each one)
- `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET`, `GOOGLE_OAUTH_REDIRECT_URI`, `GOOGLE_OAUTH_SCOPES`
- `GHL_CLIENT_ID`, `GHL_CLIENT_SECRET`, `GHL_REDIRECT_URI`, `GHL_SCOPES`
- `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_PRICE_ID_TEAM_MONTHLY`, `STRIPE_TRIAL_CREDIT_USD`

## Operational notes

### Image tag drift

Both services build from the **same Dockerfile** but have **independent Cloud Build triggers**. When you push to `main`:

- `lyralabs-app`'s trigger rebuilds the API
- `lyralabs-worker`'s trigger rebuilds the worker (if you set up its own GitHub trigger in Step 2)

The two builds happen in parallel and produce different image tags. If one trigger fires but the other doesn't, the services will be on different code revisions until the next push. **This is fine for almost all changes** — they share a codebase, and Celery + LangGraph are tolerant of minor version skew. But for breaking schema changes (e.g., changing `InboundMessage`'s shape), redeploy both and watch the logs.

If you'd rather have one trigger that builds once and deploys both services, that's Sprint 2's job (a `cloudbuild.yaml` with two parallel `gcloud run deploy` steps from the same image).

### Celery broker on Upstash free tier

Upstash free tier has only one logical Redis database (`/0`). Celery uses prefixed keys (`celery-task-meta-*`, `_kombu.binding.celery`, etc.) so collisions with general `REDIS_URL` use are unlikely. If you ever see weird behavior:

- Upgrade Upstash to a paid plan with multi-db, then split: `/0` for general, `/1` for broker, `/2` for results
- Or set `result_backend_transport_options={"global_keyprefix": "lyralabs."}` in `apps/worker/celery_app.py`

### Worker timeout

Cloud Run's max request timeout is 60 minutes; we set 15 minutes (`--timeout=900`) as a defensive ceiling. Most agent runs are under 60 seconds. If a run exceeds 15 min:

- Cloud Run terminates the container
- Celery's `task_acks_late=True` redelivers to another worker
- LangGraph resumes from the last checkpoint in Postgres
- The user sees a delayed reply, but no work is lost

If 15 min becomes a real ceiling (rare — implies a runaway agent loop), bump the timeout to 60 min and add a `recursion_limit` on `graph.ainvoke()`.

### Cost

At the configured `min-instances=1` for both services with light traffic:

| Service           | Monthly idle cost (us-east1, min=1) |
| ----------------- | ----------------------------------- |
| `lyralabs-api`    | ~$15-20                             |
| `lyralabs-worker` | ~$25-30 (cpu-always-allocated)      |
| **Total**         | **~$40-50/mo idle**                 |

Plus traffic costs proportional to actual agent runs. At 100 paying tenants the bill is in the $200-400/mo range — comfortably small for a B2B SaaS.

## What's deferred to Sprint 2

- **Migrate env vars to Secret Manager** (single source of truth, audit trail, easier rotation)
- **Wire `cloudbuild.yaml`** so one push deploys both services from one build
- **Rename `lyralabs-app` → `lyralabs-api`** (cosmetic, but requires updating the Slack manifest URL + Vercel `VITE_API_BASE`)
- **Dedicated `lyralabs-worker` service account** with narrower IAM bindings
- **Migrate to Cloud Run Worker Pools + CREMA** when those leave preview (Q2-Q3 2026 most likely)
