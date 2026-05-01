# `lyralabs-worker` on a GCE VM (with self-hosted Redis)

This deploys the Celery worker plus its Redis broker on a single Google
Compute Engine VM. The Cloud Run API service (`lyralabs-app`) connects to
the VM's Redis over the public internet, authenticated with a strong
password.

## Why a VM (not Cloud Run)

Cloud Run **Services** require an HTTP listener on `$PORT`; a Celery
worker is pull-based and never opens a port. The two ways to reconcile
that on Cloud Run are (a) wrap the worker in an HTTP healthcheck shim, or
(b) use Cloud Run **Worker Pools** (in preview, no autoscaler hooks for
Redis queue depth yet). Neither is worth the complexity at MVP scale.

A single `e2-small` VM with `restart: always` containers gives us:

- `~$13/mo` total (vs `~$25-40/mo` for an always-on Cloud Run worker + VPC connector).
- Sub-ms latency between worker and Redis (same docker network).
- Zero vendor lock-in for the broker.
- Familiar ops: `docker compose logs -f`, `htop`, `journalctl`.

Trade-off: when this VM crashes, the API can't enqueue messages until the
VM comes back. Acceptable for MVP. Mitigations are in
[Operations / Reliability](#operations) below.

---

## Cost breakdown

| Component | Monthly |
|-----------|---------|
| `e2-small` VM (2 vCPU burst, 2 GB RAM) | ~$12.50 |
| Static external IP (in-use) | $0.00 |
| Standard PD 10 GB | ~$0.40 |
| Egress (Slack/Qwen/Postgres) | $0-2 |
| **Total** | **~$13/mo** |

Drop to `e2-micro` (~$6/mo) if memory holds at <512 MB. Bump to `e2-medium`
when Celery concurrency goes past 8 or you start seeing OOM kills.

---

## Phase 1 — Clean up the failing Cloud Run worker

Skip if you already deleted it.

```nu
let project = "YOUR_PROJECT_ID"  # gcloud config get-value project
let region = "us-east1"

# Delete the Cloud Run service that was failing the startup probe.
^gcloud run services delete lyralabs-worker --region=$region --project=$project --quiet

# Delete its Cloud Build trigger (find the right name first).
^gcloud builds triggers list --project=$project --filter="name:lyralabs-worker" --format=json
# Then:
^gcloud builds triggers delete TRIGGER_ID --project=$project --quiet
```

---

## Phase 2 — Provision the VM

Run from your laptop. Replace `YOUR_PROJECT_ID`. Region matches the API.

```nu
let project = "YOUR_PROJECT_ID"
let region = "us-east1"
let zone = "us-east1-b"

# 1. Reserve a static external IP. ($0/mo while attached to a running VM.)
^gcloud compute addresses create lyralabs-worker-ip --region=$region --project=$project

let static_ip = (
    ^gcloud compute addresses describe lyralabs-worker-ip
        --region=$region --project=$project --format="value(address)"
    | str trim
)
print $"Static IP: ($static_ip)"

# 2. Create the VM. Container-Optimized OS gives us Docker preinstalled and
#    auto-applies security patches. (If you prefer Debian for ssh-y workflows,
#    swap --image-family=cos-stable for --image-family=debian-12.)
^gcloud compute instances create lyralabs-worker
    --project=$project
    --zone=$zone
    --machine-type=e2-small
    --image-family=cos-stable
    --image-project=cos-cloud
    --boot-disk-size=10GB
    --boot-disk-type=pd-standard
    --address=$static_ip
    --tags=lyralabs-worker
    --scopes=https://www.googleapis.com/auth/cloud-platform
    --metadata=enable-oslogin=TRUE

# 3. Firewall: allow Redis (6379) inbound from Cloud Run's egress range.
#    Cloud Run uses dynamic IPs, so we open 0.0.0.0/0 and rely on the
#    Redis AUTH password for security. Strong password + AOF persistence
#    is the standard self-hosted Redis pattern.
#    (To lock this down, set up a VPC Connector for the API and bind redis
#     to 10.x.x.x — see "Private networking" at the bottom.)
^gcloud compute firewall-rules create lyralabs-worker-redis
    --project=$project
    --direction=INGRESS
    --action=ALLOW
    --rules=tcp:6379
    --target-tags=lyralabs-worker
    --source-ranges=0.0.0.0/0
```

---

## Phase 3 — First-time VM setup

SSH into the VM. Container-Optimized OS already has Docker; we just
configure it.

```nu
^gcloud compute ssh lyralabs-worker --zone=$zone --project=$project
```

Inside the VM (this is a regular bash shell — COS is bash-based):

```bash
# Become root (COS is read-only by default; /home and /var/lib/docker are writable).
sudo su -

# Working directory on a writable partition.
mkdir -p /var/lyralabs
cd /var/lyralabs

# Generate a Redis password. Save the output — you'll need it for Cloud Run.
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
# -> e.g. 9bL7m4kNz3pX8vYqWcA1eR5tH6jKfDgS-something-32-chars
```

Now copy the deployment artifacts onto the VM. Two options:

**Option A — Pull from GitHub** (recommended; once you push the
`infra/vm/` files):

```bash
# Clone shallow (auth via OS Login token; no need to manage SSH keys).
toolbox  # COS comes with `toolbox` for utilities like git
git clone --depth=1 https://github.com/YOUR_GH_USER/lyralabs.git
cp -r lyralabs/infra/vm/* /var/lyralabs/
exit  # leave toolbox
```

**Option B — Quick copy from laptop** (one-shot for tonight):

From your laptop:

```nu
^gcloud compute scp infra/vm/docker-compose.yml infra/vm/deploy.sh infra/vm/.env.example \
    lyralabs-worker:/tmp/ --zone=$zone --project=$project

# Back on the VM:
sudo mv /tmp/docker-compose.yml /tmp/deploy.sh /tmp/.env.example /var/lyralabs/
```

Then on the VM:

```bash
cd /var/lyralabs
cp .env.example .env
nano .env   # fill in everything; copy values from your local .env + the Redis password you generated
chmod 600 .env
chmod +x deploy.sh
```

Critical values to set in `/var/lyralabs/.env`:

- `REDIS_PASSWORD` — the one you just generated.
- `WORKER_IMAGE` — find via:
  ```nu
  ^gcloud run services describe lyralabs-app --region=us-east1
      --format="value(spec.template.spec.containers[0].image)"
  ```
- `DATABASE_URL` / `DATABASE_URL_SYNC` — copy from local `.env`.
- `MASTER_ENCRYPTION_KEY` — **must match** the one in `lyralabs-app` exactly,
  otherwise the worker can't decrypt tenant Slack tokens.
- `QWEN_API_KEY` — copy from local `.env`.
- `SLACK_SIGNING_SECRET`, `ADMIN_JWT_SECRET` — copy from local `.env`.

Authenticate Docker against Artifact Registry and start the stack:

```bash
# COS: gcloud is exposed via /usr/share/google-cloud-cli inside `toolbox`,
# but the docker-credential-gcr helper is built in.
docker-credential-gcr configure-docker --registries=us-east1-docker.pkg.dev

# First boot: pull the image, start redis + worker.
docker compose up -d

# Watch logs.
docker compose logs -f
```

You should see Celery's "ready" banner and the worker registering tasks:

```
worker | [tasks]
worker |   . apps.worker.tasks.run_agent.resume_agent
worker |   . apps.worker.tasks.run_agent.run_agent
worker | [INFO/MainProcess] celery@... ready.
```

---

## Phase 4 — Update Cloud Run API to use the VM's Redis

Console route (fastest):

1. Open **Cloud Run → lyralabs-app → Edit & Deploy New Revision**.
2. Under **Variables & secrets** set:
   - `CELERY_BROKER_URL` = `redis://default:THE_PASSWORD@STATIC_IP:6379/0`
   - `CELERY_RESULT_BACKEND` = `redis://default:THE_PASSWORD@STATIC_IP:6379/0`
   - `REDIS_URL` = same value (some code paths read it directly).

   URL-encode the password if it contains `@`, `:`, `/`, or `?`. The
   `secrets.token_urlsafe` output is already safe.

3. Click **Deploy**.

CLI route:

```nu
let static_ip = "..."  # from Phase 2
let pwd = "..."        # the Redis password
let url = $"redis://default:($pwd)@($static_ip):6379/0"

^gcloud run services update lyralabs-app
    --region=us-east1
    --project=$project
    --update-env-vars=$"CELERY_BROKER_URL=($url),CELERY_RESULT_BACKEND=($url),REDIS_URL=($url)"
```

---

## Phase 5 — Smoke test

1. On the VM, in one terminal: `docker compose logs -f worker`.
2. From Slack: DM ARLO `hello`.
3. Cloud Run API logs should show: `enqueued task to celery (queue=agent)`.
4. Worker logs should show: `Task apps.worker.tasks.run_agent.run_agent[...] received` then `succeeded`.
5. Slack DM should show ARLO replying within 5-10 seconds.

If step 4 doesn't fire within a few seconds, check
[Troubleshooting](#troubleshooting).

---

## Phase 6 — Re-deploys (auto + manual)

Whenever Cloud Build pushes a new image (the API redeploys), you'll
want the worker to pick it up too.

**Manual re-deploy** (works today, run on the VM):

```bash
cd /var/lyralabs

# Update WORKER_IMAGE in .env to the new tag, then:
./deploy.sh
```

**Automatic re-deploy** (next iteration; not needed tonight):

Three reasonable options, pick later:

- **Watchtower** — a 5MB container that auto-pulls and restarts on
  image tag updates. Add `containrrr/watchtower` to `docker-compose.yml`.
  Easiest, but pulls on a poll (1-5 min lag).
- **Cloud Build SSH step** — append a step to your existing trigger
  that SSHes into the VM and runs `./deploy.sh`. Tightest feedback but
  requires SSH key management in Cloud Build.
- **PubSub + listener** — Cloud Build emits to PubSub on success, a tiny
  listener on the VM pulls. Cleanest but most setup.

Recommend Watchtower for now; revisit if poll lag causes issues.

---

## Operations

### Logs

```bash
# Worker only
docker compose logs -f worker

# Redis only
docker compose logs -f redis

# Last 200 lines
docker compose logs --tail=200 worker

# Filter for errors
docker compose logs worker | grep -E "(ERROR|CRITICAL|Traceback)"
```

### Restart

```bash
# Worker only (preserves Redis state)
docker compose restart worker

# Everything
docker compose restart
```

### Scale concurrency

Edit `docker-compose.yml`, change `--concurrency=4` to your target,
then `docker compose up -d worker`. Each Celery process slot uses
~80-150 MB RAM; an `e2-small` (2 GB) handles 8-10 slots comfortably.

### Monitor

```bash
# Memory + CPU
docker stats --no-stream

# Redis: queue depth and AOF size
docker exec lyralabs-redis redis-cli -a "$REDIS_PASSWORD" \
    --no-auth-warning info | grep -E "used_memory_human|aof_current"

docker exec lyralabs-redis redis-cli -a "$REDIS_PASSWORD" \
    --no-auth-warning llen agent
# 0 = idle. Sustained >10 = workers can't keep up; bump --concurrency.
```

### Reliability

The worker has `restart: always`, so a crashed Celery process auto-restarts
within seconds. The VM itself is the only single point of failure. Three
tiers of mitigation, in order of effort:

1. **Today**: nothing. VM crashes are rare (<1/month at GCE's SLA), and the
   API simply queues nothing until the VM returns. Acceptable for MVP.
2. **Next**: enable [GCE auto-healing](https://cloud.google.com/compute/docs/instance-groups/autohealing-instances-in-mig)
   via a Managed Instance Group with a TCP health check on port 6379.
   Recreates the VM on failure.
3. **Later**: run a 2-VM MIG with regional redundancy. At this point you
   probably want managed Redis (back to Upstash or Memorystore) anyway.

### Backups

Redis here is a queue, not a system of record. The only data worth
backing up is in Postgres (Supabase handles that). The Redis AOF only
matters during a graceful shutdown; on hard crash you may lose
in-flight tasks, which Celery's `task_acks_late=True` config will
re-deliver from the queue once redelivered.

---

## Troubleshooting

### Worker connects to Redis fine but tasks never fire

Check that the API is actually pushing to the same Redis the worker is
pulling from. Most common cause: the API's `CELERY_BROKER_URL` still
points at `localhost`. From the API's Cloud Run logs:

```
ConnectionRefusedError: [Errno 111] Connection refused
```

Means the API never updated. Re-run Phase 4.

### Worker logs `AUTH failed`

Password in worker's `.env` doesn't match the one Redis was started with.
On the VM:

```bash
docker compose down
nano .env  # fix REDIS_PASSWORD
docker compose up -d
```

### Redis container restart-looping

Almost always permissions on the AOF file. Check:

```bash
docker compose logs redis | tail -20
```

If you see `Can't open the append-only file`, nuke the volume:

```bash
docker compose down -v   # WARNING: drops queued tasks
docker compose up -d
```

### `MASTER_ENCRYPTION_KEY` mismatch

Worker logs show `cryptography.fernet.InvalidToken` when handling Slack
events. The key on the VM doesn't match the one Cloud Run uses to encrypt
tenant tokens. Copy the value from `lyralabs-app`'s env vars (Cloud Run
console → Variables & Secrets) into `/var/lyralabs/.env` and restart.

### "Connection refused" from API to VM IP

Either the firewall rule didn't apply (recheck `gcloud compute firewall-rules describe lyralabs-worker-redis`)
or the VM's external IP changed (it shouldn't with a static IP attached;
re-check `gcloud compute instances describe lyralabs-worker`).

---

## Private networking (optional, later)

The current setup exposes Redis on the public internet, password-protected.
If you're uncomfortable with that, the upgrade path:

1. Create a [VPC Connector](https://cloud.google.com/vpc/docs/configure-serverless-vpc-access)
   for `lyralabs-app` (~$10/mo).
2. Move the VM into the same VPC.
3. Change the docker-compose `redis.ports` to `"127.0.0.1:6379:6379"` so
   it only binds to the VM's loopback.
4. Update Cloud Run's `CELERY_BROKER_URL` to use the VM's *internal* IP
   (`10.x.x.x`).
5. Delete the public firewall rule.

Defer until you have customers whose data flows through the queue.
