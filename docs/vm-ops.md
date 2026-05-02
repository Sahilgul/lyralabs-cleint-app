# VM Operations — Manual Commands

Quick reference for the GCE VM (`lyralabs-worker`, `us-east1-b`) hosting Redis + Celery worker.

- **Workspace dir on VM:** `/var/lyralabs/`
- **Docker config dir on VM:** `/var/lib/google/lyralabs/.docker/` (compose plugin lives here)
- **Static external IP:** `34.75.67.127`
- **OS:** Container-Optimized OS (COS) — `/root` is ephemeral, `/usr` is read-only

---

## 0. Connect to the VM

From your laptop:

```bash
gcloud compute ssh lyralabs-worker --zone us-east1-b --project clinicalcopilot
sudo su -
cd /var/lyralabs
```

---

## 1. One-time setup: fix `docker compose` permanently

COS wipes `/root/` on every boot, so the docker-compose plugin symlink keeps disappearing. This systemd oneshot service rebuilds it on every boot, so you never have to `export DOCKER_CONFIG=...` again.

**Run once:**

```bash
cat > /etc/systemd/system/docker-compose-link.service <<'EOF'
[Unit]
Description=Symlink docker-compose plugin into /root/.docker/cli-plugins (rebuilt on every boot since /root is ephemeral on COS)
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/sh -c 'mkdir -p /root/.docker/cli-plugins && ln -sf /var/lib/google/lyralabs/.docker/cli-plugins/docker-compose /root/.docker/cli-plugins/docker-compose'

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now docker-compose-link.service
docker compose version  # confirm — should print v2.x.x
```

### Quick fallback if the systemd unit hasn't been set up yet

```bash
export DOCKER_CONFIG=/var/lib/google/lyralabs/.docker
docker compose version
```

This works for the current shell only.

### If the plugin binary itself is missing (`/var/lib/google/lyralabs/.docker/cli-plugins/docker-compose` is gone)

```bash
mkdir -p /var/lib/google/lyralabs/.docker/cli-plugins
curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
  -o /var/lib/google/lyralabs/.docker/cli-plugins/docker-compose
chmod +x /var/lib/google/lyralabs/.docker/cli-plugins/docker-compose
docker compose version
```

---

## 2. Deploy a new image

After Cloud Build finishes for a `main` push, grab the new SHA and roll the worker.

### One-line deploy script (recommended — set up once)

```bash
cat > /var/lyralabs/deploy.sh <<'EOF'
#!/bin/sh
# Usage: ./deploy.sh <sha256-digest>   (with or without "sha256:" prefix)
set -e
SHA="${1#sha256:}"
[ -z "$SHA" ] && { echo "usage: $0 <sha256-digest>"; exit 1; }
cd /var/lyralabs
sed -i "s|@sha256:[a-f0-9]\+|@sha256:$SHA|" .env
echo "==> .env updated:"
grep WORKER_IMAGE .env
docker compose pull worker
docker compose up -d --force-recreate worker
docker image prune -a -f --filter "until=24h"
echo "==> tailing worker logs (Ctrl-C to detach, container keeps running)"
docker compose logs --tail=50 -f worker
EOF
chmod +x /var/lyralabs/deploy.sh
```

Then every deploy is just:

```bash
/var/lyralabs/deploy.sh a6e5abe31f29c2fa4238d1d03e2631e26cc898123fd78df0ba70cf6a4280f663
```

### Manual deploy (the long way)

```bash
cd /var/lyralabs
NEW_SHA="sha256:a6e5abe31f29c2fa4238d1d03e2631e26cc898123fd78df0ba70cf6a4280f663"
sed -i "s|@sha256:[a-f0-9]\+|@${NEW_SHA}|" .env
grep WORKER_IMAGE .env  # verify
docker compose pull worker
docker compose up -d --force-recreate worker
docker compose logs -f worker
```

### Find the latest SHA from Artifact Registry

From your laptop (or VM if `gcloud` is installed):

```bash
gcloud artifacts docker images list \
  us-east1-docker.pkg.dev/clinicalcopilot/cloud-run-source-deploy/lyralabs-cleint-app/lyralabs-app \
  --include-tags --sort-by=~UPDATE_TIME --limit=3 \
  --format="value(version,createTime,tags)"
```

---

## 3. Disk cleanup

The VM is `e2-small` with limited disk. Each redeploy leaves the previous image as dangling layers. Clean periodically.

### Check what's eating space

```bash
df -h /              # filesystem usage
docker system df     # docker breakdown (images, containers, volumes, build cache)
docker images        # list all images and sizes
```

### Safe cleanup (keeps everything that running containers reference)

```bash
docker image prune -a -f
```

Removes all images not used by a running container. Safe for production: the active worker + redis images stay because their containers reference them.

### Aggressive cleanup (also removes stopped containers, networks, build cache)

```bash
docker system prune -a -f
```

Still safe — never touches running containers or named volumes. Don't add `--volumes` unless you want to wipe Redis AOF persistence.

### Keep last 24h, prune older (good default for the deploy script)

```bash
docker image prune -a -f --filter "until=24h"
```

---

## 4. Day-to-day operations

### Container status

```bash
docker compose ps                        # which containers, status, ports
docker stats --no-stream                 # CPU / memory snapshot
```

### Logs

```bash
docker compose logs -f worker            # follow worker
docker compose logs --tail=200 worker    # last 200 lines, no follow
docker compose logs -f redis             # redis logs
docker compose logs -f                   # all services
```

### Restart without re-pulling

```bash
docker compose restart worker            # graceful restart, same image
docker compose up -d --force-recreate worker  # tear down + recreate
```

### Stop / start the whole stack

```bash
docker compose down                      # stop + remove containers (keeps volumes)
docker compose up -d                     # start everything detached
```

### Run a one-off command in the worker image (e.g. Alembic)

```bash
docker compose run --rm worker alembic upgrade head
docker compose run --rm worker python -c "from packages.lyra_core.common.config import get_settings; print(get_settings().database_url[:40])"
```

---

## 5. Redis access

Connection (from inside the VM):

```bash
docker compose exec redis redis-cli -a "$(grep '^REDIS_PASSWORD=' /var/lyralabs/.env | cut -d= -f2)"
```

Then from inside `redis-cli`:

```
PING                       # → PONG
INFO clients               # connected clients
KEYS celery-task-meta-*    # backend results
LLEN agent                 # queue depth (pending tasks)
DBSIZE                     # total keys
FLUSHDB                    # ⚠ wipe everything
```

---

## 6. Diagnostics / health checks

### Is the worker healthy?

```bash
docker compose logs --tail=20 worker | grep -E "ready|received|succeeded|ERROR"
```

Look for: `celery@<hostname> ready.` after boot, then `Task ... received` / `succeeded` for each Slack message.

### Are there errors?

```bash
docker compose logs --since=1h worker 2>&1 | grep -iE "error|exception|traceback" | head -50
```

### Is Redis reachable from Cloud Run?

From your laptop:

```bash
nc -zv 34.75.67.127 6379
```

(Should connect, then close.)

### Postgres connectivity from inside the worker container

```bash
docker compose exec worker python -c "
import asyncio
from packages.lyra_core.db.session import session_scope
from sqlalchemy import text
async def go():
    async with session_scope() as s:
        r = await s.execute(text('select version()'))
        print(r.scalar())
asyncio.run(go())
"
```

---

## 7. Recovery scenarios

### Worker is stuck / OOMing / crashing

```bash
docker compose down worker
docker compose up -d worker
docker compose logs -f worker
```

### Bad deploy — roll back to previous SHA

Look up the previous SHA in Artifact Registry (see §2), then:

```bash
/var/lyralabs/deploy.sh <previous-sha>
```

### VM is full

```bash
docker system prune -a -f
journalctl --vacuum-time=2d         # trim systemd journal
df -h /                             # confirm reclaimed space
```

### After VM reboot

The systemd unit from §1 re-creates the docker-compose symlink automatically. Containers are configured with `restart: unless-stopped` in `docker-compose.yml`, so worker + Redis come back up on their own. Verify:

```bash
docker compose ps
docker compose logs --tail=20 worker
```

---

## 8. Edit `.env` on the VM

```bash
cd /var/lyralabs
nano .env                            # or vim
# After editing, restart so changes take effect:
docker compose up -d --force-recreate worker
```

Common keys to check after a redeploy:

- `WORKER_IMAGE` — the SHA being run
- `DATABASE_URL` — Supabase pooler URL
- `REDIS_URL` / `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND`
- `MASTER_ENCRYPTION_KEY` — must match Cloud Run + local
- `SLACK_*`, `DASHSCOPE_API_KEY`, `DEEPSEEK_API_KEY`, etc.

---

## 9. Useful Cloud Run commands (run from laptop)

### Logs

```bash
gcloud run services logs tail lyralabs-app --region us-east1 --project clinicalcopilot
gcloud run services logs read lyralabs-app --region us-east1 --limit=100
```

### Update env vars without a redeploy

```bash
gcloud run services update lyralabs-app \
  --region us-east1 \
  --update-env-vars "KEY1=value1,KEY2=value2"
```

### Force a new revision (e.g. after secret rotation)

```bash
gcloud run services update lyralabs-app --region us-east1 --no-traffic
gcloud run services update-traffic lyralabs-app --region us-east1 --to-latest
```

---

## 10. Quick reference card

| Task | Command |
|---|---|
| SSH in | `gcloud compute ssh lyralabs-worker --zone us-east1-b` |
| Become root | `sudo su -` then `cd /var/lyralabs` |
| Fix `docker compose` (this session) | `export DOCKER_CONFIG=/var/lib/google/lyralabs/.docker` |
| Deploy new SHA | `/var/lyralabs/deploy.sh <sha>` |
| Tail worker logs | `docker compose logs -f worker` |
| Restart worker | `docker compose restart worker` |
| Disk usage | `df -h /` and `docker system df` |
| Free disk space | `docker image prune -a -f` |
| Run migrations | `docker compose run --rm worker alembic upgrade head` |
| Redis CLI | `docker compose exec redis redis-cli -a "$REDIS_PASSWORD"` |
