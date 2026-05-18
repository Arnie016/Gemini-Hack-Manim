# NorthStar Queue Worker on Lightsail

NorthStar now supports two render modes:

- `NORTHSTAR_RENDER_MODE=inline`: default. `/api/approve` starts the render in the web process.
- `NORTHSTAR_RENDER_MODE=queue`: `/api/approve` writes a durable JSON job under `work/queue/pending` and returns quickly. A worker process runs `python -m backend.cli worker`.

The default queue is file-backed. Render and the worker must see the same `work/queue` and `work/jobs` filesystem for local queue mode. A separate Render web service and a separate Lightsail VM will not share local disk by default, so cross-host deployments should use S3/R2 queue mode with `NORTHSTAR_QUEUE_STORE=s3`.

## Render Web Env Vars

Set these on the web service that receives `northstarstudio.io` traffic:

```bash
APP_URL=https://northstarstudio.io
CANONICAL_HOST=northstarstudio.io
NORTHSTAR_RENDER_MODE=queue
NORTHSTAR_FREE_VIDEO_CREDITS=3
TEXT_PROVIDER=openai
OPENAI_MODEL=gpt-5
OPENAI_CODE_MODEL=gpt-5-mini
OPENAI_REPAIR_MODEL=gpt-5-mini
OPENAI_API_KEY=<render secret>
GEMINI_API_KEY=<render secret>
STRIPE_SECRET_KEY=<render secret>
STRIPE_WEBHOOK_SECRET=<render secret>
STRIPE_PRICE_9=<render secret>
SUPABASE_URL=<render secret>
SUPABASE_ANON_KEY=<render secret>
SUPABASE_SERVICE_ROLE_KEY=<render secret>
NORTHSTAR_ARTIFACT_STORE=local
NORTHSTAR_ARTIFACT_PREFIX=renders
```

For cross-host Render + Lightsail queueing and S3-compatible artifact publishing from the worker, add:

```bash
NORTHSTAR_QUEUE_STORE=s3
NORTHSTAR_QUEUE_BUCKET=<bucket>
NORTHSTAR_QUEUE_PREFIX=queue
NORTHSTAR_ARTIFACT_STORE=s3
NORTHSTAR_ARTIFACT_BUCKET=<bucket>
NORTHSTAR_ARTIFACT_PREFIX=renders
NORTHSTAR_ARTIFACT_PUBLIC_BASE_URL=<public base URL>
AWS_REGION=ap-southeast-1
AWS_ACCESS_KEY_ID=<secret>
AWS_SECRET_ACCESS_KEY=<secret>
```

For Cloudflare R2 or another S3-compatible endpoint, also set:

```bash
NORTHSTAR_ARTIFACT_ENDPOINT_URL=<s3-compatible endpoint URL>
```

Keep the frontend pointed at `https://northstarstudio.io`. Do not expose or call the Lightsail worker directly from browser code.

## Lightsail Setup

Assume the app is deployed at `/opt/northstar` and the shared/persistent work directory is `/opt/northstar/work`.

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv ffmpeg libcairo2-dev libpango1.0-dev texlive texlive-latex-extra
sudo mkdir -p /opt/northstar
sudo chown -R ubuntu:ubuntu /opt/northstar
cd /opt/northstar
git clone <repo-url> .
git checkout production-evolution-2026-04-22
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m manim --version
mkdir -p work/queue/pending work/queue/claimed work/queue/done work/queue/failed work/jobs
```

Create `/etc/northstar-worker.env`:

```bash
APP_URL=https://northstarstudio.io
CANONICAL_HOST=northstarstudio.io
NORTHSTAR_RENDER_MODE=queue
TEXT_PROVIDER=openai
OPENAI_MODEL=gpt-5
OPENAI_CODE_MODEL=gpt-5-mini
OPENAI_REPAIR_MODEL=gpt-5-mini
OPENAI_API_KEY=<secret>
GEMINI_API_KEY=<secret>
STRIPE_SECRET_KEY=<secret>
STRIPE_WEBHOOK_SECRET=<secret>
STRIPE_PRICE_9=<secret>
SUPABASE_URL=<secret>
SUPABASE_ANON_KEY=<secret>
SUPABASE_SERVICE_ROLE_KEY=<secret>
NORTHSTAR_ARTIFACT_STORE=s3
NORTHSTAR_QUEUE_STORE=s3
NORTHSTAR_QUEUE_BUCKET=<bucket>
NORTHSTAR_QUEUE_PREFIX=queue
NORTHSTAR_ARTIFACT_BUCKET=<bucket>
NORTHSTAR_ARTIFACT_PREFIX=renders
NORTHSTAR_ARTIFACT_PUBLIC_BASE_URL=<public base URL>
AWS_REGION=ap-southeast-1
AWS_ACCESS_KEY_ID=<secret>
AWS_SECRET_ACCESS_KEY=<secret>
```

Create `/etc/systemd/system/northstar-worker.service`:

```ini
[Unit]
Description=NorthStar queue render worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/northstar
EnvironmentFile=/etc/northstar-worker.env
ExecStart=/opt/northstar/.venv/bin/python -m backend.cli worker --poll 2 --concurrency 1 --progress plain
Restart=always
RestartSec=5
KillSignal=SIGINT
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
```

Enable and inspect the worker:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now northstar-worker
sudo systemctl status northstar-worker --no-pager
journalctl -u northstar-worker -f
```

One-shot smoke check:

```bash
cd /opt/northstar
. .venv/bin/activate
python -m backend.cli worker --once --progress plain
```

Rollback to inline rendering:

```bash
sudo systemctl stop northstar-worker
# Set this on the web process:
NORTHSTAR_RENDER_MODE=inline
```
