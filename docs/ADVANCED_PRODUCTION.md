# NorthStar Advanced Production Plan

This is the production target for a social physics-animation platform: users upload sources, generate/edit Manim videos, render long jobs, share public posts, collaborate, and pay for credits or plans.

## Target Architecture

```text
Cloudflare DNS/CDN
  -> Web/API service: FastAPI NorthStar app
  -> Supabase Auth + Postgres: users, projects, sessions, gallery, credits
  -> Queue: SQS/Redis/Celery later
  -> Render workers: Manim + ffmpeg + TeX on CPU instances
  -> Object storage: S3, Lightsail bucket, or Cloudflare R2
  -> Stripe: checkout, webhooks, credit ledger
```

The web service should not run long renders directly once production traffic starts. It should enqueue work, stream status, and read completed artifacts from durable storage.

## Stage 1: Production Media Storage

NorthStar now has an artifact storage boundary in `backend/artifact_store.py`.

Default local mode:

```bash
NORTHSTAR_ARTIFACT_STORE=local
```

S3/Lightsail/R2-compatible mode:

```bash
NORTHSTAR_ARTIFACT_STORE=s3
NORTHSTAR_ARTIFACT_BUCKET=northstarstudio-renders-sg
AWS_REGION=ap-southeast-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
NORTHSTAR_ARTIFACT_PREFIX=renders
NORTHSTAR_ARTIFACT_PUBLIC_BASE_URL=https://cdn.northstarstudio.io
```

For Cloudflare R2, also set:

```bash
NORTHSTAR_ARTIFACT_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
```

When share metadata is generated, NorthStar uploads available artifacts:

- `out.mp4`
- `captions.srt`
- `plan.json`
- `scene.py`
- `manifest.json`
- `share.html`
- `share-copy.md`

The returned manifest includes `remote_artifacts` and upgrades `video_url` when the MP4 is in object storage.

## Stage 2: Supabase Production Data

Create a Supabase project and run:

```text
supabase/migrations/0001_northstar_memory.sql
```

Required buckets:

```text
northstar-sources          private
northstar-assets           private
northstar-public-renders   public or CDN-backed
```

Required app concepts:

- `profiles`: authenticated creators
- `projects`: animation workspaces
- `source_documents`: cheat sheets, PDFs, notes, LaTeX, prompt packs
- `source_chunks`: retrieval-ready chunks
- `animation_runs`: plan/code/render lifecycle
- `assets`: generated images, audio, captions, code, thumbnails, videos
- `share_links`: public social posts
- `publish_events`: YouTube/X/Twitch/etc upload attempts
- `credit_ledger`: Stripe-backed credits

## Stage 3: Render Worker Split

Create two deployable services:

```text
northstar-web
  serves API/UI, auth, billing, gallery, uploads

northstar-worker
  consumes queued render jobs, runs Manim/ffmpeg, uploads artifacts
```

Minimum worker instance:

```text
4 GB RAM / 2 vCPU for short jobs
8 GB RAM / 2+ vCPU for longer videos
```

Worker dependencies:

```bash
ffmpeg
python3
manim
cairo
pango
dvisvgm
texlive-latex-base
texlive-latex-extra
texlive-science
```

## Stage 4: Queue And Billing Rules

Production rules:

1. A render request creates a queued `animation_runs` row.
2. Credits are reserved when the job starts.
3. Credits are consumed only after a valid MP4 exists and passes `ffprobe`.
4. Failed renders release reserved credits.
5. Workers enforce max duration, max scenes, max file size, and timeout.
6. Long jobs never block web requests.

Queue choices:

- Simple first: Supabase table with `status='queued'` and worker polling.
- Better: Redis Queue/Celery.
- AWS scale: SQS + ECS/Fargate workers.

## AWS Lightsail Setup

For the worker, choose:

```text
Platform: Linux/Unix
Blueprint: OS only -> Ubuntu
Region: Singapore ap-southeast-1
Plan: 4 GB RAM minimum, 8 GB preferred
Snapshots: enabled
```

Do not use the OpenClaw, WordPress, Node.js, Django, or Bitnami blueprints for NorthStar render workers.

For object storage:

```text
Bucket: northstarstudio-renders-sg
Region: ap-southeast-1
Start size: 100 GB if you expect public video sharing
```

## Launch Checklist

- [ ] `NORTHSTAR_ARTIFACT_STORE=s3` configured in production.
- [ ] Bucket credentials are server-only.
- [ ] Public videos use CDN/object URLs, not raw `/work` paths.
- [ ] Supabase Auth protects users/projects.
- [ ] Jobs, sources, memories, and gallery rows are owner-scoped.
- [ ] Stripe webhook updates an idempotent credit ledger.
- [ ] Render worker has a hard concurrency cap.
- [ ] Successful MP4s pass `ffprobe`.
- [ ] Failed renders do not consume credits.
- [ ] Share pages include NorthStar attribution.

## Immediate Next Engineering Steps

1. Replace public `/work` serving with signed artifact routes.
2. Persist job metadata to Supabase `animation_runs`.
3. Add `sources/upload` backend endpoint for PDFs, MD, LaTeX, images, and folders.
4. Add a worker process command that polls queued jobs.
5. Move billing from anonymous cookies into Supabase user credit ledger.
