# NorthStar Deployment

NorthStar should deploy as one Docker-backed FastAPI service first. The app renders Manim videos, writes files under `work/`, and needs ffmpeg plus TeX support, so a generic serverless deployment is not the right first production target.

## Recommended Platform

Use Render with the committed `render.yaml` Blueprint:

- Service: Docker web service
- Health check: `/api/live`
- Persistent disk: `/app/work`
- Start command: handled by `Dockerfile`
- Public app paths:
  - `/` landing page
  - `/app` creator IDE

## Render Setup

1. Push this branch to GitHub.
2. In Render, create a Blueprint from the repo.
3. Select the `production-evolution-2026-04-22` branch.
4. Confirm the service name `northstar-manim`.
5. Set environment variables:

```bash
OPENAI_API_KEY=...
APP_URL=https://northstarstudio.io
CANONICAL_HOST=northstarstudio.io
NORTHSTAR_FREE_VIDEO_CREDITS=3
```

Optional but expected for production:

```bash
GEMINI_API_KEY=...
STRIPE_SECRET_KEY=...
STRIPE_WEBHOOK_SECRET=...
STRIPE_PRICE_9=...
SUPABASE_URL=...
SUPABASE_ANON_KEY=...
SUPABASE_SERVICE_ROLE_KEY=...
ELEVENLABS_API_KEY=...
```

The Render disk is mounted at `/app/work`, which preserves renders, generated files, source uploads, settings, memories, and skills across deploys.

## Stripe Credits Plan

Use Stripe Checkout Sessions for credits. Do not use static Payment Links for the production app because NorthStar needs to update user credit balances after payment.

Launch product:

- New user trial: 3 free rendered videos
- Creator credit pack: SGD 9, 12 video credits

Implemented launch contract:

1. Create one Stripe product: `NorthStar Studio`.
2. Create one one-time price for SGD 9.
3. Store the price id in `STRIPE_PRICE_9`.
4. `GET /api/billing/status` creates an anonymous creator cookie and reports balance.
5. `POST /api/billing/checkout` creates a Stripe Checkout Session for a selected pack.
6. `POST /api/billing/webhook` verifies `Stripe-Signature` and handles `checkout.session.completed`.
7. Successful webhooks add credits to the anonymous creator balance.
8. Render endpoints require credits before render and consume one credit only after a successful MP4 render.
9. Failed renders do not consume credits.

This is good enough for an early hosted beta, but it is still anonymous-cookie billing. Before a larger launch, replace the local JSON credit store with real auth and a database-backed user/account table.

## Domain

Good domain targets:

- `northstar.video`
- `northstarstudio.ai`
- `manimstudio.ai`
- `concept.video`

Recommended first choice: `northstarstudio.io`.

Setup:

1. Buy the domain through Cloudflare Registrar or Namecheap.
2. Add `northstarstudio.io` as a custom domain in Render.
3. Add `www.northstarstudio.io` as a second custom domain or CNAME it to the Render target.
4. In Namecheap, do not use URL redirect to `http://www...`.
5. Point DNS to Render using the records Render gives you.
6. Set `APP_URL=https://northstarstudio.io`.
7. Set `CANONICAL_HOST=northstarstudio.io`.
8. In Stripe, set the webhook URL to `https://northstarstudio.io/api/billing/webhook`.

## Agent Mail

Use Google Workspace for human mail and Resend or Postmark for transactional mail.

Recommended addresses:

- `hello@yourdomain.com`
- `support@yourdomain.com`
- `renders@yourdomain.com`

Product emails to add:

- Render complete with share link
- Low credit warning
- Recharge receipt
- Failed render report with job id and logs
- Weekly creator suggestions

Do not ship production email from a personal Gmail address.
