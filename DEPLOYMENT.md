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
APP_URL=https://northstar-manim.onrender.com
```

Optional but expected for production:

```bash
GEMINI_API_KEY=...
STRIPE_SECRET_KEY=...
STRIPE_WEBHOOK_SECRET=...
STRIPE_PRICE_2=...
STRIPE_PRICE_4=...
STRIPE_PRICE_8=...
STRIPE_PRICE_10=...
ELEVENLABS_API_KEY=...
```

The Render disk is mounted at `/app/work`, which preserves renders, generated files, source uploads, settings, memories, and skills across deploys.

## Stripe Credits Plan

Use Stripe Checkout Sessions for credits. Do not use static Payment Links for the production app because NorthStar needs to update user credit balances after payment.

Launch product:

- New user trial: 3 free rendered videos
- $2: 2 video credits
- $4: 5 video credits
- $8: 12 video credits
- $10: 16 video credits

Implementation contract:

1. Create one Stripe product: `NorthStar Credits`.
2. Create four one-time prices matching the packs above.
3. Store the price ids in `STRIPE_PRICE_2`, `STRIPE_PRICE_4`, `STRIPE_PRICE_8`, and `STRIPE_PRICE_10`.
4. Add a checkout endpoint: `POST /api/billing/checkout`.
5. Add a webhook endpoint: `POST /api/billing/webhook`.
6. On `checkout.session.completed`, add credits to the signed-in user.
7. Consume one credit only after a successful MP4 render.
8. Failed renders should not consume credits.

The current local build exposes the pricing surface and billing config, but real credit enforcement still needs auth, database-backed users, checkout sessions, and webhooks.

## Domain

Good domain targets:

- `northstar.video`
- `northstarstudio.ai`
- `manimstudio.ai`
- `concept.video`

Recommended first choice: `northstar.video`.

Setup:

1. Buy the domain through Cloudflare Registrar or Namecheap.
2. Add the custom domain in Render.
3. Add the DNS records Render provides.
4. Set `APP_URL=https://yourdomain.com`.
5. Update Stripe success/cancel URLs to use `APP_URL`.

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
