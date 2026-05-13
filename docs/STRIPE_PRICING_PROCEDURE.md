# NorthStar Stripe Pricing Procedure

Date: 2026-05-13  
Current model version: `northstar-credits-v1`

## Recommendation

Use one simple freemium credit model for the beta:

| Stage | Offer | Why |
|---|---|---|
| Free trial | 3 successful rendered videos | Users must see a real MP4 before paying. |
| Creator pack | SGD 9 for 12 additional video credits | Low-friction impulse purchase, enough room to explore, no subscription anxiety. |
| Later subscription | SGD 19-49/month after retention is proven | Add only after users repeatedly exhaust credits. |

Do not start with three subscriptions. NorthStar is still proving render reliability and repeat usage. A single credit pack is clearer, easier to test, and maps directly to platform cost.

## What Counts As A Credit

One credit is charged only after a successful MP4 render is produced.

Do not charge for:

- failed renders
- plan generation
- code generation before render success
- repair attempts
- previewing existing videos
- editing code/timeline

This keeps creator trust high because users pay for completed outputs, not attempts.

## Stripe Setup

Use Stripe Checkout Sessions, not static Payment Links, because the app needs webhook metadata to credit the correct creator balance.

1. Create one product in Stripe:
   - Name: `NorthStar Studio Credits - Creator Pack`
   - Type/category: electronically supplied service
   - Description: `12 NorthStar video render credits for editable Manim animations.`

2. Create one one-time Price:
   - Currency: `SGD`
   - Amount: `9.00`
   - Billing type: one-time

3. Copy the Price ID into Render:
   - `STRIPE_PRICE_9=price_...`

4. Set the server keys in Render:
   - `STRIPE_SECRET_KEY=sk_live_...`
   - `STRIPE_WEBHOOK_SECRET=whsec_...`

5. Configure the webhook endpoint:
   - URL: `https://northstarstudio.io/api/billing/webhook`
   - Event: `checkout.session.completed`

6. Confirm the app reports:
   - `GET /api/billing/status`
   - `stripe_checkout_configured: true`

## App Contract

The backend exposes:

- `GET /api/billing/status`: returns free credits, current balance, pack features, Stripe setup status, and pricing version.
- `POST /api/billing/checkout`: creates a Stripe Checkout Session for `pack_price_usd: 9`.
- `POST /api/billing/webhook`: verifies `Stripe-Signature`, handles `checkout.session.completed`, and adds credits once.

Checkout metadata must include:

- `northstar_user_id`
- `pack_price_usd`
- `video_credits`
- `product=northstar_credits`
- `pricing_model_version=northstar-credits-v1`
- `credit_unit=successful_manim_mp4_render`

## Pricing Copy

Use this public copy:

> Start with 3 free rendered videos. When you are ready, the SGD 9 Creator pack adds 12 more render credits, Pro model access, Crazy mode, more active sessions, editable Manim code, captions, timeline, and share packages.

Short CTA:

> Get 12 Creator credits

## When To Add Subscriptions

Add subscriptions only when analytics show:

- users consume at least 15-25 credits repeatedly
- users return across multiple days
- users share/export videos externally
- support burden is manageable
- render cost per successful video is stable

Suggested later tiers:

| Tier | Price | Use |
|---|---:|---|
| Starter | SGD 9 one-time | Keep as credit pack. |
| Creator | SGD 19/month | Regular short-form explainers. |
| Studio | SGD 49/month | Longer videos, advanced models, more storage. |

Until then, keep the buying decision simple.

## Testing Checklist

1. In Render, confirm `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, and `STRIPE_PRICE_9`.
2. Open `https://northstarstudio.io/app?settings=credits`.
3. Confirm Checkout says `Live`.
4. Click `SGD 9`.
5. Complete Stripe Checkout.
6. Confirm redirect to `/app?checkout=success&pack=9`.
7. Confirm credits increase by 12.
8. Render one MP4.
9. Confirm credits decrease by 1 only after success.
10. Confirm failed renders do not decrease credits.

