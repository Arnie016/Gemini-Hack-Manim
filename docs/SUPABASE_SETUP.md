# NorthStar Supabase Setup

This is the production data layer for NorthStar: users, projects, cheat-sheet sources, indexed memory, render history, gallery assets, share links, and future publishing events.

## 1. Create the Supabase project

1. Create a new Supabase project.
2. In Database > Extensions, enable `vector` and `pgcrypto`.
3. In SQL Editor, run `supabase/migrations/0001_northstar_memory.sql`.
4. In Authentication > Providers, enable email login first. Add Google/GitHub later. The migration creates a profile row automatically for every new auth user.
5. In Storage, confirm these buckets exist:
   - `northstar-sources` private: cheat sheets, notes, prompt packs.
   - `northstar-assets` private: generated images, audio, code, captions.
   - `northstar-public-renders` public: public gallery/share videos.

## 2. Environment variables

Set these on Render and in local `.env`:

```bash
SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_ANON_KEY=...
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_STORAGE_BUCKET_SOURCES=northstar-sources
SUPABASE_STORAGE_BUCKET_ASSETS=northstar-assets
SUPABASE_STORAGE_BUCKET_PUBLIC=northstar-public-renders
```

Keep `SUPABASE_SERVICE_ROLE_KEY` server-only. Never expose it in `web/index.html`.

## 3. How the tables map to product features

- `projects`: one creator workspace or animation series.
- `source_documents`: uploaded cheat sheets, notes, outlines, references, prompt packs, YouTube/web sources.
- `source_chunks`: retrieval-ready text chunks with optional embeddings.
- `memories`: durable creator preferences and reusable facts.
- `animation_runs`: every plan/code/render lifecycle record.
- `assets`: video, image, audio, code, captions, share bundles, future Blender/LaTeX assets.
- `share_links`: public links that promote NorthStar from every render.
- `publish_events`: future YouTube/Twitch/X upload attempts and audit trail.
- `user_provider_configs`: bring-your-own provider configuration metadata. Store real keys in a vault, not this table.

## 4. Cheat-sheet upload flow

The hosted flow should be:

1. Upload file to `northstar-sources/{user_id}/{project_id}/{source_id}/original`.
2. Extract text server-side.
3. Store metadata and summary in `source_documents`.
4. Split content into chunks and store in `source_chunks`.
5. Generate embeddings using your selected embedding provider.
6. Add top extracted rules, formulas, and user preferences into `memories`.
7. During planning, retrieve chunks by project, source kind, tags, and semantic similarity.

Use `kind='cheat_sheet'` for PDFs/images/markdown that should strongly ground the animation.

## 5. Row-level security model

The migration enables RLS on user-owned tables. Users can only read/write rows where `owner_id = auth.uid()`. Public share pages are readable only when `share_links.is_public = true`.

Storage objects are also protected. Use paths that start with the user's auth id:

```text
northstar-sources/{auth.uid()}/{project_id}/{source_id}/original.pdf
northstar-assets/{auth.uid()}/{run_id}/generated.png
northstar-public-renders/{auth.uid()}/{run_id}/out.mp4
```

Backend jobs that need privileged operations should use the service role key. Browser requests should use Supabase Auth plus the anon key.

## 6. Provider keys and API freedom

Do not store raw OpenAI, Gemini, Anthropic, X, YouTube, or Twitch secrets in normal Postgres rows. Use one of:

- Supabase Vault, with `user_provider_configs.vault_secret_id`.
- A dedicated encrypted secrets manager.
- Bring-your-own-key only in local desktop mode.

The UI can safely store provider name, label, preferred model, and status in `user_provider_configs`.

## 7. Stripe plan mapping

Recommended launch pricing:

- Free trial: 3 rendered videos.
- `Creator credits` at `SGD 9`: 12 additional render credits.

Create one one-time Stripe Price and store its id in `STRIPE_PRICE_9`. Later, if subscription tiers are added, store the active plan in `profiles.plan_key`.

## 8. Publishing integrations

The database is ready for direct publishing, but each platform still needs OAuth apps:

- YouTube: Google Cloud OAuth app, YouTube Data API, upload scope.
- X: X developer app, OAuth 2.0, post/tweet scope.
- Twitch: Twitch developer app, OAuth scopes for channel/video workflows.

When upload is connected, write each attempt to `publish_events` with platform, status, external URL, error, and metadata.

## 9. Local-to-production migration path

Current local job artifacts live under `work/jobs/{job_id}`. For hosted production:

1. Keep render artifacts local while a job is running.
2. On completion, upload MP4, captions, thumbnail, code, and share bundle to Supabase Storage.
3. Upsert `animation_runs` with `local_job_id`, `video_path`, `thumbnail_path`, and `metadata`.
4. Create a `share_links` row if the creator chooses public sharing.
5. Hydrate the Gallery tab from `animation_runs` instead of local-only `renderDeck`.

## 10. Production checklist

- RLS enabled and tested with two users.
- Service role key only on the backend.
- Storage buckets have size limits and private/public separation.
- Upload limits enforced before extraction.
- Embeddings generated asynchronously.
- Stripe webhook updates `profiles.plan_key` and credits idempotently.
- Share pages include canonical URLs and NorthStar attribution.
- Publishing jobs are auditable in `publish_events`.
