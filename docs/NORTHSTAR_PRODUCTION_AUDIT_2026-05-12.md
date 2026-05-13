# NorthStar Studio Production Audit

Date: 2026-05-12  
Scope: agentic IDE for physics, math, and source-guided Manim storytelling animations  
Workspace audited: `/Users/arnav/Desktop/Gemini-Hack-Manim-main/.production-branch`

## Executive Summary

NorthStar Studio has a strong private-beta foundation: it can plan animations, generate Manim code, render MP4s, repair failed code, accept source material, expose generated code, package share artifacts, integrate Stripe Checkout, and deploy through Render with Docker. That is already far beyond a normal hackathon demo.

The main issue is that the product currently exposes too much internal machinery before users get value. It feels like a powerful builder IDE rather than a low-cognitive-effort animation product for teachers, creators, students, and technical storytellers. To become production-level, NorthStar needs three shifts:

1. **Creator-first UX:** default to prompt/source upload -> storyboard -> render -> share. Hide terminal, raw JSON, API settings, and advanced controls until needed.
2. **Reliable render pipeline:** unify every render path, enforce physics-safe Manim patterns, add real render smoke checks, validate MP4 output, and make failure recovery understandable.
3. **Durable project system:** move sources, sessions, assets, credits, share links, and memory from browser/local JSON into authenticated backend storage, likely Supabase.

Readiness estimate:

| Area | Current Level | Production Target | Notes |
|---|---:|---:|---|
| Core prompt-to-render demo | 70% | 90% | Works, but render reliability and failure UX need hardening. |
| First-run UX | 45% | 90% | Too many controls; setup/API concepts leak into main flow. |
| Render reliability | 60% | 90% | Good repair/fallback base, but needs smoke checks and unified pipeline. |
| Source/LaTeX grounding | 45% | 90% | Promising, but browser-local and lossy. |
| Timeline/editor | 35% | 85% | Storyboard editor, not yet a professional timeline. |
| Asset/gallery/share | 45% | 85% | Useful packages exist, but persistence and publishing are shallow. |
| Billing/business readiness | 35% | 90% | Stripe works for beta; needs auth-backed ledger. |
| Security/multi-user | 20% | 95% | Biggest blocker for public launch. |

## Highest Priority Blockers

### 1. No Real Auth Or Tenant Boundary

Most backend routes operate on shared server state: settings, files, memories, skills, jobs, downloads, share pages, terminal, and provider keys. Job artifacts are also exposed from `/work`.

Relevant files:
- `/Users/arnav/Desktop/Gemini-Hack-Manim-main/.production-branch/backend/main.py`
- `/Users/arnav/Desktop/Gemini-Hack-Manim-main/.production-branch/backend/settings_store.py`
- `/Users/arnav/Desktop/Gemini-Hack-Manim-main/.production-branch/backend/file_store.py`
- `/Users/arnav/Desktop/Gemini-Hack-Manim-main/.production-branch/supabase/migrations/0001_northstar_memory.sql`

Production fix:
- Add Supabase Auth or server-side sessions.
- Enforce ownership checks for jobs, files, memories, assets, skills, share links, provider settings, and billing.
- Keep platform OpenAI keys server-side; make BYOK an authenticated, encrypted/Vault-backed feature.
- Serve private artifacts through signed URLs instead of mounting all of `/work` publicly.

### 2. Browser Terminal Is Not Production Safe

`/api/terminal/run` accepts arbitrary-ish commands and runs through `shell=True` with partial string blocking. This is acceptable for a local developer IDE, not a public hosted product.

Relevant files:
- `/Users/arnav/Desktop/Gemini-Hack-Manim-main/.production-branch/backend/main.py`
- `/Users/arnav/Desktop/Gemini-Hack-Manim-main/.production-branch/backend/terminal_runner.py`

Production fix:
- Disable terminal in hosted mode.
- Replace it with strict diagnostics: `health`, `manim version`, `ffmpeg version`, `disk usage`, `render smoke`.
- Keep the full terminal only for local desktop/developer mode.

### 3. First Screen Has Too Much Machinery

The UI exposes Explorer, workspace files, creative sliders, backend health, code/terminal controls, context/memory/skills, model picker, timeline internals, and raw plan JSON before the user knows what to do.

Relevant file:
- `/Users/arnav/Desktop/Gemini-Hack-Manim-main/.production-branch/web/index.html`

Production fix:
- Add a default **Creator Mode**:
  - Big prompt/source upload area.
  - Length, audience, style, aspect chips.
  - One primary action: `Create animation`.
  - Storyboard cards.
  - Preview.
  - Share/download.
- Move code, terminal, raw JSON, Python path, model IDs, and advanced settings behind `Advanced`.

### 4. Render Health Is Too Shallow

Render currently proves process liveness, not render readiness. Preflight checks dependencies but does not prove Manim can render Text/MathTex, ffmpeg can postprocess, or disk has enough free space.

Relevant files:
- `/Users/arnav/Desktop/Gemini-Hack-Manim-main/.production-branch/backend/main.py`
- `/Users/arnav/Desktop/Gemini-Hack-Manim-main/.production-branch/render.yaml`
- `/Users/arnav/Desktop/Gemini-Hack-Manim-main/.production-branch/Dockerfile`

Production fix:
- Add `/api/render-health` with cached probes:
  - `python -m manim --version`
  - `ffmpeg -version`
  - LaTeX/dvisvgm availability
  - writable output directory
  - disk free threshold
  - tiny Text render smoke test
  - optional MathTex smoke test
  - ffprobe output validation
- Keep Render health endpoint fast, but expose deeper checks in UI/admin.

### 5. Sources And Memory Are Too Browser-Local

Sources are mostly `localStorage` plus copied text files. Supabase schema exists, but runtime does not use it yet. Uploaded files are summarized client-side, clipped, then passed through `director_brief`, so source grounding is lossy.

Relevant files:
- `/Users/arnav/Desktop/Gemini-Hack-Manim-main/.production-branch/web/index.html`
- `/Users/arnav/Desktop/Gemini-Hack-Manim-main/.production-branch/backend/context_store.py`
- `/Users/arnav/Desktop/Gemini-Hack-Manim-main/.production-branch/backend/file_store.py`
- `/Users/arnav/Desktop/Gemini-Hack-Manim-main/.production-branch/docs/SUPABASE_SETUP.md`
- `/Users/arnav/Desktop/Gemini-Hack-Manim-main/.production-branch/supabase/migrations/0001_northstar_memory.sql`

Production fix:
- Add backend `/api/sources/upload` multipart endpoint.
- Store originals, extracted text, metadata, chunks, embeddings, and source provenance.
- Use Supabase tables for `projects`, `source_documents`, `source_chunks`, `memories`, `animation_runs`, and `assets`.
- Use local JSON/disk only as desktop fallback.

## UX And Product Improvements

### Replace Raw JSON Review With Storyboard Review

Current flow asks users to inspect editable plan JSON. Non-coders need storyboard cards instead:

- Scene title
- Duration
- Visual goal
- Equation
- Narration
- Source evidence
- Required labels
- Risk warnings

Keep raw JSON under `Advanced edit`.

### Make First-Run Checklist Adaptive

Current checklist can imply image generation is required. It should adapt:

1. Describe animation.
2. Optional: attach cheat sheet, LaTeX, notes, PDF, or images.
3. Review storyboard.
4. Render.
5. Share/download.

### Simplify Settings

Split settings into:

- `Account & credits`
- `Rendering readiness`
- `Providers`
- `Advanced local setup`

Do not show API keys, Python executables, model IDs, or terminal commands in the normal creator path.

### Failure UX

Replace technical failure dumps with clear action categories:

- Renderer missing
- Scene too complex
- LaTeX failed
- Model/API quota
- Asset missing
- Temporary render failure

Actions:

- `Simplify and retry`
- `Use storyboard fallback`
- `Open setup fix`
- `Show logs`
- `Report issue`

## Render Reliability Improvements

### Unify Render Entry Points

Currently `/api/approve`, `/api/animate`, and `/api/render-code` have different render paths and different repair/fallback behavior.

Production fix:
- Route all rendering through one async job pipeline:
  - plan normalization
  - dependency preflight
  - static code safety
  - render
  - repair
  - fallback
  - postprocess
  - ffprobe validation
  - billing settlement

Relevant files:
- `/Users/arnav/Desktop/Gemini-Hack-Manim-main/.production-branch/backend/main.py`
- `/Users/arnav/Desktop/Gemini-Hack-Manim-main/.production-branch/backend/job_manager.py`
- `/Users/arnav/Desktop/Gemini-Hack-Manim-main/.production-branch/backend/renderer_stream.py`
- `/Users/arnav/Desktop/Gemini-Hack-Manim-main/.production-branch/backend/video_postprocess.py`

### Add Physics-Safe Generation Contracts

The current prompt has generic safety rules. Physics animation needs domain-specific constraints:

- No real-time numerical simulations during render.
- No unbounded loops.
- Max sampled points for curves.
- Bounded `ValueTracker`.
- One moving marker/vector per scene by default.
- Stable axes ranges.
- Explicit units and labels.
- Use canonical scene primitives.

Needed primitives:

- Wave animator
- Vector/free-body diagram
- Equation reveal
- Graph builder
- Energy bar
- Number line
- Phase diagram
- Circuit sketch
- Lab apparatus
- Definition card

Relevant files:
- `/Users/arnav/Desktop/Gemini-Hack-Manim-main/.production-branch/backend/prompts.py`
- `/Users/arnav/Desktop/Gemini-Hack-Manim-main/.production-branch/backend/code_format.py`
- `/Users/arnav/Desktop/Gemini-Hack-Manim-main/.production-branch/backend/job_manager.py`

### Strengthen Static Safety Checks

Add checks for:

- Exactly one `GeneratedScene`.
- Reject `while`.
- Count `ValueTracker`.
- Validate `ImageMobject` paths.
- Limit `Axes` coordinate ranges.
- Limit text/mobject counts.
- Detect unsupported Manim APIs.
- Detect risky MathTex blocks.
- Reject large nested `VGroup`/animation loops.

### Validate MP4 Outputs

Postprocess should not silently accept unnormalized output in hosted mode.

Validate with `ffprobe`:

- codec is H.264
- pixel format is `yuv420p`
- duration > 0
- size under cap
- resolution/aspect matches target
- playable output with faststart

### Add Queue And Concurrency Limits

Render starter instances should not run multiple heavy jobs at once.

Add:

- global semaphore or queue
- queue position in UI
- per-user concurrency cap
- job TTL cleanup
- disk usage monitoring
- dead-job recovery

## Source, LaTeX, And Memory Improvements

### Backend Source Upload Endpoint

Add `/api/sources/upload`:

- multipart upload
- preserve folder paths
- support Markdown, TXT, CSV, JSON, YAML, SRT/VTT, TeX, PY, PDF, DOCX
- extract text
- create source document records
- create chunks
- calculate metadata
- attach to project/session

### PDF And Image Source Handling

Current PDF handling is asset-level, not source-grounded. Needed:

- PDF text extraction
- scanned PDF OCR
- image OCR for screenshots/cheat sheets
- extracted diagrams as possible visual references
- page-level source citations

### LaTeX Pipeline

Current LaTeX extraction is heuristic. Needed:

- extract equations from uploaded/pasted text
- normalize macros
- split long align blocks
- detect unsupported commands
- preflight small MathTex compile
- fallback to plain Text when TeX unavailable
- attach equations to scene schema

### Scene Schema Extensions

Extend plan JSON:

```json
{
  "source_refs": ["src_123#chunk_4"],
  "equations": ["E = hf - \\phi"],
  "visual_goal": "Show photon energy crossing threshold",
  "misconceptions": ["Brighter red light does not eject electrons below threshold"],
  "required_labels": ["threshold frequency", "work function"]
}
```

This reduces hallucination and makes the animation more educational.

### Automatic Memory Extraction

After upload, automatically extract:

- formulas
- definitions
- topic level
- user preferences
- required facts
- recurring style constraints
- source provenance

Store project-scoped memory and retrieve automatically. Do not make users manually select memory for normal flows.

## Timeline, Editor, Assets, And Gallery

### Timeline

Current timeline is a storyboard editor. A production animation IDE needs:

- scene track
- caption track
- voiceover track
- music/SFX track
- overlay/image track
- markers
- snapping
- transitions
- keyframes or per-object timing
- waveform lane
- undoable commands

Store timeline as canonical project JSON and render from that.

### Code Editor

Current editor is a textarea/highlight overlay. Upgrade to CodeMirror 6 or Monaco:

- Python highlighting
- lint diagnostics
- line mapping from render errors
- Manim snippets
- symbol navigation
- AI patch/diff review
- autocomplete for safe primitives
- command palette

### Asset Library

Dropped assets often live as object URLs and do not persist reliably.

Needed asset API:

- upload originals
- metadata extraction
- thumbnails
- waveforms for audio
- preview frames for video
- source/license notes
- persistent project references
- timeline placement

### Image Generation Workflow

Current image generation is useful but shallow. Add:

- per-scene image prompts
- accepted/rejected variants
- prompt lineage
- version history
- aspect validation
- background removal
- upscaling
- style lock
- image metadata in manifest

### Voiceover And Audio

Current voiceover can generate and mux audio, but it needs:

- per-scene script lanes
- waveform preview
- timing fit
- silence trimming
- volume ducking
- pronunciation dictionary
- voice preview
- per-scene retakes
- caption sync from final audio

Potential bug to fix: context cues should not be prepended into spoken TTS script.

### Gallery

Current gallery is mostly client/session-level plus presets. Needed:

- backend `/api/jobs` listing
- thumbnails
- duration/aspect/model/source metadata
- tags/folders/favorites
- open old project state
- version history
- search
- package/share/download actions

## Business, Stripe, Deployment, And Ops

### Billing

Current Stripe credits are good enough for trusted beta, but not public launch:

- anonymous cookie credits can be reset
- ledger is filesystem-backed
- free trial is browser-specific
- Pro/Crazy gating is partly frontend-only
- webhook handles mainly checkout completion

Production fix:

- Auth-backed user credit ledger in Supabase.
- Stripe Checkout metadata tied to authenticated user.
- Idempotent credit transactions.
- Refund/dispute/expired-session handling.
- Admin credit adjustment tools.

### Provider Keys

Hosted mode should use platform keys by default. BYOK should require:

- auth
- encryption/Vault/secrets manager
- no shared `work/config.json`
- clear usage attribution

### Supabase Path

The schema is already a strong start. Implement a thin data layer first for:

- authenticated user/profile
- projects
- animation runs
- source documents/chunks
- assets
- share links
- credit ledger

Keep render scratch files on disk temporarily, then move durable assets to Supabase Storage or object storage.

### Security

Before public launch:

- Auth and ownership checks.
- Disable hosted terminal.
- CSRF protection for cookie state.
- Rate limits.
- Strict CORS.
- CSP.
- Request body limits.
- Upload scanning/validation.
- Sandboxed Manim subprocesses.
- Private artifact serving.

### Observability

Add:

- structured JSON logs
- request ID, user ID, job ID, Stripe event ID
- Sentry or equivalent
- uptime monitoring
- provider latency/cost metrics
- render queue dashboard
- dead job recovery
- storage cleanup jobs
- admin tools

### Sharing And Virality

Share packages are a real strength. Improve with:

- branded slugs
- persistent share table
- consent control before public sharing
- OG image generation
- thumbnail frame picker
- view counts
- platform presets: YouTube Shorts, TikTok, Reels, X, Reddit, classroom
- public share pages that say “Made with NorthStar Studio”

## Minor Tweaks With High Leverage

1. Rename `Approve` to `Storyboard` or `Review`.
2. Rename `Code` to `Code (Advanced)`.
3. Hide terminal by default.
4. Hide raw JSON by default.
5. Replace `/health`, `/settings`, `/attach` hints with visible buttons.
6. Make “Use platform credits” the default.
7. Make “Bring your own key” advanced.
8. Make render status copy less technical.
9. Show queue position and expected render time.
10. Add “Simplify and retry” after failed renders.
11. Add “Use deterministic storyboard” as explicit fallback.
12. Add source citation badges on storyboard scenes.
13. Add `MathTex check passed/failed` in readiness.
14. Add persistent “Project saved” state.
15. Add thumbnail extraction after render.
16. Add “Open last render” and “Reopen project.”
17. Add aspect presets with platform labels.
18. Add one-click export presets.
19. Add sample projects with real rendered MP4s.
20. Update `screenshot/UI.png` to remove hackathon-era visual language.

## Recommended Roadmap

### Phase 0: Keep Private Beta Safe

Goal: make the current hosted app safe enough for trusted testers.

- Disable terminal in hosted mode.
- Hide advanced settings by default.
- Add owner-ish job cookie checks as a temporary bridge.
- Add render queue concurrency cap.
- Add render health endpoint.
- Improve failure classification.
- Update UI to Creator Mode first screen.

### Phase 1: Make Prompt-To-Render Reliable

Goal: a user can create a short physics animation without understanding Manim.

- Storyboard cards instead of JSON review.
- Unified render pipeline.
- Physics-safe prompt contract.
- Stronger static safety checks.
- MathTex preflight/fallback.
- MP4 ffprobe validation.
- Render fallback clearly labeled.

### Phase 2: Make Source-Guided Animations Real

Goal: upload notes/cheat sheets/LaTeX/PDF and get grounded animations.

- `/api/sources/upload`.
- PDF/DOCX/text extraction.
- Source document registry.
- Source chunks and retrieval.
- Scene `source_refs` and equations.
- Automatic memory extraction.
- Source badges in storyboard.

### Phase 3: Make It A Real Animation IDE

Goal: users can edit, iterate, and reuse work professionally.

- Canonical project model.
- Backend autosave/revisions.
- Real timeline tracks.
- Persistent asset library.
- CodeMirror/Monaco editor.
- Gallery backed by `/api/jobs`.
- Voiceover timeline.
- Thumbnail and export presets.

### Phase 4: Public Paid Product

Goal: launch safely with payments and shareability.

- Supabase Auth.
- Credit ledger.
- Stripe reconciliation.
- Private artifact serving.
- Signed share links.
- Rate limits/security headers/CSP.
- Analytics/Sentry/monitoring.
- Admin dashboard.
- Terms/privacy for uploaded learning material.

## Suggested First Implementation Order

1. Disable hosted terminal and hide it behind Developer Mode.
2. Add `/api/render-health` with cached Manim/ffmpeg/TeX/disk checks.
3. Add render queue semaphore and queue status.
4. Replace plan JSON review with storyboard cards.
5. Add source citation fields to plan schema.
6. Add backend source upload registry.
7. Add Supabase-backed project/session persistence.
8. Add authenticated credit ledger.
9. Add persistent gallery/job listing.
10. Upgrade editor/timeline only after project model exists.

## Bottom Line

NorthStar should not try to be “just a Manim wrapper.” The valuable product is an AI-native animation IDE where a creator can drop in notes, LaTeX, PDFs, images, and a prompt, then get a grounded storyboard, editable Manim code, a reliable render, and a shareable output.

The path to production is clear:

- Make the default flow simple.
- Make render success predictable.
- Make source grounding durable.
- Make projects persistent.
- Add auth, billing ledger, and security before public launch.

If those are handled, NorthStar can become a serious creator tool for educational and technical animations rather than a one-off demo.
