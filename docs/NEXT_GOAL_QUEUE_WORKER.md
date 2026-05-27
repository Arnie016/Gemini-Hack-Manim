# NorthStar Queue Worker Goal

- Build queue-backed render worker support.
- Add `NORTHSTAR_RENDER_MODE=inline|queue`, default `inline`.
- Add file-backed durable queue under `work/queue`.
- Add `python -m backend.cli worker` with `--once`, `--poll`, `--concurrency 1`, `--progress`.
- In queue mode, `/api/approve` should enqueue and return quickly.
- Worker should claim jobs, run existing render pipeline, publish artifacts through `backend/artifact_store.py`, and mark done/failed.
- Keep frontend calling `northstarstudio.io` only; never call Lightsail directly.
- Keep inline mode, Stripe, OpenAI-first text flow, and local dev working.
- Add `docs/LIGHTSAIL_WORKER.md` with exact Render env vars and Lightsail systemd commands.
- Use relevant local plugins/skills as guardrails: Build Web Apps, Render, Stripe, Supabase, Codex Security, Playwright/browser if frontend changes.
- Run `py_compile`, CLI help checks, fast science backtest, `git diff --check`, and a queue smoke test if feasible.
- Commit and push if safe.
- Report changed files, verification, Render env vars, Lightsail commands, blockers, and commit hash.
