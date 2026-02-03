# Gemini → Manim Animator (Hackathon MVP)

Generate a short Manim animation from a text prompt. Gemini is central: it plans scenes, generates Manim code, and optionally repairs render failures.

## Quick Start

1. Install Python deps:

```bash
pip install -r requirements.txt
```

2. Set env vars:

```bash
export GEMINI_API_KEY="..."
export GEMINI_MODEL="gemini-3-flash-preview"  # optional
# export MANIM_PY="/path/to/python"            # optional
```

3. Run the server:

```bash
uvicorn backend.main:app --reload --port 8000
```

4. Open:

```
http://localhost:8000
```

## API

`POST /api/animate`

Request:
```json
{ "idea": "Explain gradient descent visually in 20 seconds" }
```

Response (success):
```json
{
  "ok": true,
  "job_id": "20240203-abcdef",
  "video_path": "work/jobs/20240203-abcdef/out.mp4",
  "plan": { ... }
}
```

Response (failure):
```json
{
  "ok": false,
  "job_id": "20240203-abcdef",
  "error": "Render failed",
  "logs": "..."
}
```

## Output Files

Each request writes to:
```
work/jobs/<job_id>/
  scene.py
  plan.json
  out.mp4
  logs.txt
```

## Local Manim Dependencies

Manim Community Edition must be installed locally, along with its system deps. On macOS this typically includes:
- Cairo
- ffmpeg
- (optional) LaTeX if you use MathTex

If Manim is installed in a different Python, set `MANIM_PY` to that interpreter.

## Notes

- The plan uses JSON output mode from Gemini to keep structure tight.
- One repair attempt is made if rendering fails.
- Total duration is capped to 20 seconds by the system prompt.
