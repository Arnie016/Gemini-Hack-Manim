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
# export GEMINI_IMAGE_MODEL="gemini-2.5-flash-image"  # optional
# export GEMINI_IMAGE_ASPECT="9:16"                   # optional
# export MANIM_PY="/path/to/python"            # optional
```

You can also save these settings from the UI (stored locally in `work/config.json`).

3. Run the server:

```bash
uvicorn backend.main:app --reload --port 8000
```

4. Open:

```
http://localhost:8000
```

The UI includes a three-panel layout (files, canvas, chat), setup, templates, advanced controls, and an editor-style workflow.

## API

`POST /api/animate`

Request:
```json
{
  "idea": "Explain gradient descent visually in 20 seconds",
  "image_prompt": "Scientist walking in a park at sunrise",
  "image_mode": "background",
  "include_images": true,
  "audience": "general",
  "tone": "epic",
  "style": "cinematic",
  "pace": "medium",
  "color_palette": "cool",
  "include_equations": true,
  "include_graphs": true,
  "include_narration": true,
  "target_seconds": 60,
  "max_scenes": 8,
  "max_objects": 6,
  "aspect_ratio": "9:16",
  "quality": "pql",
  "director_brief": "Focus on intuition, show a visual metaphor, end with a recap."
}
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
  assets/
    background.png
    foreground.png
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
- Total duration is not capped; use target_seconds if you want a specific length.
- Image generation is optional and uses the Gemini image model when enabled.

## Advanced Creative Controls

You can customize the output by passing these fields in the request:
- `audience`: general | high school | undergrad | expert
- `tone`: epic | calm | playful | serious
- `style`: cinematic | clean | chalkboard | neon
- `pace`: slow | medium | fast
- `color_palette`: cool | warm | neon | monochrome
- `include_equations`: true/false
- `include_graphs`: true/false
- `include_narration`: true/false
- `target_seconds`: number (optional; no hard cap)
- `max_scenes`: number
- `max_objects`: number
- `aspect_ratio`: 9:16 | 16:9 | 1:1
- `quality`: pql | pqm | pqh
- `director_brief`: extra creative guidance appended to the director brief
- `memory_ids`: array of memory ids to include as context
- `skill_ids`: array of skill ids to include as instructions
- `image_model`: override the image model for this request

## Template Library

Fetch curated templates:
```
GET /api/templates
```
The UI loads these templates and auto-fills the controls.

## Plan → Approve Workflow (Agentic)

Create a plan first:
```
POST /api/plan
{ "idea": "Explain the photoelectric effect", "model": "gemini-3-flash-preview" }
```

Approve (or edit) the plan to render:
```
POST /api/approve
{
  "job_id": "...",
  "plan_text": "{...}",
  "include_images": true,
  "image_prompt": "Scientist in a park",
  "image_mode": "background"
}
```

## Render Edited Code

You can edit the generated code and re-render:

```
POST /api/render-code
{ "code": "...", "quality": "pqm" }
```
## Context Memories

Create and list memories:
```
GET /api/memories
POST /api/memories { "title": "...", "content": "..." }
DELETE /api/memories/{id}
```

## Skill Library

Create and list skills (saved as Markdown under `work/skills/`):
```
GET /api/skills
POST /api/skills { "name": "...", "content": "..." }
DELETE /api/skills/{id}
```

Generate a skill via Gemini:
```
POST /api/skills/generate { "idea": "...", "name": "..." }
```

## Health Checks

```
GET /api/health
```
Returns Manim/ffmpeg status and version output.

## UI Features

- Setup panel (API key, image model, Manim python path)
- Template library with presets
- Editor mode (timeline + inspector)
- Editable code panel with re-render
- Context memories and skills
