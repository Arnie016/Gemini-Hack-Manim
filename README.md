![NorthStar UI](screenshot/UI.png)

# NorthStar

NorthStar is an agentic IDE for Manim storytelling animations. It turns one
prompt into a scene plan, lets you audit/edit the plan, generates Manim code,
renders the animation, and attempts a repair pass when rendering fails:
`Plan -> Audit -> Approve -> Code -> Render -> Repair`.

Built by **Arnav Salkade**.

## Links
- Demo UI: https://skill-deploy-c6ioczee1j-codex-agent-deploys.vercel.app
- Repo: https://github.com/Arnie016/Gemini-Hack-Manim

## Model providers
- OpenAI is the default text provider for planning, code generation, diagnosis, and repair.
- Gemini is still supported as an alternate text provider and for Gemini image generation.
- The IDE stores provider/model settings locally through the Settings panel.

## Run (local backend)
```bash
cd Gemini-Hack-Manim
python3 -m venv .venv && source .venv/bin/activate
brew install cairo pango pkg-config ffmpeg
python -m pip install -U pip -r requirements.txt
export OPENAI_API_KEY="YOUR_KEY"
# Optional: export GEMINI_API_KEY="YOUR_KEY" for Gemini text or image generation.
python -m uvicorn backend.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000` for the landing page or `http://127.0.0.1:8000/app` for the IDE.  
Output files: `work/jobs/<job_id>/`  
Credits: Manim by Grant Sanderson (3Blue1Brown) + Manim Community.

## Devpost tags
`gemini-api` `manim` `education` `edtech` `ai-video` `multimodal-ai` `fastapi` `python` `creator-tools` `scientific-visualization`
