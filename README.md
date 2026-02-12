# NorthStar

NorthStar turns one prompt into a rendered Manim explainer with an agentic flow:
`Plan -> Approve -> Code -> Render`.

Built by **Arnav Salkade**.

![NorthStar UI](screenshot/UI.png)

## Links
- Demo UI: https://skill-deploy-c6ioczee1j-codex-agent-deploys.vercel.app
- Repo: https://github.com/Arnie016/Gemini-Hack-Manim

## Why Gemini matters
- Plans scenes (structured JSON)
- Writes Manim code from the approved plan
- Diagnoses render failures and repairs/retries

## Run (local backend)
```bash
cd "/Users/hema/Desktop/Hackathons/Gemini-Hack-Manim"
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip -r requirements.txt
python -m pip install manim
brew install ffmpeg
export GEMINI_API_KEY="YOUR_KEY"
python -m uvicorn backend.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000`  
Output files: `work/jobs/<job_id>/`  
Credits: Manim by Grant Sanderson (3Blue1Brown) + Manim Community.

## Deploy (Render, real API + UI)
1. Push this repo with `Dockerfile` + `render.yaml`.
2. Open: `https://dashboard.render.com/blueprint/new?repo=https://github.com/Arnie016/Gemini-Hack-Manim`
3. Set `GEMINI_API_KEY` in Render env vars.
4. Deploy. Use the Render URL as your full app (not Vercel preview).

## Devpost tags
`gemini-api` `manim` `education` `edtech` `ai-video` `multimodal-ai` `fastapi` `python` `creator-tools` `scientific-visualization`
