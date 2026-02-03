from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gemini_http import GeminiError, generate_content, generate_image
from prompts import (
    MANIM_CODE_SYSTEM,
    REPAIR_SYSTEM,
    SCENE_PLAN_SCHEMA,
    SCENE_PLAN_SYSTEM,
    manim_code_user_prompt,
    scene_plan_user_prompt,
)
from renderer import render_with_manim
from storage import job_paths, new_job_id
from templates import TEMPLATES

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
JOBS = WORK / "jobs"
WEB = ROOT / "web"

WORK.mkdir(parents=True, exist_ok=True)
JOBS.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.mount("/work", StaticFiles(directory=WORK), name="work")


class AnimateReq(BaseModel):
    idea: str
    image_prompt: Optional[str] = None
    image_mode: str = "background"
    include_images: bool = False
    audience: str = "general"
    tone: str = "epic"
    style: str = "cinematic"
    pace: str = "medium"
    color_palette: str = "cool"
    include_equations: bool = True
    include_graphs: bool = True
    include_narration: bool = True
    target_seconds: Optional[float] = None
    max_scenes: Optional[int] = None
    max_objects: Optional[int] = None
    aspect_ratio: str = "9:16"
    quality: str = "pql"
    director_brief: Optional[str] = None


class RenderCodeReq(BaseModel):
    code: str
    quality: str = "pql"


def _build_director_brief(req: AnimateReq) -> str:
    eq_text = "Include 1–2 simple equations." if req.include_equations else "Avoid equations."
    graph_text = "Include at least one simple graph/axis." if req.include_graphs else "Avoid graphs."
    narr_text = (
        "Include short on-screen narration per scene."
        if req.include_narration
        else "Use minimal on-screen text."
    )
    lines = [
        f"Audience: {req.audience}",
        f"Tone: {req.tone}",
        f"Style: {req.style}",
        f"Pace: {req.pace}",
        f"Color palette: {req.color_palette}",
        eq_text,
        graph_text,
        narr_text,
        f"Aspect ratio: {req.aspect_ratio}",
    ]
    if req.target_seconds:
        lines.append(
            f"Target length: ~{req.target_seconds} seconds. Allocate scene seconds to match."
        )
    if req.max_scenes:
        lines.append(f"Max scenes: {req.max_scenes}")
    if req.max_objects:
        lines.append(f"Max objects per scene: {req.max_objects}")
    if req.director_brief:
        lines.append(f"Additional brief: {req.director_brief}")
    return "\n".join(lines)


def _render_settings(req: AnimateReq) -> str:
    ratio = req.aspect_ratio.strip()
    if ratio == "9:16":
        return (
            "Aspect ratio 9:16 (vertical). Set config.pixel_width=1080, "
            "config.pixel_height=1920, config.frame_width=9, config.frame_height=16."
        )
    if ratio == "16:9":
        return (
            "Aspect ratio 16:9 (horizontal). Set config.pixel_width=1920, "
            "config.pixel_height=1080, config.frame_width=16, config.frame_height=9."
        )
    if ratio == "1:1":
        return (
            "Aspect ratio 1:1 (square). Set config.pixel_width=1080, "
            "config.pixel_height=1080, config.frame_width=10, config.frame_height=10."
        )
    return f"Aspect ratio {ratio}. Choose appropriate config.pixel_width/pixel_height."


def _parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/templates")
def templates():
    return {"templates": TEMPLATES}


@app.post("/api/animate")
def animate(req: AnimateReq):
    job_id = new_job_id()
    paths = job_paths(JOBS, job_id)
    paths.job_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = paths.job_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    image_warning: Optional[str] = None

    # 1) Plan
    try:
        plan_text = generate_content(
            scene_plan_user_prompt(req.idea, director_brief=_build_director_brief(req)),
            system_text=SCENE_PLAN_SYSTEM,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": SCENE_PLAN_SCHEMA,
            },
        )
        plan = _parse_json(plan_text)
        paths.plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    except (GeminiError, json.JSONDecodeError) as exc:
        return JSONResponse(
            {
                "ok": False,
                "job_id": job_id,
                "error": f"Plan generation failed: {exc}",
            },
            status_code=400,
        )

    # 2) Optional image generation
    assets_description = ""
    bg_rel = None
    fg_rel = None
    if req.include_images and req.image_prompt and req.image_prompt.strip():
        mode = req.image_mode.lower().strip()
        if mode not in {"background", "foreground", "both"}:
            return JSONResponse(
                {
                    "ok": False,
                    "job_id": job_id,
                    "error": f"Invalid image_mode '{req.image_mode}'",
                },
                status_code=400,
            )

        base_prompt = req.image_prompt.strip()
        try:
            if mode in {"background", "both"}:
                bg_bytes = generate_image(
                    f"Wide background scene, no text, cinematic lighting: {base_prompt}"
                )
                bg_path = assets_dir / "background.png"
                bg_path.write_bytes(bg_bytes)
                bg_rel = "assets/background.png"

            if mode in {"foreground", "both"}:
                fg_bytes = generate_image(
                    f"Single character or prop, centered, clean background, no text: {base_prompt}"
                )
                fg_path = assets_dir / "foreground.png"
                fg_path.write_bytes(fg_bytes)
                fg_rel = "assets/foreground.png"
        except GeminiError as exc:
            image_warning = f"Image generation failed: {exc}"
            bg_rel = None
            fg_rel = None

    if bg_rel:
        assets_description += (
            f"- background: {bg_rel} (full-frame backdrop, low motion, z_index -10)\n"
        )
    if fg_rel:
        assets_description += (
            f"- foreground: {fg_rel} (small prop/character in lower third)\n"
        )

    # 3) Code
    try:
        code = generate_content(
            manim_code_user_prompt(
                plan_text,
                assets_description=assets_description,
                render_settings=_render_settings(req),
            ),
            system_text=MANIM_CODE_SYSTEM,
        )
        paths.scene_path.write_text(code, encoding="utf-8")
    except GeminiError as exc:
        return JSONResponse(
            {
                "ok": False,
                "job_id": job_id,
                "error": f"Code generation failed: {exc}",
                "plan": plan,
            },
            status_code=400,
        )

    # 4) Render
    ok, logs = render_with_manim(paths.scene_path, paths.out_mp4, quality=req.quality)
    paths.logs_path.write_text(logs, encoding="utf-8")

    if not ok:
        # 5) Repair (single attempt)
        repair_user = (
            "The render failed.\n"
            "Here are the logs:\n"
            f"{logs}\n\n"
            "Here is the code:\n"
            f"{code}\n\n"
            "Return a fixed full python file."
        )
        try:
            code2 = generate_content(repair_user, system_text=REPAIR_SYSTEM)
            paths.scene_path.write_text(code2, encoding="utf-8")
            ok, logs = render_with_manim(paths.scene_path, paths.out_mp4)
            paths.logs_path.write_text(logs, encoding="utf-8")
        except GeminiError as exc:
            return JSONResponse(
                {
                    "ok": False,
                    "job_id": job_id,
                    "error": f"Repair failed: {exc}",
                    "logs": logs,
                    "plan": plan,
                },
                status_code=500,
            )

    if not ok:
        return JSONResponse(
            {
                "ok": False,
                "job_id": job_id,
                "error": "Render failed",
                "logs": logs,
                "plan": plan,
            },
            status_code=500,
        )

    response = {
        "ok": True,
        "job_id": job_id,
        "video_path": str(paths.out_mp4.relative_to(ROOT)),
        "plan": plan,
        "code": paths.scene_path.read_text(encoding="utf-8"),
    }
    if image_warning:
        response["image_warning"] = image_warning
    if bg_rel or fg_rel:
        response["assets"] = {"background": bg_rel, "foreground": fg_rel}
    return response


@app.post("/api/render-code")
def render_code(req: RenderCodeReq):
    job_id = new_job_id()
    paths = job_paths(JOBS, job_id)
    paths.job_dir.mkdir(parents=True, exist_ok=True)

    paths.scene_path.write_text(req.code, encoding="utf-8")
    ok, logs = render_with_manim(paths.scene_path, paths.out_mp4, quality=req.quality)
    paths.logs_path.write_text(logs, encoding="utf-8")

    if not ok:
        return JSONResponse(
            {
                "ok": False,
                "job_id": job_id,
                "error": "Render failed",
                "logs": logs,
            },
            status_code=500,
        )

    return {
        "ok": True,
        "job_id": job_id,
        "video_path": str(paths.out_mp4.relative_to(ROOT)),
        "logs": logs,
    }
