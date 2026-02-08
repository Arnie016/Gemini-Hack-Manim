from __future__ import annotations

import base64
import json
import logging
import re
import secrets
import time
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, List
from urllib.parse import parse_qs, urlparse

from fastapi import FastAPI, Response, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from .gemini_http import GeminiError, generate_content, generate_image
from .prompts import (
    MANIM_CODE_SYSTEM,
    REPAIR_SYSTEM,
    SCENE_PLAN_SCHEMA,
    SCENE_PLAN_SYSTEM,
    manim_code_user_prompt,
    scene_plan_user_prompt,
)
from .renderer import concat_videos, cut_video_range, render_with_manim
from .storage import job_paths, new_job_id
from .templates import TEMPLATES
from .code_format import CodeSanitizationError, sanitize_manim_code
from .context_store import (
    add_memory,
    delete_memory,
    get_memories_by_ids,
    get_skills_by_ids,
    list_memories,
    list_skills,
    save_skill,
    delete_skill,
)
from .settings_store import load_settings, update_settings
from .file_store import (
    create_folder,
    delete_path,
    list_tree,
    read_file,
    rename_path,
    write_file,
)
from .terminal_runner import TerminalError, run_terminal_command
from .job_manager import JobManager
from .job_state import JobState, append_event, load_state, write_state

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
JOBS = WORK / "jobs"
WEB = ROOT / "web"

WORK.mkdir(parents=True, exist_ok=True)
JOBS.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.mount("/work", StaticFiles(directory=WORK), name="work")
job_manager = JobManager()
logger = logging.getLogger("northstar.api")


@app.middleware("http")
async def security_headers_middleware(request, call_next):
    request_id = request.headers.get("X-Request-ID") or secrets.token_hex(8)
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


class AnimateReq(BaseModel):
    idea: str
    image_prompt: Optional[str] = None
    image_mode: str = "background"
    include_images: bool = False
    image_variants: int = 1
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
    memory_ids: Optional[list[str]] = None
    skill_ids: Optional[list[str]] = None
    image_model: Optional[str] = None


class RenderCodeReq(BaseModel):
    code: str
    quality: str = "pql"


class TerminalReq(BaseModel):
    command: str


class SettingsReq(BaseModel):
    api_key: Optional[str] = None
    text_model: Optional[str] = None
    image_model: Optional[str] = None
    manim_py: Optional[str] = None
    output_copy_dir: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    elevenlabs_voice_id: Optional[str] = None
    elevenlabs_model_id: Optional[str] = None


class PlanReq(BaseModel):
    idea: str
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
    director_brief: Optional[str] = None
    memory_ids: Optional[list[str]] = None
    skill_ids: Optional[list[str]] = None
    include_images: bool = False
    image_prompt: Optional[str] = None
    model: Optional[str] = None


class ApproveReq(BaseModel):
    job_id: str
    plan_text: str
    image_prompt: Optional[str] = None
    image_mode: str = "background"
    include_images: bool = False
    image_variants: int = 1
    image_model: Optional[str] = None
    quality: str = "pql"
    aspect_ratio: str = "9:16"
    model: Optional[str] = None


class JobDownloadReq(BaseModel):
    job_id: str


class FileReq(BaseModel):
    path: str
    content: Optional[str] = None
    overwrite: bool = False


class RenameReq(BaseModel):
    from_path: str
    to_path: str
    overwrite: bool = False


class ImageGenReq(BaseModel):
    job_id: str
    image_prompt: str
    image_mode: str = "background"
    image_model: Optional[str] = None
    variants: int = 1


class UploadJobAssetReq(BaseModel):
    job_id: str
    filename: str
    content_base64: str
    role: str = "background"


class AppendReq(BaseModel):
    base_job_id: str
    next_job_id: str


class CutRangeReq(BaseModel):
    start_sec: float
    end_sec: float


class VoiceoverReq(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    voice_id: Optional[str] = None
    model_id: Optional[str] = None
    script_text: Optional[str] = None
    include_chat_context: bool = False
    chat_context: Optional[str] = None
    use_gemini_script: bool = True


class SourceIndexReq(BaseModel):
    url: str
    notes: Optional[str] = None
    source_type: str = "auto"  # auto | youtube | web
    model: Optional[str] = None


class ScriptPackReq(BaseModel):
    languages: Optional[List[str]] = None
    model: Optional[str] = None


class GeminiRefineReq(BaseModel):
    idea: str
    director_brief: Optional[str] = None
    image_prompt: Optional[str] = None
    audience: Optional[str] = None
    tone: Optional[str] = None
    style: Optional[str] = None
    pace: Optional[str] = None
    color_palette: Optional[str] = None
    model: Optional[str] = None


class OnboardingReq(BaseModel):
    prompt: Optional[str] = None
    audience: Optional[str] = None
    model: Optional[str] = None
    image_model: Optional[str] = None


class CrazyRunReq(BaseModel):
    idea: str
    variants: int = 3
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
    memory_ids: Optional[list[str]] = None
    skill_ids: Optional[list[str]] = None
    include_images: bool = False
    image_prompt: Optional[str] = None
    image_mode: str = "background"
    image_variants: int = 1
    model: Optional[str] = None
    image_model: Optional[str] = None


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", "")
    if request.url.path.startswith("/api/"):
        return JSONResponse(
            {
                "ok": False,
                "error": "Validation error",
                "details": exc.errors(),
                "request_id": request_id,
            },
            status_code=422,
        )
    return JSONResponse({"detail": exc.errors()}, status_code=422)


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "")
    logger.exception("Unhandled error [%s] %s", request_id, request.url.path, exc_info=exc)
    if request.url.path.startswith("/api/"):
        return JSONResponse(
            {
                "ok": False,
                "error": "Internal server error",
                "request_id": request_id,
            },
            status_code=500,
        )
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


def _extract_youtube_id(url: str) -> Optional[str]:
    try:
        u = urlparse(url)
    except Exception:
        return None
    host = (u.netloc or "").lower()
    path = u.path or ""

    if "youtu.be" in host:
        seg = path.strip("/").split("/")
        return seg[0] if seg and seg[0] else None

    if "youtube.com" in host:
        if path.startswith("/watch"):
            return parse_qs(u.query).get("v", [None])[0]
        if path.startswith("/shorts/") or path.startswith("/embed/"):
            seg = path.strip("/").split("/")
            return seg[1] if len(seg) > 1 else None
    return None


def _normalize_source_kind(url: str, source_type: str) -> tuple[str, Optional[str]]:
    st = (source_type or "auto").strip().lower()
    yt = _extract_youtube_id(url)
    if st == "youtube":
        return "youtube", yt
    if st == "web":
        return "web", None
    # auto
    if yt:
        return "youtube", yt
    return "web", None


def _slug_from_url(url: str, fallback: str = "source") -> str:
    try:
        u = urlparse(url)
        tail = (u.path or "").strip("/").split("/")[-1]
        raw = tail or (u.netloc or "") or fallback
    except Exception:
        raw = fallback
    slug = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    return (slug or fallback)[:48]


def _index_source_with_gemini(
    *,
    url: str,
    notes: str,
    kind: str,
    video_id: Optional[str],
    api_key: Optional[str],
    model: Optional[str],
) -> Dict[str, Any]:
    system = (
        "You index external learning sources for animation planning. "
        "Return STRICT JSON only, no markdown. "
        "Be explicit about assumptions when content is sparse."
    )
    schema = {
        "type": "OBJECT",
        "properties": {
            "title": {"type": "STRING"},
            "summary": {"type": "STRING"},
            "key_points": {"type": "ARRAY", "items": {"type": "STRING"}},
            "prompt_hint": {"type": "STRING"},
        },
        "required": ["title", "summary", "key_points", "prompt_hint"],
    }
    context = (
        f"Source kind: {kind}\n"
        f"URL: {url}\n"
        f"YouTube video id: {video_id or 'n/a'}\n"
        f"User notes/transcript:\n{notes.strip() or '(none provided)'}\n\n"
        "Create:\n"
        "1) A short trustworthy title.\n"
        "2) A 3-6 sentence summary for visual storytelling.\n"
        "3) 4-8 key points in order.\n"
        "4) A prompt_hint (2-3 lines) that improves animation planning prompts."
    )
    text = generate_content(
        context,
        system_text=system,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": schema,
        },
        api_key=api_key,
        model=model,
    )
    return _parse_json(text)


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
        lines.append(f"Additional brief: {str(req.director_brief)[:1200]}")
    if getattr(req, "include_images", False) and getattr(req, "image_prompt", None):
        lines.append(
            f"Include visual assets based on: {str(getattr(req, 'image_prompt'))[:600]}"
        )
    memories = get_memories_by_ids(req.memory_ids)
    if memories:
        lines.append("Context memories to incorporate:")
        for mem in memories:
            lines.append(f"- {mem.get('title')}: {mem.get('content')}")
    skills = get_skills_by_ids(req.skill_ids)
    if skills:
        lines.append("Skill instructions to follow:")
        for skill in skills:
            lines.append(skill.get("content", "").strip())
    return "\n".join(lines)


def _render_settings(req: AnimateReq) -> str:
    return _render_settings_ratio(req.aspect_ratio)


def _parse_aspect_ratio(ratio: str) -> tuple[float, float] | None:
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*[:/]\s*(\d+(?:\.\d+)?)\s*$", str(ratio or ""))
    if not m:
        return None
    w = float(m.group(1))
    h = float(m.group(2))
    if w <= 0 or h <= 0:
        return None
    return (w, h)


def _render_settings_ratio(ratio: str) -> str:
    ratio = str(ratio or "").strip() or "9:16"
    parsed = _parse_aspect_ratio(ratio)
    if not parsed:
        return (
            f"Aspect ratio {ratio}. "
            "Use a ratio like 9:16, 16:9, 4:3, 3:2, 1:1 and set matching "
            "config.pixel_width/pixel_height and frame_width/frame_height."
        )

    w, h = parsed
    short_px = 1080
    if w >= h:
        pixel_height = short_px
        pixel_width = int(round((w / h) * short_px / 2.0) * 2)
        frame_height = 9.0 if abs((w / h) - 1.0) > 0.05 else 10.0
        frame_width = round(frame_height * (w / h), 3)
        orientation = "horizontal"
    else:
        pixel_width = short_px
        pixel_height = int(round((h / w) * short_px / 2.0) * 2)
        frame_width = 9.0 if abs((w / h) - 1.0) > 0.05 else 10.0
        frame_height = round(frame_width * (h / w), 3)
        orientation = "vertical"

    return (
        f"Aspect ratio {ratio} ({orientation}). "
        f"Set config.pixel_width={pixel_width}, config.pixel_height={pixel_height}, "
        f"config.frame_width={frame_width}, config.frame_height={frame_height}."
    )


def _job_files(paths) -> list[str]:
    """Return a best-effort list of job artifact paths (relative to repo root).

    Only includes files that exist to avoid confusing the Explorer UI with
    non-existent placeholders.
    """
    candidates: list[Path] = [
        paths.plan_path,
        paths.scene_path,
        paths.out_mp4,
        paths.logs_path,
        paths.job_dir / "state.json",
        paths.job_dir / "events.log",
        paths.job_dir / "captions.srt",
        paths.job_dir / "export.zip",
    ]
    assets_dir = paths.job_dir / "assets"
    if assets_dir.exists():
        for p in sorted(assets_dir.glob("*")):
            if p.is_file():
                candidates.append(p)

    out: list[str] = []
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                out.append(str(p.relative_to(ROOT)))
        except Exception:
            continue
    # De-dupe while preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    return uniq


def _parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _bounded_text(
    field: str,
    value: Optional[str],
    *,
    max_len: int,
    required: bool = False,
    truncate: bool = False,
) -> str:
    txt = (value or "").strip()
    if required and not txt:
        raise ValueError(f"{field} is required")
    if len(txt) > max_len:
        if truncate:
            return txt[:max_len]
        raise ValueError(f"{field} is too long (max {max_len} chars)")
    return txt


def _policy_block_reason(*texts: Optional[str]) -> Optional[str]:
    merged = " ".join([(t or "") for t in texts]).lower()
    if not merged.strip():
        return None
    blocked = [
        ("child sexual", "Requests involving sexual content with minors are blocked."),
        ("child porn", "Requests involving sexual content with minors are blocked."),
        ("rape", "Sexual violence content is not supported."),
        ("bestiality", "Explicit abusive sexual content is not supported."),
        ("gore", "Graphic violence/gore content is not supported for this app."),
        ("beheading", "Graphic violence/gore content is not supported for this app."),
        ("terrorist manifesto", "Extremist propaganda content is not supported."),
    ]
    for token, msg in blocked:
        if token in merged:
            return msg
    return None


def _default_onboarding_steps() -> list[Dict[str, str]]:
    return [
        {
            "id": "template",
            "target": "#templateSelect",
            "title": "Choose a starter template",
            "body": "Pick a template in Explorer so the scene style, pacing, and defaults load fast.",
            "hint": "Left panel -> Template",
            "icon_prompt": (
                "Minimal line icon for choosing a storyboard template, dark matte background, "
                "neon cyan accents, clean UI style"
            ),
        },
        {
            "id": "plan",
            "target": "#chatInput",
            "title": "Describe the idea",
            "body": "Type the concept in one line. Gemini will turn it into a scene-by-scene plan.",
            "hint": "Right panel -> Prompt box",
            "icon_prompt": (
                "Minimal line icon of a prompt box with spark cursor, dark matte background, "
                "electric blue and teal accent"
            ),
        },
        {
            "id": "images",
            "target": "#imageGenDetails",
            "title": "Generate image assets",
            "body": "Use Nano Banana to create background and foreground variants, then drag into scene cards.",
            "hint": "Middle panel -> Image generation",
            "icon_prompt": (
                "Minimal icon showing image variants and drag and drop to timeline card, "
                "dark matte background, subtle green accent"
            ),
        },
        {
            "id": "timeline",
            "target": "#timelineTrack",
            "title": "Refine scenes on the timeline",
            "body": "Edit each scene focus and duration. Keep one clear concept per scene for readability.",
            "hint": "Middle panel -> Timeline",
            "icon_prompt": (
                "Minimal timeline icon with labeled scene blocks and edit handles, "
                "cinematic dark UI, high contrast"
            ),
        },
        {
            "id": "preview",
            "target": "#previewSlot",
            "title": "Approve and render",
            "body": "Run plan -> code -> render. If a render fails, Gemini diagnoses and retries automatically.",
            "hint": "Middle panel -> Preview area",
            "icon_prompt": (
                "Minimal icon for render pipeline plan code render with play symbol, "
                "dark matte background, blue green glow"
            ),
        },
        {
            "id": "steps",
            "target": "#agentSteps",
            "title": "Track every phase",
            "body": "Watch Plan, Approve, Code, and Render status. Open details to inspect each phase output.",
            "hint": "Right panel -> Phase tracker",
            "icon_prompt": (
                "Minimal icon showing four progress stages with diagnostics panel, "
                "dark interface, subtle neon highlights"
            ),
        },
    ]


def _normalize_onboarding_steps(raw_steps: Any) -> list[Dict[str, str]]:
    defaults = _default_onboarding_steps()
    if not isinstance(raw_steps, list):
        return defaults

    allowed_targets = {step["target"] for step in defaults}
    out: list[Dict[str, str]] = []
    for idx, default in enumerate(defaults):
        src = raw_steps[idx] if idx < len(raw_steps) else {}
        src = src if isinstance(src, dict) else {}
        merged = dict(default)

        title = str(src.get("title") or "").strip()
        body = str(src.get("body") or "").strip()
        hint = str(src.get("hint") or "").strip()
        icon_prompt = str(src.get("icon_prompt") or "").strip()
        target = str(src.get("target") or "").strip()

        if title:
            merged["title"] = title[:140]
        if body:
            merged["body"] = body[:360]
        if hint:
            merged["hint"] = hint[:140]
        if icon_prompt:
            merged["icon_prompt"] = icon_prompt[:320]
        if target in allowed_targets:
            merged["target"] = target
        out.append(merged)
    return out


def _merge_plans(base_plan: Dict[str, Any], next_plan: Dict[str, Any]) -> Dict[str, Any]:
    base_scenes = list(base_plan.get("scenes") or [])
    next_scenes = list(next_plan.get("scenes") or [])
    merged_scenes = base_scenes + next_scenes
    title_a = str(base_plan.get("title") or "").strip()
    title_b = str(next_plan.get("title") or "").strip()
    if title_a and title_b and title_a != title_b:
        merged_title = f"{title_a} + {title_b}"
    else:
        merged_title = title_a or title_b or "Combined Story"
    total = 0.0
    for sc in merged_scenes:
        try:
            total += float(sc.get("seconds") or 0)
        except Exception:
            continue
    return {
        "title": merged_title,
        "total_seconds": round(total, 2),
        "scenes": merged_scenes,
    }


def _cut_plan_range(plan: Dict[str, Any], *, start_sec: float, end_sec: float) -> Dict[str, Any]:
    """Remove [start_sec, end_sec] from plan timeline and keep scenes coherent."""
    src_scenes = list(plan.get("scenes") or [])
    if not src_scenes:
        return plan
    start_sec = max(0.0, float(start_sec))
    end_sec = max(start_sec, float(end_sec))

    out_scenes: list[Dict[str, Any]] = []
    cursor = 0.0
    for sc in src_scenes:
        dur = max(1.0, float(sc.get("seconds") or 1.0))
        seg_start = cursor
        seg_end = cursor + dur
        cursor = seg_end

        # No overlap with cut range.
        if seg_end <= start_sec or seg_start >= end_sec:
            out_scenes.append(dict(sc))
            continue

        # Fully removed scene.
        if seg_start >= start_sec and seg_end <= end_sec:
            continue

        # Partial overlap; keep remaining duration.
        removed = max(0.0, min(seg_end, end_sec) - max(seg_start, start_sec))
        remain = max(0.5, dur - removed)
        keep = dict(sc)
        keep["seconds"] = round(remain, 2)
        out_scenes.append(keep)

    if not out_scenes:
        out_scenes = [
            {
                "seconds": 3,
                "goal": "Quick outro",
                "elements": [],
                "actions": [],
                "narration": "Quick outro after cut.",
                "assets": {},
            }
        ]

    total = 0.0
    for sc in out_scenes:
        try:
            total += float(sc.get("seconds") or 0.0)
        except Exception:
            continue
    merged = dict(plan)
    merged["scenes"] = out_scenes
    merged["total_seconds"] = round(total, 2)
    return merged


def _resolve_output_copy_dir(raw: Optional[str]) -> Optional[Path]:
    """Resolve optional output-copy destination under host filesystem.

    This is intentionally permissive because users may want Desktop or external
    folders, but we still normalize and reject empty inputs.
    """
    txt = (raw or "").strip()
    if not txt:
        return None
    p = Path(txt).expanduser()
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def _copy_output_video(job_id: str, *, output_copy_dir: Optional[str]) -> tuple[bool, str]:
    paths = job_paths(JOBS, job_id)
    if not paths.out_mp4.exists():
        return False, "No rendered video found for this job."
    out_dir = _resolve_output_copy_dir(output_copy_dir)
    if out_dir is None:
        return False, "No output copy directory configured."
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = paths.out_mp4.suffix or ".mp4"
    target = out_dir / f"{job_id}{ext}"
    shutil.copy2(paths.out_mp4, target)
    return True, str(target)


def _voiceover_text_from_plan(paths) -> str:
    if not paths.plan_path.exists():
        return ""
    try:
        plan = json.loads(paths.plan_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    scenes = list(plan.get("scenes") or [])
    lines: list[str] = []
    for sc in scenes:
        n = str(sc.get("narration") or "").strip()
        if n:
            lines.append(n)
    return " ".join(lines).strip()


def _srt_to_plain_text(srt_path: Path) -> str:
    if not srt_path.exists():
        return ""
    text = srt_path.read_text(encoding="utf-8", errors="ignore")
    out: list[str] = []
    for ln in text.splitlines():
        line = ln.strip()
        if not line:
            continue
        if line.isdigit():
            continue
        if "-->" in line:
            continue
        out.append(line)
    return " ".join(out).strip()


def _scene_ranges_from_plan(plan: Dict[str, Any]) -> list[tuple[float, float]]:
    scenes = list(plan.get("scenes") or [])
    cursor = 0.0
    out: list[tuple[float, float]] = []
    for sc in scenes:
        sec = max(0.5, float(sc.get("seconds") or 0.5))
        start = cursor
        end = cursor + sec
        out.append((start, end))
        cursor = end
    return out


def _fmt_srt_ts(ts: float) -> str:
    ts = max(0.0, float(ts))
    h = int(ts // 3600)
    m = int((ts % 3600) // 60)
    s = int(ts % 60)
    ms = int(round((ts - int(ts)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _build_srt_from_lines(plan: Dict[str, Any], lines: list[str]) -> str:
    spans = _scene_ranges_from_plan(plan)
    out: list[str] = []
    idx = 1
    for i, (start, end) in enumerate(spans):
        text = (lines[i] if i < len(lines) else "").strip()
        if not text:
            continue
        out.append(str(idx))
        out.append(f"{_fmt_srt_ts(start)} --> {_fmt_srt_ts(end)}")
        out.append(text)
        out.append("")
        idx += 1
    return "\n".join(out).strip() + "\n"


def _language_label(code: str) -> str:
    labels = {
        "en": "English",
        "hi": "Hindi",
        "es": "Spanish",
        "pt": "Portuguese",
        "fr": "French",
        "de": "German",
        "ja": "Japanese",
    }
    c = (code or "").strip().lower()
    return labels.get(c, c or "Unknown")


def _multilingual_script_packs(
    *,
    plan: Dict[str, Any],
    languages: list[str],
    api_key: Optional[str],
    model: Optional[str],
) -> dict[str, Any]:
    scenes = list(plan.get("scenes") or [])
    base_lines = [str(sc.get("narration") or "").strip() for sc in scenes]
    if not base_lines:
        base_lines = [str(sc.get("goal") or "").strip() for sc in scenes]

    langs = []
    for code in languages:
        c = (code or "").strip().lower()
        if c and c not in langs:
            langs.append(c)
    if not langs:
        langs = ["en", "hi", "es"]

    schema = {
        "type": "OBJECT",
        "properties": {
            "packs": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "language": {"type": "STRING"},
                        "scene_lines": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "voice_script": {"type": "STRING"},
                    },
                    "required": ["language", "scene_lines", "voice_script"],
                },
            }
        },
        "required": ["packs"],
    }
    user = (
        "Build multilingual narration packs for this scene plan.\n"
        f"Languages: {', '.join(langs)}\n"
        "Rules:\n"
        "- Keep one narration line per scene.\n"
        "- Keep style natural and engaging, not robotic.\n"
        "- Preserve scientific meaning.\n\n"
        f"Plan JSON:\n{json.dumps(plan)[:9000]}\n"
    )
    try:
        raw = generate_content(
            user,
            system_text=(
                "You are a multilingual science narrator. "
                "Return strict JSON only."
            ),
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": schema,
            },
            api_key=api_key,
            model=model,
        )
        parsed = _parse_json(raw)
        packs = list(parsed.get("packs") or [])
    except Exception:
        packs = []

    out_packs: list[dict[str, Any]] = []
    by_lang: dict[str, dict[str, Any]] = {}
    for p in packs:
        code = str(p.get("language") or "").strip().lower()
        if not code:
            continue
        by_lang[code] = {
            "language": code,
            "scene_lines": [str(x).strip() for x in (p.get("scene_lines") or [])],
            "voice_script": str(p.get("voice_script") or "").strip(),
        }

    for code in langs:
        item = by_lang.get(code, {})
        lines = item.get("scene_lines") or []
        if len(lines) < len(base_lines):
            lines = lines + base_lines[len(lines):]
        if len(lines) > len(base_lines):
            lines = lines[: len(base_lines)]
        script = item.get("voice_script") or " ".join([x for x in lines if x]).strip()
        out_packs.append(
            {
                "language": code,
                "language_label": _language_label(code),
                "scene_lines": lines,
                "voice_script": script,
                "captions_srt": _build_srt_from_lines(plan, lines),
            }
        )
    return {"packs": out_packs}


def _voiceover_script_with_gemini(
    *,
    paths,
    api_key: str,
    model: Optional[str],
    chat_context: str,
) -> str:
    plan_text = ""
    total_seconds = 0.0
    if paths.plan_path.exists():
        plan_text = paths.plan_path.read_text(encoding="utf-8", errors="ignore")
        try:
            plan_obj = json.loads(plan_text)
            total_seconds = float(plan_obj.get("total_seconds") or 0)
        except Exception:
            total_seconds = 0.0
    if total_seconds <= 0:
        total_seconds = 60.0
    target_words = int(max(80, min(420, total_seconds * 2.4)))

    system = (
        "You are a narration writer for short science explainer videos. "
        "Write in a natural, engaging voice, concise sentences, no stage directions, no bullet points. "
        "Avoid repeating equations verbatim; explain them in plain language. "
        "Return only the script."
    )
    user = (
        f"Target length: about {target_words} words for ~{int(total_seconds)} seconds.\n\n"
        f"Scene plan JSON:\n{plan_text[:4000]}\n\n"
    )
    if chat_context:
        user += f"Context cues:\n{chat_context[:1200]}\n\n"
    user += "Write the final voiceover script now."

    try:
        script = generate_content(user, system_text=system, api_key=api_key, model=model)
    except Exception:
        script = ""
    return (script or "").strip()

def _add_elevenlabs_voiceover(*, paths, api_key: str, voice_id: str, model_id: Optional[str], text: str) -> tuple[bool, str]:
    import requests

    if not paths.out_mp4.exists():
        return False, "Render an MP4 first before adding voiceover."
    payload_text = (text or "").strip()
    if not payload_text:
        return False, "No narration text found. Add captions or scene narration first."

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    body = {
        "text": payload_text,
        "model_id": (model_id or "eleven_multilingual_v2"),
    }
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=120)
        if resp.status_code >= 400:
            return False, f"ElevenLabs error {resp.status_code}: {resp.text[:500]}"
    except Exception as exc:
        return False, f"ElevenLabs request failed: {exc}"

    audio_mp3 = paths.job_dir / "voiceover.mp3"
    audio_mp3.write_bytes(resp.content)

    merged_tmp = paths.job_dir / "out-voiceover.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(paths.out_mp4),
        "-i",
        str(audio_mp3),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(merged_tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not merged_tmp.exists():
        return False, ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()[-4000:]

    merged_tmp.replace(paths.out_mp4)
    return True, str(paths.out_mp4.relative_to(ROOT))


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/favicon.ico")
def favicon():
    # Browser requests this by default; avoid noisy 404 logs during demos.
    return Response(status_code=204)


def _generate_assets(
    *,
    job_dir: Path,
    image_prompt: str,
    image_mode: str,
    api_key: Optional[str],
    image_model: Optional[str],
    variants: int = 1,
) -> tuple[list[str], list[str], Optional[str], str]:
    """Generate optional background/foreground assets in job_dir/assets.

    Returns: (bg_rel, fg_rel, warning, assets_description)
    """
    assets_dir = job_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    mode = (image_mode or "background").lower().strip()
    if mode not in {"background", "foreground", "both"}:
        raise ValueError(f"Invalid image_mode '{image_mode}'")

    base_prompt = (image_prompt or "").strip()
    if not base_prompt:
        return None, None, None, ""

    bg_rel: list[str] = []
    fg_rel: list[str] = []
    warning: Optional[str] = None
    try:
        variants = max(1, int(variants or 1))
        if mode in {"background", "both"}:
            for i in range(1, variants + 1):
                bg_bytes = generate_image(
                    (
                        f"Variation {i}/{variants}. "
                        "Generate a cinematic full-frame background plate for an educational explainer. "
                        "Keep center-safe composition, clean contrast, and low clutter. "
                        f"Prompt: {base_prompt}"
                    ),
                    api_key=api_key,
                    model=image_model,
                )
                name = "background.png" if variants == 1 else f"background-{i}.png"
                (assets_dir / name).write_bytes(bg_bytes)
                bg_rel.append(f"assets/{name}")

        if mode in {"foreground", "both"}:
            for i in range(1, variants + 1):
                fg_bytes = generate_image(
                    (
                        f"Variation {i}/{variants}. "
                        "Generate a foreground explainer asset (avatar, prop, card, or diagram) with crisp edges and high readability. "
                        "Allow typography when requested by the prompt (equation/TDX cards). "
                        f"Prompt: {base_prompt}"
                    ),
                    api_key=api_key,
                    model=image_model,
                )
                name = "foreground.png" if variants == 1 else f"foreground-{i}.png"
                (assets_dir / name).write_bytes(fg_bytes)
                fg_rel.append(f"assets/{name}")
    except GeminiError as exc:
        warning = f"Image generation failed: {exc}"
        bg_rel = []
        fg_rel = []

    desc = ""
    if bg_rel:
        desc += f"- background: {bg_rel[0]} (full-frame backdrop, low motion, z_index -10)\n"
    if fg_rel:
        desc += f"- foreground: {fg_rel[0]} (small prop/character in lower third)\n"
    return bg_rel, fg_rel, warning, desc


@app.get("/api/health")
def health():
    return _health_snapshot(load_settings())


def _health_snapshot(settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    import subprocess
    import shutil

    def _run(cmd: list[str], *, timeout_s: float = 10) -> tuple[bool, str]:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
            ok = proc.returncode == 0
            out = (proc.stdout or "") + (proc.stderr or "")
            return ok, out.strip()
        except Exception as exc:
            return False, str(exc)

    settings = settings or load_settings()
    manim_py_setting = settings.get("manim_py")
    candidates: list[str] = []
    if manim_py_setting:
        candidates.append(manim_py_setting)
    venv_py = ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        candidates.append(str(venv_py))
    candidates.extend(["python3", "python"])
    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    def _exists(cmd: str) -> bool:
        # Filter out non-existent executables so the UI doesn't show confusing
        # "[Errno 2] No such file or directory: 'python'".
        try:
            p = Path(cmd)
            if p.is_absolute() or "/" in cmd:
                return p.exists()
        except Exception:
            pass
        return shutil.which(cmd) is not None

    candidates = [c for c in candidates if _exists(c)]

    manim_ok = False
    manim_out = ""
    used_py = candidates[0] if candidates else (manim_py_setting or "python3")
    for cand in candidates:
        ok, out = _run([cand, "-m", "manim", "--version"])
        if ok:
            manim_ok = True
            manim_out = out
            used_py = cand
            break
        if not manim_out:
            manim_out = out

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        # Some builds are surprisingly slow to print version info.
        ffmpeg_ok, ffmpeg_out = _run([ffmpeg_path, "-version"], timeout_s=20)
    else:
        ffmpeg_ok, ffmpeg_out = False, "ffmpeg not found on PATH"
    eleven_api = (settings.get("elevenlabs_api_key") or "").strip()
    eleven_voice = (settings.get("elevenlabs_voice_id") or "").strip()
    eleven_model = (settings.get("elevenlabs_model_id") or "").strip() or "eleven_multilingual_v2"

    return {
        "manim_ok": manim_ok,
        "manim_version": manim_out.splitlines()[0] if manim_out else "",
        "ffmpeg_ok": ffmpeg_ok,
        "ffmpeg_version": ffmpeg_out.splitlines()[0] if ffmpeg_out else "",
        "ffmpeg_path": ffmpeg_path or "",
        "manim_py": used_py or manim_py_setting or "python3",
        "python_candidates": candidates,
        "elevenlabs_ready": bool(eleven_api and eleven_voice),
        "elevenlabs_voice_id": eleven_voice,
        "elevenlabs_model_id": eleven_model,
    }


def _output_path_writable() -> tuple[bool, str]:
    probe = JOBS / f".preflight-write-{secrets.token_hex(4)}.tmp"
    try:
        JOBS.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True, ""
    except Exception as exc:
        try:
            probe.unlink(missing_ok=True)
        except Exception:
            pass
        return False, str(exc)


def _preflight_payload(settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    settings = settings or load_settings()
    health_data = _health_snapshot(settings)
    api_ok = bool((settings.get("api_key") or "").strip())
    write_ok, write_error = _output_path_writable()
    checks = {
        "api_key": api_ok,
        "manim": bool(health_data.get("manim_ok")),
        "ffmpeg": bool(health_data.get("ffmpeg_ok")),
        "output_writable": write_ok,
    }
    missing = [name for name, ok in checks.items() if not ok]
    fix_action = ""
    if not checks["api_key"]:
        fix_action = "open_settings_api"
    elif not checks["manim"] or not checks["ffmpeg"]:
        fix_action = "open_settings_render_get_started"
    elif not checks["output_writable"]:
        fix_action = "check_output_permissions"
    return {
        "ok": not missing,
        "checks": checks,
        "missing": missing,
        "output_root": str(JOBS),
        "write_error": write_error,
        "fix_action": fix_action,
        "health": health_data,
    }


@app.get("/api/preflight")
def preflight():
    return _preflight_payload(load_settings())


@app.get("/api/settings")
def get_settings():
    settings = load_settings()
    return {
        "has_api_key": bool(settings.get("api_key")),
        "text_model": settings.get("text_model"),
        "image_model": settings.get("image_model"),
        "manim_py": settings.get("manim_py"),
        "output_copy_dir": settings.get("output_copy_dir") or "",
        "has_elevenlabs_key": bool(settings.get("elevenlabs_api_key")),
        "elevenlabs_voice_id": settings.get("elevenlabs_voice_id") or "",
        "elevenlabs_model_id": settings.get("elevenlabs_model_id") or "",
        "project_root": str(ROOT),
        "work_root": str(WORK),
    }


@app.post("/api/settings")
def set_settings(req: SettingsReq):
    settings = update_settings(
        {
            "api_key": req.api_key,
            "text_model": req.text_model,
            "image_model": req.image_model,
            "manim_py": req.manim_py,
            "output_copy_dir": req.output_copy_dir,
            "elevenlabs_api_key": req.elevenlabs_api_key,
            "elevenlabs_voice_id": req.elevenlabs_voice_id,
            "elevenlabs_model_id": req.elevenlabs_model_id,
        }
    )
    return {
        "ok": True,
        "has_api_key": bool(settings.get("api_key")),
        "text_model": settings.get("text_model"),
        "image_model": settings.get("image_model"),
        "manim_py": settings.get("manim_py"),
        "output_copy_dir": settings.get("output_copy_dir") or "",
        "has_elevenlabs_key": bool(settings.get("elevenlabs_api_key")),
        "elevenlabs_voice_id": settings.get("elevenlabs_voice_id") or "",
        "elevenlabs_model_id": settings.get("elevenlabs_model_id") or "",
    }


@app.post("/api/terminal/run")
def terminal_run(req: TerminalReq):
    settings = load_settings()
    try:
        out = run_terminal_command(req.command, manim_py=settings.get("manim_py") or "python3")
        return {"ok": True, "output": out}
    except TerminalError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.get("/api/files")
def list_files():
    return {"tree": list_tree()}


@app.post("/api/files/folder")
def create_folder_api(req: FileReq):
    try:
        path = create_folder(req.path)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    return {"ok": True, "path": path}


@app.post("/api/files/file")
def create_file_api(req: FileReq):
    try:
        path = write_file(req.path, req.content or "", overwrite=req.overwrite)
    except FileExistsError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    return {"ok": True, "path": path}


@app.put("/api/files/file")
def update_file_api(req: FileReq):
    try:
        path = write_file(req.path, req.content or "", overwrite=True)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    return {"ok": True, "path": path}


@app.get("/api/files/file")
def read_file_api(path: str):
    try:
        content = read_file(path)
    except FileNotFoundError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    return {"ok": True, "path": path, "content": content}


@app.delete("/api/files/file")
def delete_file_api(path: str):
    try:
        delete_path(path)
    except FileNotFoundError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    return {"ok": True}


@app.post("/api/files/rename")
def rename_file_api(req: RenameReq):
    try:
        res = rename_path(req.from_path, req.to_path, overwrite=req.overwrite)
    except FileNotFoundError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
    except FileExistsError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    return {"ok": True, "rename": res}


@app.post("/api/images/generate")
def generate_images(req: ImageGenReq):
    paths = job_paths(JOBS, req.job_id)
    if not paths.job_dir.exists():
        return JSONResponse(
            {"ok": False, "job_id": req.job_id, "error": "Unknown job_id"},
            status_code=404,
        )

    try:
        image_prompt = _bounded_text("image_prompt", req.image_prompt, max_len=1200, required=True)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    reason = _policy_block_reason(image_prompt)
    if reason:
        return JSONResponse({"ok": False, "error": reason}, status_code=400)

    settings = load_settings()
    api_key = settings.get("api_key")
    image_model = req.image_model or settings.get("image_model")
    try:
        bg_rel, fg_rel, warning, desc = _generate_assets(
            job_dir=paths.job_dir,
            image_prompt=image_prompt,
            image_mode=req.image_mode,
            api_key=api_key,
            image_model=image_model,
            variants=req.variants,
        )
    except ValueError as exc:
        return JSONResponse(
            {"ok": False, "job_id": req.job_id, "error": str(exc)}, status_code=400
        )

    resp: Dict[str, Any] = {
        "ok": True,
        "job_id": req.job_id,
        "assets": {
            "background": bg_rel[0] if bg_rel else None,
            "foreground": fg_rel[0] if fg_rel else None,
            "backgrounds": bg_rel,
            "foregrounds": fg_rel,
        },
        "assets_description": desc,
    }
    if warning:
        resp["warning"] = warning
    return resp


@app.post("/api/docs/index")
def docs_index(req: SourceIndexReq):
    try:
        url = _bounded_text("url", req.url, max_len=1200, required=True)
        notes = _bounded_text("notes", req.notes, max_len=30000, required=False)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    if not url or not (url.startswith("http://") or url.startswith("https://")):
        return JSONResponse(
            {"ok": False, "error": "Please provide a valid http(s) URL."},
            status_code=400,
        )
    kind, video_id = _normalize_source_kind(url, req.source_type)
    settings = load_settings()
    api_key = settings.get("api_key")
    text_model = req.model or settings.get("text_model")

    title = ""
    summary = ""
    key_points: list[str] = []
    prompt_hint = ""
    warning: Optional[str] = None
    try:
        indexed = _index_source_with_gemini(
            url=url,
            notes=notes,
            kind=kind,
            video_id=video_id,
            api_key=api_key,
            model=text_model,
        )
        title = str(indexed.get("title", "")).strip()
        summary = str(indexed.get("summary", "")).strip()
        key_points = [str(x).strip() for x in indexed.get("key_points", []) if str(x).strip()]
        prompt_hint = str(indexed.get("prompt_hint", "")).strip()
    except Exception as exc:
        # Safe fallback keeps the UX moving even if model call fails.
        warning = f"Indexing fallback used: {exc}"
        if kind == "youtube":
            title = f"YouTube source ({video_id or 'video'})"
        else:
            title = f"Web source: {_slug_from_url(url, 'article')}"
        summary = notes[:500] if notes else (
            "Source indexed without transcript. Add notes/transcript for stronger summaries."
        )
        key_points = [summary] if summary else []
        prompt_hint = "Focus on one concept per scene and verify assumptions from the source."

    if not title:
        title = f"{kind.title()} source"
    if not summary:
        summary = "No summary available. Add transcript/notes and re-index."
    if not key_points:
        key_points = [summary]

    source_id = f"src-{int(time.time())}-{secrets.token_hex(2)}"
    slug = _slug_from_url(url, "source")
    file_suggested_path = f"notes/sources/{slug}-{source_id[-4:]}.md"
    md_lines = [
        f"# {title}",
        "",
        f"- Source kind: {kind}",
        f"- URL: {url}",
    ]
    if video_id:
        md_lines.append(f"- YouTube ID: {video_id}")
    md_lines.extend(
        [
            "",
            "## Summary",
            summary,
            "",
            "## Key Points",
        ]
    )
    md_lines.extend([f"- {p}" for p in key_points])
    md_lines.extend(
        [
            "",
            "## Prompt Hint",
            prompt_hint or "Use this source to tighten storyboard sequencing.",
        ]
    )
    if notes:
        md_lines.extend(["", "## Notes / Transcript (user-provided)", notes[:4000]])
    markdown = "\n".join(md_lines).strip() + "\n"

    resp: Dict[str, Any] = {
        "ok": True,
        "indexed": {
            "id": source_id,
            "kind": kind,
            "url": url,
            "video_id": video_id,
            "title": title,
            "summary": summary,
            "key_points": key_points[:12],
            "prompt_hint": prompt_hint,
            "file_suggested_path": file_suggested_path,
            "markdown": markdown,
        },
    }
    if warning:
        resp["warning"] = warning
    return resp


@app.post("/api/jobs/upload-asset")
def upload_job_asset(req: UploadJobAssetReq):
    paths = job_paths(JOBS, req.job_id)
    if not paths.job_dir.exists():
        return JSONResponse(
            {"ok": False, "job_id": req.job_id, "error": "Unknown job_id"},
            status_code=404,
        )

    role = (req.role or "background").strip().lower()
    if role not in {"background", "foreground"}:
        return JSONResponse(
            {"ok": False, "error": "role must be background or foreground"},
            status_code=400,
        )

    raw = (req.content_base64 or "").strip()
    if not raw:
        return JSONResponse({"ok": False, "error": "content_base64 is required"}, status_code=400)
    if "," in raw and raw.split(",", 1)[0].lower().startswith("data:"):
        raw = raw.split(",", 1)[1]

    try:
        blob = base64.b64decode(raw, validate=True)
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid base64 image content"}, status_code=400)

    if not blob:
        return JSONResponse({"ok": False, "error": "Decoded file is empty"}, status_code=400)
    if len(blob) > 10 * 1024 * 1024:
        return JSONResponse({"ok": False, "error": "File is too large (max 10MB)"}, status_code=413)

    suffix = Path(req.filename or "").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".png"
    assets_dir = paths.job_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    name = f"{role}-{int(time.time() * 1000)}{suffix}"
    out = assets_dir / name
    out.write_bytes(blob)
    rel = f"assets/{name}"
    return {"ok": True, "job_id": req.job_id, "asset_path": rel, "role": role}


@app.get("/api/memories")
def get_memories():
    return {"memories": list_memories()}


@app.post("/api/memories")
def create_memory(payload: Dict[str, Any]):
    title = str(payload.get("title", "")).strip()
    content = str(payload.get("content", "")).strip()
    if not title or not content:
        return JSONResponse({"ok": False, "error": "Title and content required"}, status_code=400)
    entry = add_memory(title, content)
    return {"ok": True, "memory": entry}


@app.delete("/api/memories/{memory_id}")
def remove_memory(memory_id: str):
    ok = delete_memory(memory_id)
    return {"ok": ok}


@app.get("/api/skills")
def get_skills():
    return {"skills": list_skills()}


@app.post("/api/skills")
def create_skill(payload: Dict[str, Any]):
    name = str(payload.get("name", "")).strip()
    content = str(payload.get("content", "")).strip()
    if not name:
        return JSONResponse({"ok": False, "error": "Name required"}, status_code=400)
    skill = save_skill(name, content)
    return {"ok": True, "skill": skill}


@app.delete("/api/skills/{skill_id}")
def remove_skill(skill_id: str):
    ok = delete_skill(skill_id)
    return {"ok": ok}


@app.post("/api/skills/generate")
def generate_skill(payload: Dict[str, Any]):
    idea = str(payload.get("idea", "")).strip()
    name = str(payload.get("name", "")).strip() or idea[:40]
    if not idea:
        return JSONResponse({"ok": False, "error": "Idea required"}, status_code=400)
    settings = load_settings()
    api_key = settings.get("api_key")
    system = (
        "You write concise Markdown instructions for a custom skill. "
        "Return ONLY Markdown. Start with a short title line."
    )
    try:
        text = generate_content(idea, system_text=system, api_key=api_key)
        skill = save_skill(name, text)
        return {"ok": True, "skill": skill}
    except GeminiError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.get("/api/templates")
def templates():
    return {"templates": TEMPLATES}


@app.post("/api/gemini/refine")
def gemini_refine(req: GeminiRefineReq):
    try:
        idea = _bounded_text("idea", req.idea, max_len=6000, required=True)
        director_brief = _bounded_text(
            "director_brief",
            req.director_brief,
            max_len=24000,
            required=False,
            truncate=True,
        )
        image_prompt = _bounded_text("image_prompt", req.image_prompt, max_len=1600, required=False)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    reason = _policy_block_reason(idea, director_brief, image_prompt)
    if reason:
        return JSONResponse({"ok": False, "error": reason}, status_code=400)

    settings = load_settings()
    api_key = settings.get("api_key")
    text_model = req.model or settings.get("text_model")

    schema = {
        "type": "OBJECT",
        "properties": {
            "refined_idea": {"type": "STRING"},
            "refined_director_brief": {"type": "STRING"},
            "refined_image_prompt": {"type": "STRING"},
            "checklist": {"type": "ARRAY", "items": {"type": "STRING"}},
        },
        "required": ["refined_idea", "refined_director_brief", "refined_image_prompt", "checklist"],
    }
    user = (
        "Refine this creator prompt for a Manim explainer pipeline.\n"
        "Keep scientific meaning intact, improve clarity and structure, remove redundancy.\n"
        "Do not over-lengthen text.\n\n"
        f"Idea:\n{idea}\n\n"
        f"Audience: {req.audience or 'general'}\n"
        f"Tone: {req.tone or 'epic'}\n"
        f"Style: {req.style or 'cinematic'}\n"
        f"Pace: {req.pace or 'medium'}\n"
        f"Color palette: {req.color_palette or 'cool'}\n\n"
        f"Director brief:\n{director_brief or '(none)'}\n\n"
        f"Image prompt:\n{image_prompt or '(none)'}\n\n"
        "Return strict JSON only. "
        "Checklist should be 4-8 concise bullets for creator review."
    )
    try:
        out = generate_content(
            user,
            system_text=(
                "You are a senior creative director for science explainers. "
                "Tighten prompts for planning, coding, and rendering quality. "
                "Keep language concise and actionable."
            ),
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": schema,
            },
            api_key=api_key,
            model=text_model,
        )
        obj = _parse_json(out)
    except (GeminiError, json.JSONDecodeError) as exc:
        return JSONResponse({"ok": False, "error": f"Gemini refine failed: {exc}"}, status_code=400)

    return {
        "ok": True,
        "refined_idea": str(obj.get("refined_idea") or idea).strip(),
        "refined_director_brief": str(obj.get("refined_director_brief") or director_brief or "").strip(),
        "refined_image_prompt": str(obj.get("refined_image_prompt") or image_prompt or "").strip(),
        "checklist": [str(x).strip() for x in (obj.get("checklist") or []) if str(x).strip()][:8],
    }


@app.post("/api/onboarding/quickstart")
def onboarding_quickstart(req: OnboardingReq):
    try:
        prompt = _bounded_text("prompt", req.prompt, max_len=4000, required=False)
        audience = _bounded_text("audience", req.audience, max_len=200, required=False) or "first-time creators"
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    settings = load_settings()
    api_key = settings.get("api_key")
    text_model = req.model or settings.get("text_model")
    image_model = req.image_model or settings.get("image_model")

    steps = _default_onboarding_steps()
    intro_title = "NorthStar quick tour"
    intro_body = "Follow the highlights to go from idea to rendered explainer in under a minute."
    outro_title = "You are ready to create"
    outro_body = "Press Create plan, generate assets, approve, and render your first scene."
    warnings: list[str] = []

    if api_key:
        schema = {
            "type": "OBJECT",
            "properties": {
                "intro_title": {"type": "STRING"},
                "intro_body": {"type": "STRING"},
                "outro_title": {"type": "STRING"},
                "outro_body": {"type": "STRING"},
                "steps": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "title": {"type": "STRING"},
                            "body": {"type": "STRING"},
                            "hint": {"type": "STRING"},
                            "target": {"type": "STRING"},
                            "icon_prompt": {"type": "STRING"},
                        },
                        "required": ["title", "body", "hint", "target", "icon_prompt"],
                    },
                },
            },
            "required": ["intro_title", "intro_body", "outro_title", "outro_body", "steps"],
        }
        targets = [step["target"] for step in _default_onboarding_steps()]
        user = (
            "Write a concise, premium onboarding walkthrough for a Manim creator studio.\n"
            "Audience: "
            + audience
            + "\n"
            + ("Current user intent:\n" + prompt + "\n\n" if prompt else "")
            + "Use exactly six steps mapped to these targets in order:\n"
            + "\n".join([f"{idx + 1}. {target}" for idx, target in enumerate(targets)])
            + "\n\n"
            + "Each step should have:\n"
            + "- title (max 6 words)\n"
            + "- body (max 24 words)\n"
            + "- hint (max 6 words)\n"
            + "- icon_prompt (max 20 words)\n"
            + "Return strict JSON only."
        )
        try:
            out = generate_content(
                user,
                system_text=(
                    "You are a product onboarding writer for creative AI tools. "
                    "Keep copy calm, clear, and action-oriented."
                ),
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                },
                api_key=api_key,
                model=text_model,
            )
            obj = _parse_json(out)
            intro_title = str(obj.get("intro_title") or intro_title).strip()[:120]
            intro_body = str(obj.get("intro_body") or intro_body).strip()[:260]
            outro_title = str(obj.get("outro_title") or outro_title).strip()[:120]
            outro_body = str(obj.get("outro_body") or outro_body).strip()[:260]
            steps = _normalize_onboarding_steps(obj.get("steps"))
        except (GeminiError, json.JSONDecodeError, ValueError) as exc:
            warnings.append(f"Gemini onboarding copy fallback: {exc}")
    else:
        warnings.append("GEMINI_API_KEY is not set; using built-in onboarding copy and placeholders.")

    tour_id = f"{int(time.time())}-{secrets.token_hex(4)}"
    tour_dir = WORK / "onboarding" / tour_id
    tour_dir.mkdir(parents=True, exist_ok=True)

    for idx, step in enumerate(steps, start=1):
        step["icon_url"] = ""
        if not api_key:
            continue
        icon_prompt = str(step.get("icon_prompt") or "").strip()
        if not icon_prompt:
            continue
        try:
            img = generate_image(icon_prompt, model=image_model, api_key=api_key)
            out_path = tour_dir / f"step-{idx}.png"
            out_path.write_bytes(img)
            step["icon_url"] = f"/work/onboarding/{tour_id}/{out_path.name}"
        except GeminiError as exc:
            warnings.append(f"Step {idx} icon fallback: {str(exc)[:180]}")

    return {
        "ok": True,
        "tour_id": tour_id,
        "intro_title": intro_title,
        "intro_body": intro_body,
        "outro_title": outro_title,
        "outro_body": outro_body,
        "steps": steps,
        "warnings": warnings[:12],
        "models": {
            "text_model": text_model,
            "image_model": image_model,
        },
    }


@app.post("/api/plan")
def plan(req: PlanReq):
    try:
        idea = _bounded_text("idea", req.idea, max_len=6000, required=True)
        _ = _bounded_text(
            "director_brief",
            req.director_brief,
            max_len=24000,
            required=False,
            truncate=True,
        )
        _ = _bounded_text("image_prompt", req.image_prompt, max_len=1200, required=False)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    reason = _policy_block_reason(idea, req.director_brief, req.image_prompt)
    if reason:
        return JSONResponse({"ok": False, "error": reason}, status_code=400)

    job_id = new_job_id()
    paths = job_paths(JOBS, job_id)
    paths.job_dir.mkdir(parents=True, exist_ok=True)

    settings = load_settings()
    api_key = settings.get("api_key")
    text_model = req.model or settings.get("text_model")

    try:
        plan_text = generate_content(
            scene_plan_user_prompt(idea, director_brief=_build_director_brief(req)),
            system_text=SCENE_PLAN_SYSTEM,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": SCENE_PLAN_SCHEMA,
            },
            api_key=api_key,
            model=text_model,
        )
        plan_obj = _parse_json(plan_text)
        paths.plan_path.write_text(json.dumps(plan_obj, indent=2), encoding="utf-8")
    except (GeminiError, json.JSONDecodeError) as exc:
        return JSONResponse(
            {"ok": False, "job_id": job_id, "error": f"Plan generation failed: {exc}"},
            status_code=400,
        )

    st = JobState(
        job_id=job_id,
        status="planned",
        step="idle",
        message="Plan ready.",
        updated_at=__import__("time").time(),
        plan_path=str(paths.plan_path),
        scene_path=str(paths.scene_path),
        logs_path=str(paths.logs_path),
    )
    write_state(paths.job_dir, st)
    append_event(paths.job_dir, type_="state", payload={"status": st.status, "step": st.step, "message": st.message})

    return {
        "ok": True,
        "job_id": job_id,
        "plan": plan_obj,
        "plan_text": json.dumps(plan_obj, indent=2),
        "job_files": _job_files(paths),
    }


@app.post("/api/approve")
def approve(req: ApproveReq):
    try:
        _ = _bounded_text("plan_text", req.plan_text, max_len=180000, required=True)
        _ = _bounded_text("image_prompt", req.image_prompt, max_len=1200, required=False)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    paths = job_paths(JOBS, req.job_id)
    paths.job_dir.mkdir(parents=True, exist_ok=True)
    if job_manager.is_running(req.job_id):
        return {"ok": True, "job_id": req.job_id, "status": "already_running"}

    settings = load_settings()
    preflight = _preflight_payload(settings)
    if not preflight.get("ok"):
        return JSONResponse(
            {
                "ok": False,
                "job_id": req.job_id,
                "error": "Preflight failed. Open Settings and run Get started.",
                "preflight": preflight,
            },
            status_code=400,
        )
    api_key = settings.get("api_key")
    text_model = req.model or settings.get("text_model")
    manim_py = settings.get("manim_py")
    if manim_py:
        # If the user saved "python" on macOS (often missing), don't hard-fail renders.
        import shutil
        from pathlib import Path as _Path

        try:
            p = _Path(str(manim_py))
            exists = (p.exists() if (p.is_absolute() or "/" in str(manim_py)) else False) or (shutil.which(str(manim_py)) is not None)
        except Exception:
            exists = False
        if not exists:
            manim_py = None

    try:
        plan_obj = _parse_json(req.plan_text)
        paths.plan_path.write_text(json.dumps(plan_obj, indent=2), encoding="utf-8")
    except json.JSONDecodeError as exc:
        return JSONResponse(
            {"ok": False, "job_id": req.job_id, "error": f"Invalid plan JSON: {exc}"},
            status_code=400,
        )

    # Describe any pre-generated assets already present on disk (created via /api/images/generate).
    assets_description = ""
    bg_candidates = sorted((paths.job_dir / "assets").glob("background*.png"))
    fg_candidates = sorted((paths.job_dir / "assets").glob("foreground*.png"))
    if not bg_candidates and not fg_candidates and req.include_images and req.image_prompt and req.image_prompt.strip():
        try:
            bg_rel, fg_rel, _warning, desc = _generate_assets(
                job_dir=paths.job_dir,
                image_prompt=req.image_prompt,
                image_mode=req.image_mode,
                api_key=api_key,
                image_model=req.image_model or settings.get("image_model"),
                variants=max(1, int(req.image_variants or 1)),
            )
            assets_description = desc or ""
        except ValueError as exc:
            return JSONResponse(
                {"ok": False, "job_id": req.job_id, "error": str(exc)}, status_code=400
            )
    else:
        if bg_candidates:
            bg_rel = str(bg_candidates[0].relative_to(paths.job_dir))
            assets_description += f"- background: {bg_rel} (full-frame backdrop, low motion, z_index -10)\n"
        if fg_candidates:
            fg_rel = str(fg_candidates[0].relative_to(paths.job_dir))
            assets_description += f"- foreground: {fg_rel} (small prop/character in lower third)\n"

    # Mark planned state and kick off async approve worker.
    st = JobState(
        job_id=req.job_id,
        status="running",
        step="code",
        message="Queued…",
        updated_at=__import__("time").time(),
        plan_path=str(paths.plan_path),
        scene_path=str(paths.scene_path),
        logs_path=str(paths.logs_path),
    )
    write_state(paths.job_dir, st)
    append_event(paths.job_dir, type_="state", payload={"status": st.status, "step": st.step, "message": st.message})

    job_manager.start_approve(
        job_id=req.job_id,
        job_dir=paths.job_dir,
        plan_obj=plan_obj,
        plan_text=req.plan_text,
        assets_description=assets_description,
        render_settings=_render_settings_ratio(req.aspect_ratio),
        quality=req.quality,
        manim_py=manim_py,
        api_key=api_key,
        text_model=text_model,
    )

    return {"ok": True, "job_id": req.job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    paths = job_paths(JOBS, job_id)
    if not paths.job_dir.exists():
        return JSONResponse({"ok": False, "job_id": job_id, "error": "Unknown job_id"}, status_code=404)
    st = load_state(paths.job_dir, job_id)
    resp: Dict[str, Any] = {
        "ok": True,
        "job_id": job_id,
        "status": st.status,
        "step": st.step,
        "message": st.message,
        "updated_at": st.updated_at,
        "error": st.error,
        "diagnosis": st.diagnosis,
        "code_diff": st.code_diff,
        "retry_result": st.retry_result,
        "running": job_manager.is_running(job_id),
    }
    if paths.out_mp4.exists():
        resp["video_path"] = str(paths.out_mp4.relative_to(ROOT))
    if paths.scene_path.exists():
        resp["code"] = paths.scene_path.read_text(encoding="utf-8")
    if paths.plan_path.exists():
        resp["plan"] = json.loads(paths.plan_path.read_text(encoding="utf-8"))
    if paths.logs_path.exists():
        # Avoid sending huge logs; UI will use SSE for tail.
        txt = paths.logs_path.read_text(encoding="utf-8")
        resp["logs_tail"] = txt[-6000:]
    captions = paths.job_dir / "captions.srt"
    if captions.exists():
        resp["captions_path"] = str(captions.relative_to(ROOT))
    resp["job_files"] = _job_files(paths) + ([str(captions.relative_to(ROOT))] if captions.exists() else [])
    return resp


@app.get("/api/jobs/{job_id}/events")
def job_events(job_id: str):
    import time
    import json as _json

    paths = job_paths(JOBS, job_id)
    if not paths.job_dir.exists():
        return JSONResponse({"ok": False, "job_id": job_id, "error": "Unknown job_id"}, status_code=404)

    def gen():
        # Initial state snapshot.
        st = load_state(paths.job_dir, job_id)
        yield f"event: state\ndata: {_json.dumps({'status': st.status, 'step': st.step, 'message': st.message, 'error': st.error})}\n\n"

        log_pos = 0
        state_mtime = 0.0
        code_mtime = 0.0
        sent_code = False
        while True:
            # Stream log growth.
            try:
                if paths.logs_path.exists():
                    size = paths.logs_path.stat().st_size
                    if size < log_pos:
                        log_pos = 0
                    if size > log_pos:
                        with paths.logs_path.open("r", encoding="utf-8", errors="ignore") as f:
                            f.seek(log_pos)
                            chunk = f.read(min(24_000, size - log_pos))
                            log_pos = f.tell()
                        if chunk:
                            yield f"event: log\ndata: {_json.dumps({'chunk': chunk})}\n\n"
            except Exception:
                pass

            # Stream code once it exists (gives IDE-like feedback).
            try:
                if paths.scene_path.exists():
                    mt = paths.scene_path.stat().st_mtime
                    if (not sent_code) or (mt != code_mtime and st.status in {"running", "repairing"}):
                        code_mtime = mt
                        sent_code = True
                        txt = paths.scene_path.read_text(encoding="utf-8", errors="ignore")
                        yield f"event: code\ndata: {_json.dumps({'code': txt})}\n\n"
            except Exception:
                pass

            # Stream state changes.
            try:
                sp = paths.job_dir / "state.json"
                if sp.exists():
                    mt = sp.stat().st_mtime
                    if mt != state_mtime:
                        state_mtime = mt
                        st = load_state(paths.job_dir, job_id)
                        yield f"event: state\ndata: {_json.dumps({'status': st.status, 'step': st.step, 'message': st.message, 'error': st.error})}\n\n"
                        if st.status in {"done", "failed"}:
                            break
            except Exception:
                pass

            time.sleep(0.4)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/jobs/{job_id}/download")
def job_download(job_id: str):
    import zipfile

    paths = job_paths(JOBS, job_id)
    if not paths.job_dir.exists():
        return JSONResponse({"ok": False, "job_id": job_id, "error": "Unknown job_id"}, status_code=404)

    zip_path = paths.job_dir / "export.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths.job_dir.rglob("*"):
            if p.is_dir():
                continue
            # Avoid zipping the zip itself while writing.
            if p.name == zip_path.name:
                continue
            zf.write(p, arcname=str(p.relative_to(paths.job_dir)))

    return FileResponse(zip_path, filename=f"{job_id}.zip")


@app.post("/api/jobs/{job_id}/copy-output")
def copy_job_output(job_id: str):
    settings = load_settings()
    ok, out = _copy_output_video(job_id, output_copy_dir=settings.get("output_copy_dir"))
    if not ok:
        return JSONResponse({"ok": False, "error": out}, status_code=400)
    return {"ok": True, "copied_path": out}


@app.post("/api/jobs/{job_id}/voiceover")
def add_voiceover(job_id: str, req: Optional[VoiceoverReq] = None):
    req = req or VoiceoverReq()
    settings = load_settings()
    api_key = (settings.get("elevenlabs_api_key") or "").strip()
    voice_id = (req.voice_id or settings.get("elevenlabs_voice_id") or "").strip()
    model_id = (req.model_id or settings.get("elevenlabs_model_id") or "").strip() or None
    gemini_key = (settings.get("api_key") or "").strip()
    gemini_model = settings.get("text_model")
    if not api_key or not voice_id:
        return JSONResponse(
            {
                "ok": False,
                "error": "Configure ElevenLabs API key and Voice ID in Settings first.",
            },
            status_code=400,
        )

    paths = job_paths(JOBS, job_id)
    if not paths.job_dir.exists():
        return JSONResponse({"ok": False, "error": "Unknown job_id"}, status_code=404)

    script_text = (req.script_text or "").strip()
    chat_context = (req.chat_context or "").strip()
    if script_text:
        text = script_text
    else:
        text = _srt_to_plain_text(paths.job_dir / "captions.srt")
        if not text:
            text = _voiceover_text_from_plan(paths)
        if req.use_gemini_script and gemini_key:
            text = _voiceover_script_with_gemini(
                paths=paths,
                api_key=gemini_key,
                model=gemini_model,
                chat_context=chat_context if req.include_chat_context else "",
            ) or text

    if req.include_chat_context and chat_context:
        context_block = f"Context cues:\n{chat_context[:1800]}"
        text = f"{context_block}\n\n{text}"

    text = text.strip()
    if len(text) > 3800:
        text = text[:3800]
    if not text:
        return JSONResponse({"ok": False, "error": "No narration text available for voiceover."}, status_code=400)

    ok, msg = _add_elevenlabs_voiceover(
        paths=paths,
        api_key=api_key,
        voice_id=voice_id,
        model_id=model_id,
        text=text,
    )
    if not ok:
        return JSONResponse({"ok": False, "error": msg}, status_code=400)
    return {
        "ok": True,
        "video_path": msg,
        "voice_id": voice_id,
        "model_id": model_id or "",
        "job_files": _job_files(paths),
    }


@app.post("/api/jobs/{job_id}/script-packs")
def build_script_packs(job_id: str, req: Optional[ScriptPackReq] = None):
    req = req or ScriptPackReq()
    paths = job_paths(JOBS, job_id)
    if not paths.job_dir.exists():
        return JSONResponse({"ok": False, "error": "Unknown job_id"}, status_code=404)
    if not paths.plan_path.exists():
        return JSONResponse({"ok": False, "error": "Plan not found for this job."}, status_code=400)
    try:
        plan = json.loads(paths.plan_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"Invalid plan JSON: {exc}"}, status_code=400)

    settings = load_settings()
    api_key = settings.get("api_key")
    model = req.model or settings.get("text_model")
    languages = [str(x).strip().lower() for x in (req.languages or []) if str(x).strip()]
    packs_data = _multilingual_script_packs(
        plan=plan,
        languages=languages,
        api_key=api_key,
        model=model,
    )

    scripts_dir = paths.job_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    out: list[Dict[str, Any]] = []
    for p in packs_data.get("packs", []):
        code = str(p.get("language") or "").strip().lower()
        if not code:
            continue
        script_txt = str(p.get("voice_script") or "").strip()
        captions_srt = str(p.get("captions_srt") or "").strip() + "\n"
        script_path = scripts_dir / f"narration-{code}.txt"
        srt_path = scripts_dir / f"captions-{code}.srt"
        script_path.write_text(script_txt + "\n", encoding="utf-8")
        srt_path.write_text(captions_srt, encoding="utf-8")
        out.append(
            {
                "language": code,
                "language_label": p.get("language_label") or _language_label(code),
                "script_path": str(script_path.relative_to(ROOT)),
                "captions_path": str(srt_path.relative_to(ROOT)),
                "preview": script_txt[:220],
            }
        )

    return {
        "ok": True,
        "job_id": job_id,
        "packs": out,
        "job_files": _job_files(paths),
    }


@app.post("/api/crazy-run")
def crazy_run(req: CrazyRunReq):
    try:
        idea = _bounded_text("idea", req.idea, max_len=6000, required=True)
        _ = _bounded_text(
            "director_brief",
            req.director_brief,
            max_len=24000,
            required=False,
            truncate=True,
        )
        _ = _bounded_text("image_prompt", req.image_prompt, max_len=1200, required=False)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    reason = _policy_block_reason(idea, req.director_brief, req.image_prompt)
    if reason:
        return JSONResponse({"ok": False, "error": reason}, status_code=400)

    settings = load_settings()
    preflight = _preflight_payload(settings)
    if not preflight.get("ok"):
        return JSONResponse(
            {
                "ok": False,
                "error": "Preflight failed. Open Settings and run Get started.",
                "preflight": preflight,
            },
            status_code=400,
        )

    api_key = settings.get("api_key")
    text_model = req.model or settings.get("text_model")
    image_model = req.image_model or settings.get("image_model")
    manim_py = settings.get("manim_py")
    if manim_py:
        import shutil
        from pathlib import Path as _Path

        try:
            p = _Path(str(manim_py))
            exists = (p.exists() if (p.is_absolute() or "/" in str(manim_py)) else False) or (shutil.which(str(manim_py)) is not None)
        except Exception:
            exists = False
        if not exists:
            manim_py = None

    count = max(1, min(5, int(req.variants or 3)))
    variant_briefs = [
        "Variant focus: hook-first, energetic pacing, minimal equations.",
        "Variant focus: visual analogy-first, smooth pacing, strong intuition.",
        "Variant focus: equation + graph-first, rigorous but concise.",
        "Variant focus: story-first with cinematic transitions and minimal text.",
        "Variant focus: exam-ready structure with clear definitions and checkpoints.",
    ]

    jobs: list[Dict[str, Any]] = []
    errors: list[str] = []
    for i in range(count):
        job_id = new_job_id()
        paths = job_paths(JOBS, job_id)
        paths.job_dir.mkdir(parents=True, exist_ok=True)
        try:
            brief = _build_director_brief(req) + "\n" + variant_briefs[i % len(variant_briefs)]
            plan_text = generate_content(
                scene_plan_user_prompt(idea, director_brief=brief),
                system_text=SCENE_PLAN_SYSTEM,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": SCENE_PLAN_SCHEMA,
                },
                api_key=api_key,
                model=text_model,
            )
            plan_obj = _parse_json(plan_text)
            paths.plan_path.write_text(json.dumps(plan_obj, indent=2), encoding="utf-8")

            assets_description = ""
            if req.include_images and req.image_prompt and req.image_prompt.strip():
                bg_rel, fg_rel, _warning, desc = _generate_assets(
                    job_dir=paths.job_dir,
                    image_prompt=req.image_prompt,
                    image_mode=req.image_mode,
                    api_key=api_key,
                    image_model=image_model,
                    variants=max(1, int(req.image_variants or 1)),
                )
                assets_description = desc or ""
                _ = (bg_rel, fg_rel)

            st = JobState(
                job_id=job_id,
                status="running",
                step="code",
                message="Queued…",
                updated_at=__import__("time").time(),
                plan_path=str(paths.plan_path),
                scene_path=str(paths.scene_path),
                logs_path=str(paths.logs_path),
            )
            write_state(paths.job_dir, st)
            append_event(
                paths.job_dir,
                type_="state",
                payload={"status": st.status, "step": st.step, "message": st.message},
            )
            job_manager.start_approve(
                job_id=job_id,
                job_dir=paths.job_dir,
                plan_obj=plan_obj,
                plan_text=json.dumps(plan_obj, indent=2),
                assets_description=assets_description,
                render_settings=_render_settings_ratio(req.aspect_ratio),
                quality=req.quality,
                manim_py=manim_py,
                api_key=api_key,
                text_model=text_model,
            )
            jobs.append(
                {
                    "job_id": job_id,
                    "variant_index": i + 1,
                    "variant_label": f"Variant {i + 1}",
                    "plan": plan_obj,
                    "plan_text": json.dumps(plan_obj, indent=2),
                    "job_files": _job_files(paths),
                }
            )
        except Exception as exc:
            errors.append(f"Variant {i + 1}: {exc}")

    if not jobs:
        return JSONResponse(
            {"ok": False, "error": "Crazy mode failed.", "errors": errors},
            status_code=500,
        )
    return {"ok": True, "jobs": jobs, "errors": errors}


@app.post("/api/jobs/append")
def append_job_video(req: AppendReq):
    base_paths = job_paths(JOBS, req.base_job_id)
    next_paths = job_paths(JOBS, req.next_job_id)
    if not base_paths.job_dir.exists():
        return JSONResponse(
            {"ok": False, "error": f"Unknown base_job_id: {req.base_job_id}"},
            status_code=404,
        )
    if not next_paths.job_dir.exists():
        return JSONResponse(
            {"ok": False, "error": f"Unknown next_job_id: {req.next_job_id}"},
            status_code=404,
        )
    if not base_paths.out_mp4.exists():
        return JSONResponse(
            {"ok": False, "error": f"Base job has no output video: {base_paths.out_mp4.relative_to(ROOT)}"},
            status_code=400,
        )
    if not next_paths.out_mp4.exists():
        return JSONResponse(
            {"ok": False, "error": f"Next job has no output video: {next_paths.out_mp4.relative_to(ROOT)}"},
            status_code=400,
        )

    merged_tmp = next_paths.job_dir / "out-merged.mp4"
    ok, logs = concat_videos(base_paths.out_mp4, next_paths.out_mp4, merged_tmp)
    (next_paths.job_dir / "append_logs.txt").write_text(logs or "", encoding="utf-8")
    if not ok or not merged_tmp.exists():
        return JSONResponse(
            {
                "ok": False,
                "error": "Video stitching failed",
                "logs": logs[-6000:] if logs else "",
            },
            status_code=500,
        )
    merged_tmp.replace(next_paths.out_mp4)

    merged_plan: Dict[str, Any] | None = None
    if base_paths.plan_path.exists() and next_paths.plan_path.exists():
        try:
            base_plan = json.loads(base_paths.plan_path.read_text(encoding="utf-8"))
            nxt_plan = json.loads(next_paths.plan_path.read_text(encoding="utf-8"))
            merged_plan = _merge_plans(base_plan, nxt_plan)
            next_paths.plan_path.write_text(
                json.dumps(merged_plan, indent=2),
                encoding="utf-8",
            )
        except Exception:
            merged_plan = None

    return {
        "ok": True,
        "job_id": req.next_job_id,
        "video_path": str(next_paths.out_mp4.relative_to(ROOT)),
        "plan": merged_plan,
        "job_files": _job_files(next_paths),
    }


@app.post("/api/jobs/{job_id}/cut-range")
def cut_job_video_range(job_id: str, req: CutRangeReq):
    paths = job_paths(JOBS, job_id)
    if not paths.job_dir.exists():
        return JSONResponse({"ok": False, "error": "Unknown job_id"}, status_code=404)
    if not paths.out_mp4.exists():
        return JSONResponse({"ok": False, "error": "No rendered video to cut yet."}, status_code=400)

    start_sec = float(req.start_sec)
    end_sec = float(req.end_sec)
    if start_sec < 0 or end_sec <= start_sec:
        return JSONResponse({"ok": False, "error": "Invalid cut range."}, status_code=400)

    tmp_out = paths.job_dir / "out-cut.mp4"
    ok, logs = cut_video_range(
        paths.out_mp4,
        start_s=start_sec,
        end_s=end_sec,
        out_video=tmp_out,
    )
    (paths.job_dir / "cut_logs.txt").write_text(logs or "", encoding="utf-8")
    if not ok or not tmp_out.exists():
        return JSONResponse(
            {
                "ok": False,
                "error": "Video cut failed",
                "logs": (logs or "")[-7000:],
            },
            status_code=500,
        )

    tmp_out.replace(paths.out_mp4)
    plan_obj: Dict[str, Any] | None = None
    if paths.plan_path.exists():
        try:
            existing = json.loads(paths.plan_path.read_text(encoding="utf-8"))
            plan_obj = _cut_plan_range(existing, start_sec=start_sec, end_sec=end_sec)
            paths.plan_path.write_text(json.dumps(plan_obj, indent=2), encoding="utf-8")
        except Exception:
            plan_obj = None

    return {
        "ok": True,
        "job_id": job_id,
        "video_path": str(paths.out_mp4.relative_to(ROOT)),
        "plan": plan_obj,
        "job_files": _job_files(paths),
    }


@app.post("/api/animate")
def animate(req: AnimateReq):
    try:
        idea = _bounded_text("idea", req.idea, max_len=6000, required=True)
        _ = _bounded_text(
            "director_brief",
            req.director_brief,
            max_len=24000,
            required=False,
            truncate=True,
        )
        _ = _bounded_text("image_prompt", req.image_prompt, max_len=1200, required=False)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    reason = _policy_block_reason(idea, req.director_brief, req.image_prompt)
    if reason:
        return JSONResponse({"ok": False, "error": reason}, status_code=400)

    job_id = new_job_id()
    paths = job_paths(JOBS, job_id)
    paths.job_dir.mkdir(parents=True, exist_ok=True)
    image_warning: Optional[str] = None
    settings = load_settings()
    api_key = settings.get("api_key")
    text_model = settings.get("text_model")
    image_model = req.image_model or settings.get("image_model")
    manim_py = settings.get("manim_py")

    # 1) Plan
    try:
        plan_text = generate_content(
            scene_plan_user_prompt(idea, director_brief=_build_director_brief(req)),
            system_text=SCENE_PLAN_SYSTEM,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": SCENE_PLAN_SCHEMA,
            },
            api_key=api_key,
            model=text_model,
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
    bg_rel: list[str] = []
    fg_rel: list[str] = []
    if req.include_images and req.image_prompt and req.image_prompt.strip():
        try:
            bg_rel, fg_rel, image_warning, assets_description = _generate_assets(
                job_dir=paths.job_dir,
                image_prompt=req.image_prompt,
                image_mode=req.image_mode,
                api_key=api_key,
                image_model=image_model,
                variants=max(1, int(req.image_variants or 1)),
            )
        except ValueError as exc:
            return JSONResponse(
                {"ok": False, "job_id": job_id, "error": str(exc)}, status_code=400
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
            api_key=api_key,
            model=text_model,
        )
        code = sanitize_manim_code(code)
        paths.scene_path.write_text(code, encoding="utf-8")
    except (GeminiError, CodeSanitizationError) as exc:
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
    ok, logs = render_with_manim(
        paths.scene_path,
        paths.out_mp4,
        quality=req.quality,
        manim_py=manim_py,
    )
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
            code2 = generate_content(repair_user, system_text=REPAIR_SYSTEM, api_key=api_key)
            code2 = sanitize_manim_code(code2)
            paths.scene_path.write_text(code2, encoding="utf-8")
            ok, logs = render_with_manim(
                paths.scene_path,
                paths.out_mp4,
                quality=req.quality,
                manim_py=manim_py,
            )
            paths.logs_path.write_text(logs, encoding="utf-8")
        except (GeminiError, CodeSanitizationError) as exc:
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
        "job_files": _job_files(paths),
    }
    if image_warning:
        response["image_warning"] = image_warning
    if bg_rel or fg_rel:
        response["assets"] = {
            "background": bg_rel[0] if bg_rel else None,
            "foreground": fg_rel[0] if fg_rel else None,
            "backgrounds": bg_rel,
            "foregrounds": fg_rel,
        }
    return response


@app.post("/api/render-code")
def render_code(req: RenderCodeReq):
    job_id = new_job_id()
    paths = job_paths(JOBS, job_id)
    paths.job_dir.mkdir(parents=True, exist_ok=True)

    # If the user edits code, keep it mostly as-is, but normalize tabs/trailing whitespace.
    try:
        clean_code = sanitize_manim_code(req.code)
    except CodeSanitizationError as exc:
        return JSONResponse(
            {"ok": False, "job_id": job_id, "error": f"Invalid code: {exc}"},
            status_code=400,
        )
    paths.scene_path.write_text(clean_code, encoding="utf-8")
    settings = load_settings()
    ok, logs = render_with_manim(
        paths.scene_path,
        paths.out_mp4,
        quality=req.quality,
        manim_py=settings.get("manim_py"),
    )
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
        "job_files": _job_files(paths),
    }
