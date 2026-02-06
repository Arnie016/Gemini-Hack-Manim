from __future__ import annotations

import base64
import json
import re
import secrets
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .gemini_http import GeminiError, generate_content, generate_image
from .prompts import (
    MANIM_CODE_SYSTEM,
    REPAIR_SYSTEM,
    SCENE_PLAN_SCHEMA,
    SCENE_PLAN_SYSTEM,
    manim_code_user_prompt,
    scene_plan_user_prompt,
)
from .renderer import concat_videos, render_with_manim
from .storage import job_paths, new_job_id
from .templates import TEMPLATES
from .code_format import format_python
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


class UploadJobAssetReq(BaseModel):
    job_id: str
    filename: str
    content_base64: str
    role: str = "background"


class AppendReq(BaseModel):
    base_job_id: str
    next_job_id: str


class SourceIndexReq(BaseModel):
    url: str
    notes: Optional[str] = None
    source_type: str = "auto"  # auto | youtube | web
    model: Optional[str] = None


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
        lines.append(f"Additional brief: {req.director_brief}")
    if getattr(req, "include_images", False) and getattr(req, "image_prompt", None):
        lines.append(
            f"Include visual assets based on: {getattr(req, 'image_prompt')}"
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


def _render_settings_ratio(ratio: str) -> str:
    ratio = ratio.strip()
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
) -> tuple[Optional[str], Optional[str], Optional[str], str]:
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

    bg_rel = None
    fg_rel = None
    warning: Optional[str] = None
    try:
        if mode in {"background", "both"}:
            bg_bytes = generate_image(
                f"Wide background scene, no text, cinematic lighting: {base_prompt}",
                api_key=api_key,
                model=image_model,
            )
            (assets_dir / "background.png").write_bytes(bg_bytes)
            bg_rel = "assets/background.png"

        if mode in {"foreground", "both"}:
            fg_bytes = generate_image(
                f"Single character or prop, centered, clean background, no text: {base_prompt}",
                api_key=api_key,
                model=image_model,
            )
            (assets_dir / "foreground.png").write_bytes(fg_bytes)
            fg_rel = "assets/foreground.png"
    except GeminiError as exc:
        warning = f"Image generation failed: {exc}"
        bg_rel = None
        fg_rel = None

    desc = ""
    if bg_rel:
        desc += f"- background: {bg_rel} (full-frame backdrop, low motion, z_index -10)\n"
    if fg_rel:
        desc += f"- foreground: {fg_rel} (small prop/character in lower third)\n"
    return bg_rel, fg_rel, warning, desc


@app.get("/api/health")
def health():
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

    settings = load_settings()
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
    return {
        "manim_ok": manim_ok,
        "manim_version": manim_out.splitlines()[0] if manim_out else "",
        "ffmpeg_ok": ffmpeg_ok,
        "ffmpeg_version": ffmpeg_out.splitlines()[0] if ffmpeg_out else "",
        "ffmpeg_path": ffmpeg_path or "",
        "manim_py": used_py or manim_py_setting or "python3",
        "python_candidates": candidates,
    }


@app.get("/api/settings")
def get_settings():
    settings = load_settings()
    return {
        "has_api_key": bool(settings.get("api_key")),
        "text_model": settings.get("text_model"),
        "image_model": settings.get("image_model"),
        "manim_py": settings.get("manim_py"),
    }


@app.post("/api/settings")
def set_settings(req: SettingsReq):
    settings = update_settings(
        {
            "api_key": req.api_key,
            "text_model": req.text_model,
            "image_model": req.image_model,
            "manim_py": req.manim_py,
        }
    )
    return {
        "ok": True,
        "has_api_key": bool(settings.get("api_key")),
        "text_model": settings.get("text_model"),
        "image_model": settings.get("image_model"),
        "manim_py": settings.get("manim_py"),
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

    settings = load_settings()
    api_key = settings.get("api_key")
    image_model = req.image_model or settings.get("image_model")
    try:
        bg_rel, fg_rel, warning, desc = _generate_assets(
            job_dir=paths.job_dir,
            image_prompt=req.image_prompt,
            image_mode=req.image_mode,
            api_key=api_key,
            image_model=image_model,
        )
    except ValueError as exc:
        return JSONResponse(
            {"ok": False, "job_id": req.job_id, "error": str(exc)}, status_code=400
        )

    resp: Dict[str, Any] = {
        "ok": True,
        "job_id": req.job_id,
        "assets": {"background": bg_rel, "foreground": fg_rel},
        "assets_description": desc,
    }
    if warning:
        resp["warning"] = warning
    return resp


@app.post("/api/docs/index")
def docs_index(req: SourceIndexReq):
    url = (req.url or "").strip()
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        return JSONResponse(
            {"ok": False, "error": "Please provide a valid http(s) URL."},
            status_code=400,
        )

    notes = (req.notes or "").strip()
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


@app.post("/api/plan")
def plan(req: PlanReq):
    job_id = new_job_id()
    paths = job_paths(JOBS, job_id)
    paths.job_dir.mkdir(parents=True, exist_ok=True)

    settings = load_settings()
    api_key = settings.get("api_key")
    text_model = req.model or settings.get("text_model")

    try:
        plan_text = generate_content(
            scene_plan_user_prompt(req.idea, director_brief=_build_director_brief(req)),
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
    paths = job_paths(JOBS, req.job_id)
    paths.job_dir.mkdir(parents=True, exist_ok=True)
    if job_manager.is_running(req.job_id):
        return {"ok": True, "job_id": req.job_id, "status": "already_running"}

    settings = load_settings()
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
    bg_rel = "assets/background.png"
    fg_rel = "assets/foreground.png"
    if (paths.job_dir / bg_rel).exists():
        assets_description += f"- background: {bg_rel} (full-frame backdrop, low motion, z_index -10)\n"
    if (paths.job_dir / fg_rel).exists():
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


@app.post("/api/animate")
def animate(req: AnimateReq):
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
            scene_plan_user_prompt(req.idea, director_brief=_build_director_brief(req)),
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
    bg_rel = None
    fg_rel = None
    if req.include_images and req.image_prompt and req.image_prompt.strip():
        try:
            bg_rel, fg_rel, image_warning, assets_description = _generate_assets(
                job_dir=paths.job_dir,
                image_prompt=req.image_prompt,
                image_mode=req.image_mode,
                api_key=api_key,
                image_model=image_model,
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
        code = format_python(code)
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
            code2 = format_python(code2)
            paths.scene_path.write_text(code2, encoding="utf-8")
            ok, logs = render_with_manim(
                paths.scene_path,
                paths.out_mp4,
                quality=req.quality,
                manim_py=manim_py,
            )
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
        "job_files": _job_files(paths),
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

    # If the user edits code, keep it mostly as-is, but normalize tabs/trailing whitespace.
    paths.scene_path.write_text(format_python(req.code), encoding="utf-8")
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
