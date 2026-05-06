from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .code_format import CodeSanitizationError, sanitize_manim_code
from .gemini_http import GeminiError, generate_content
from .job_state import JobState, append_event, load_state, write_state
from .prompts import MANIM_CODE_SYSTEM, REPAIR_SYSTEM, manim_code_user_prompt
from .renderer_stream import render_with_manim_stream


MANIM_PREFLIGHT_TIMEOUT_S = 15
DEFAULT_OPENAI_CODE_MODEL = "gpt-5-mini"


def _task_model(selected_model: Optional[str], provider: Optional[str], env_key: str) -> Optional[str]:
    if (provider or "").strip().lower() != "openai":
        return selected_model
    return (os.getenv(env_key) or os.getenv("OPENAI_CODE_MODEL") or DEFAULT_OPENAI_CODE_MODEL).strip() or selected_model


def _probe_manim_package(py: str) -> tuple[bool, str]:
    proc = subprocess.run(
        [
            py,
            "-c",
            "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('manim') else 1)",
        ],
        capture_output=True,
        text=True,
        timeout=MANIM_PREFLIGHT_TIMEOUT_S,
    )
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if proc.returncode == 0:
        return True, "Manim package is installed."
    return False, out or "Manim package is not installed for this Python."


def _append_failure_log(logs_path: Path, section: str, message: str) -> None:
    logs_path.parent.mkdir(parents=True, exist_ok=True)
    with logs_path.open("a", encoding="utf-8") as f:
        f.write(f"\n\n=== {section} ===\n")
        f.write((message or "Unknown failure").strip() + "\n")


def _compact_text(value: Any, *, max_chars: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = text.encode("ascii", "replace").decode("ascii")
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _fallback_manim_code(plan_obj: Dict[str, Any]) -> str:
    title = _compact_text(plan_obj.get("title") or "NorthStar storyboard", max_chars=80)
    raw_scenes = plan_obj.get("scenes") if isinstance(plan_obj.get("scenes"), list) else []
    scenes: list[Dict[str, Any]] = []
    for idx, sc in enumerate(raw_scenes[:8], start=1):
        if not isinstance(sc, dict):
            continue
        try:
            seconds = float(sc.get("seconds") or 4)
        except (TypeError, ValueError):
            seconds = 4.0
        bullets = []
        for field in ("elements", "actions"):
            values = sc.get(field) if isinstance(sc.get(field), list) else []
            for item in values[:3]:
                if len(bullets) >= 4:
                    break
                bullets.append(_compact_text(item, max_chars=110))
        scenes.append(
            {
                "index": idx,
                "seconds": max(2.0, min(7.0, seconds)),
                "goal": _compact_text(sc.get("goal") or f"Scene {idx}", max_chars=150),
                "narration": _compact_text(sc.get("narration") or "", max_chars=180),
                "bullets": [item for item in bullets if item],
            }
        )
    if not scenes:
        scenes.append(
            {
                "index": 1,
                "seconds": 4.0,
                "goal": "Create a clear first storyboard.",
                "narration": "NorthStar created a fallback storyboard because code generation did not finish.",
                "bullets": ["Review the plan.", "Edit the code.", "Render again when ready."],
            }
        )

    return f'''from manim import *
import textwrap

TITLE = {json.dumps(title, ensure_ascii=True)}
SCENES = {json.dumps(scenes, ensure_ascii=True, indent=2)}


def wrap_text(text, width=34, max_lines=4):
    clean = " ".join(str(text or "").split())
    lines = textwrap.wrap(clean, width=width)[:max_lines]
    return "\\n".join(lines) if lines else ""


class GeneratedScene(Scene):
    def construct(self):
        self.camera.background_color = "#08111f"
        total = max(1, len(SCENES))
        for idx, scene in enumerate(SCENES, start=1):
            duration = max(2.0, min(7.0, float(scene.get("seconds") or 4)))
            progress = idx / total

            grid = NumberPlane(
                x_range=[-8, 8, 1],
                y_range=[-5, 5, 1],
                background_line_style={{"stroke_color": BLUE_E, "stroke_width": 1, "stroke_opacity": 0.16}},
            )
            heading = Text(wrap_text(TITLE, 28, 1), font_size=30, color=WHITE).to_edge(UP, buff=0.34)
            badge = Text(f"Scene {{idx}}/{{total}}", font_size=20, color=BLUE_B).to_corner(UL, buff=0.35)
            goal = Text(wrap_text(scene.get("goal"), 33, 3), font_size=27, color=WHITE, line_spacing=0.9)
            goal.move_to(UP * 2.0)

            beam = VGroup(
                Arrow(LEFT * 3.6, RIGHT * 3.6, buff=0, color=BLUE_B, stroke_width=5),
                Rectangle(width=0.38, height=2.15, color=TEAL_B, stroke_width=4).shift(LEFT * 0.9),
                Rectangle(width=0.38, height=2.15, color=GREEN_B, stroke_width=4).shift(RIGHT * 0.9),
                Circle(radius=0.38, color=YELLOW_B, stroke_width=4).shift(RIGHT * 3.15),
            ).move_to(DOWN * 0.2)

            bullet_lines = []
            for item in scene.get("bullets", [])[:4]:
                wrapped = wrap_text(item, 34, 2).split("\\n")
                if wrapped and wrapped[0]:
                    bullet_lines.append(f"- {{wrapped[0]}}")
                    bullet_lines.extend([f"  {{line}}" for line in wrapped[1:] if line])
            bullets = Text("\\n".join(bullet_lines), font_size=19, color=GRAY_A, line_spacing=0.75)
            bullets.next_to(beam, DOWN, buff=0.45)

            narration = Text(wrap_text(scene.get("narration"), 40, 2), font_size=20, color=TEAL_A, line_spacing=0.8)
            narration.to_edge(DOWN, buff=0.72)

            start = LEFT * 3.7 + DOWN * 3.05
            end = RIGHT * 3.7 + DOWN * 3.05
            track = Line(start, end, color=GRAY_E, stroke_width=7)
            fill = Line(start, start + (end - start) * progress, color=GREEN_C, stroke_width=7)

            card = VGroup(heading, badge, goal, beam, bullets, narration, track, fill)
            self.add(grid)
            self.play(FadeIn(heading), FadeIn(badge), Write(goal), run_time=0.7)
            self.play(Create(beam), FadeIn(bullets, shift=UP * 0.08), Create(track), Create(fill), run_time=0.85)
            if scene.get("narration"):
                self.play(FadeIn(narration, shift=UP * 0.08), run_time=0.45)
            self.wait(max(0.3, duration - 2.0))
            self.play(FadeOut(card), FadeOut(grid), run_time=0.25)
'''


def _build_srt(plan: Dict[str, Any]) -> str:
    def fmt(ts: float) -> str:
        ts = max(0.0, float(ts))
        h = int(ts // 3600)
        m = int((ts % 3600) // 60)
        s = int(ts % 60)
        ms = int(round((ts - int(ts)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    scenes = plan.get("scenes") or []
    lines: list[str] = []
    cursor = 0.0
    idx = 1
    for sc in scenes:
        dur = float(sc.get("seconds") or 0)
        text = str(sc.get("narration") or "").strip()
        if not text:
            cursor += max(0.0, dur)
            continue
        start = cursor
        end = cursor + max(0.5, dur)
        lines.append(str(idx))
        lines.append(f"{fmt(start)} --> {fmt(end)}")
        lines.append(text)
        lines.append("")
        idx += 1
        cursor += max(0.0, dur)
    return "\n".join(lines).strip() + "\n"


def _diagnose_logs(logs: str) -> str:
    """Extract the most useful render failure lines for the UI."""
    text = logs or ""
    if not text.strip():
        return "Render failed before logs were written."
    interesting: list[str] = []
    needles = (
        "traceback",
        "error",
        "exception",
        "failed",
        "modulenotfounderror",
        "nameerror",
        "typeerror",
        "valueerror",
        "attributeerror",
    )
    for line in text.splitlines():
        clean = line.strip()
        if not clean:
            continue
        low = clean.lower()
        if any(token in low for token in needles):
            interesting.append(clean)
    if not interesting:
        interesting = [line.strip() for line in text.splitlines() if line.strip()][-12:]
    return "\n".join(interesting[-24:])[:4000]


def _code_diff(before: str, after: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile="scene.py.before",
        tofile="scene.py.after",
        lineterm="",
    )
    return "\n".join(diff)[:12000]


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._threads: Dict[str, threading.Thread] = {}

    def is_running(self, job_id: str) -> bool:
        with self._lock:
            t = self._threads.get(job_id)
            return bool(t and t.is_alive())

    def start_approve(
        self,
        *,
        job_id: str,
        job_dir: Path,
        plan_obj: Dict[str, Any],
        plan_text: str,
        assets_description: str,
        render_settings: str,
        quality: str,
        manim_py: Optional[str],
        api_key: Optional[str],
        text_model: Optional[str],
        text_provider: Optional[str] = None,
    ) -> None:
        with self._lock:
            if job_id in self._threads and self._threads[job_id].is_alive():
                return

            t = threading.Thread(
                target=self._approve_worker,
                daemon=True,
                kwargs={
                    "job_id": job_id,
                    "job_dir": job_dir,
                    "plan_obj": plan_obj,
                    "plan_text": plan_text,
                    "assets_description": assets_description,
                    "render_settings": render_settings,
                    "quality": quality,
                    "manim_py": manim_py,
                    "api_key": api_key,
                    "text_model": text_model,
                    "text_provider": text_provider,
                },
            )
            self._threads[job_id] = t
            t.start()

    def _approve_worker(
        self,
        *,
        job_id: str,
        job_dir: Path,
        plan_obj: Dict[str, Any],
        plan_text: str,
        assets_description: str,
        render_settings: str,
        quality: str,
        manim_py: Optional[str],
        api_key: Optional[str],
        text_model: Optional[str],
        text_provider: Optional[str] = None,
    ) -> None:
        import traceback

        plan_path = job_dir / "plan.json"
        scene_path = job_dir / "scene.py"
        logs_path = job_dir / "logs.txt"
        out_mp4 = job_dir / "out.mp4"
        captions_path = job_dir / "captions.srt"

        state = load_state(job_dir, job_id)
        try:
            state.status = "running"
            state.step = "code"
            state.message = "Generating Manim code…"
            state.updated_at = time.time()
            write_state(job_dir, state)
            append_event(
                job_dir,
                type_="state",
                payload={"status": state.status, "step": state.step, "message": state.message},
            )

            # Captions are cheap: generate from plan narration.
            try:
                captions_path.write_text(_build_srt(plan_obj), encoding="utf-8")
            except Exception:
                pass

            try:
                code = generate_content(
                    manim_code_user_prompt(
                        json.dumps(plan_obj),
                        assets_description=assets_description,
                        render_settings=render_settings,
                    ),
                    system_text=MANIM_CODE_SYSTEM,
                    api_key=api_key,
                    model=_task_model(text_model, text_provider, "OPENAI_CODE_MODEL"),
                    provider=text_provider,
                )
                code = sanitize_manim_code(code)
            except (GeminiError, CodeSanitizationError) as exc:
                _append_failure_log(
                    logs_path,
                    "storyboard fallback",
                    f"Model code generation failed; rendering deterministic storyboard fallback instead.\n{exc}",
                )
                code = _fallback_manim_code(plan_obj)
                state.diagnosis = f"Model code generation failed; rendered storyboard fallback instead. {exc}"[:4000]
                state.retry_result = "storyboard_fallback"
            scene_path.write_text(code, encoding="utf-8")

            state.step = "render"
            state.message = "Rendering MP4…"
            state.updated_at = time.time()
            state.plan_path = str(plan_path)
            state.scene_path = str(scene_path)
            state.logs_path = str(logs_path)
            write_state(job_dir, state)
            append_event(
                job_dir,
                type_="state",
                payload={"status": state.status, "step": state.step, "message": state.message},
            )

            # If Manim isn't installed for the selected python, fail fast (but keep the generated code).
            # This avoids long "frozen" renders + pointless repair attempts.
            try:
                py = manim_py or "python3"
                manim_ok, manim_out = _probe_manim_package(py)
                if not manim_ok:
                    _append_failure_log(
                        logs_path,
                        "preflight",
                        f"Manim not available for this Python.\n{manim_out}",
                    )
                    state.status = "failed"
                    state.step = "render"
                    state.error = "Manim missing (install manim or choose a different Python in Settings → Rendering)"
                    state.diagnosis = _diagnose_logs(manim_out)
                    state.retry_result = "not_attempted"
                    state.message = "Failed."
                    state.updated_at = time.time()
                    write_state(job_dir, state)
                    append_event(
                        job_dir,
                        type_="state",
                        payload={"status": state.status, "step": state.step, "error": state.error},
                    )
                    return
            except Exception as exc:
                _append_failure_log(logs_path, "preflight", f"Preflight check failed: {exc}")

            ok = render_with_manim_stream(
                scene_file=scene_path,
                out_mp4=out_mp4,
                logs_path=logs_path,
                quality=quality,
                manim_py=manim_py,
            )

            if not ok:
                state.status = "repairing"
                state.step = "repair"
                state.message = "Repairing code (1 retry)…"
                state.updated_at = time.time()
                write_state(job_dir, state)
                append_event(
                    job_dir,
                    type_="state",
                    payload={"status": state.status, "step": state.step, "message": state.message},
                )

                try:
                    logs = logs_path.read_text(encoding="utf-8")
                except Exception:
                    logs = ""
                state.diagnosis = _diagnose_logs(logs)
                state.retry_result = "repair_requested"
                write_state(job_dir, state)

                repair_user = (
                    "The render failed.\n"
                    "Here are the logs:\n"
                    f"{logs}\n\n"
                    "Here is the code:\n"
                    f"{scene_path.read_text(encoding='utf-8')}\n\n"
                    "Return a fixed full python file."
                )
                old_code = scene_path.read_text(encoding="utf-8")
                code2 = generate_content(
                    repair_user,
                    system_text=REPAIR_SYSTEM,
                    api_key=api_key,
                    model=_task_model(text_model, text_provider, "OPENAI_REPAIR_MODEL"),
                    provider=text_provider,
                )
                code2 = sanitize_manim_code(code2)
                scene_path.write_text(code2, encoding="utf-8")
                state.code_diff = _code_diff(old_code, code2)

                state.status = "running"
                state.step = "render"
                state.message = "Rendering MP4 (retry)…"
                state.retry_result = "retry_started"
                state.updated_at = time.time()
                write_state(job_dir, state)
                append_event(
                    job_dir,
                    type_="state",
                    payload={"status": state.status, "step": state.step, "message": state.message},
                )

                ok = render_with_manim_stream(
                    scene_file=scene_path,
                    out_mp4=out_mp4,
                    logs_path=logs_path,
                    quality=quality,
                    manim_py=manim_py,
                )

            if not ok:
                state.status = "failed"
                state.step = "render"
                state.error = "Render failed"
                state.retry_result = "retry_failed"
                try:
                    state.diagnosis = _diagnose_logs(logs_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
                state.message = "Failed."
                state.updated_at = time.time()
                write_state(job_dir, state)
                append_event(
                    job_dir,
                    type_="state",
                    payload={"status": state.status, "step": state.step, "error": state.error},
                )
                return

            state.status = "done"
            state.step = "idle"
            state.message = "Render complete."
            state.video_path = str(out_mp4)
            if state.retry_result == "retry_started":
                state.retry_result = "fixed_on_retry"
            state.updated_at = time.time()
            write_state(job_dir, state)
            append_event(
                job_dir,
                type_="state",
                payload={"status": state.status, "step": state.step, "message": state.message},
            )

        except GeminiError as exc:
            err = str(exc)
            try:
                _append_failure_log(logs_path, "model request", err)
            except Exception:
                pass
            state.status = "failed"
            state.step = state.step or "code"
            state.error = err
            state.diagnosis = err[:4000]
            state.message = "Failed."
            state.updated_at = time.time()
            write_state(job_dir, state)
            append_event(job_dir, type_="state", payload={"status": state.status, "step": state.step, "error": state.error})
        except CodeSanitizationError as exc:
            err = f"Invalid generated code: {exc}"
            try:
                _append_failure_log(logs_path, "code validation", err)
            except Exception:
                pass
            state.status = "failed"
            state.step = state.step or "code"
            state.error = err
            state.diagnosis = err[:4000]
            state.message = "Failed."
            state.updated_at = time.time()
            write_state(job_dir, state)
            append_event(job_dir, type_="state", payload={"status": state.status, "step": state.step, "error": state.error})
        except Exception as exc:
            # Never let the thread die silently (UI would look "stuck rendering").
            try:
                logs_path.parent.mkdir(parents=True, exist_ok=True)
                with logs_path.open("a", encoding="utf-8") as f:
                    f.write("\n\n=== worker crash ===\n")
                    f.write(str(exc) + "\n")
                    f.write(traceback.format_exc() + "\n")
            except Exception:
                pass
            state.status = "failed"
            state.step = state.step or "render"
            state.error = f"Internal error: {exc}"
            try:
                state.diagnosis = _diagnose_logs(logs_path.read_text(encoding="utf-8"))
            except Exception:
                state.diagnosis = state.error
            state.message = "Failed."
            state.updated_at = time.time()
            write_state(job_dir, state)
            append_event(job_dir, type_="state", payload={"status": state.status, "step": state.step, "error": state.error})
