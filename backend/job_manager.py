from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .code_format import format_python
from .gemini_http import GeminiError, generate_content
from .job_state import JobState, append_event, load_state, write_state
from .prompts import MANIM_CODE_SYSTEM, REPAIR_SYSTEM, manim_code_user_prompt
from .renderer_stream import render_with_manim_stream


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

            code = generate_content(
                manim_code_user_prompt(
                    json.dumps(plan_obj),
                    assets_description=assets_description,
                    render_settings=render_settings,
                ),
                system_text=MANIM_CODE_SYSTEM,
                api_key=api_key,
                model=text_model,
            )
            code = format_python(code)
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
                proc = subprocess.run(
                    [py, "-m", "manim", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if proc.returncode != 0:
                    with logs_path.open("a", encoding="utf-8") as f:
                        f.write("\n\n=== preflight ===\n")
                        f.write("Manim not available for this Python.\n")
                        f.write((proc.stdout or "") + (proc.stderr or "") + "\n")
                    state.status = "failed"
                    state.step = "render"
                    state.error = "Manim missing (install manim or choose a different Python in Settings → Rendering)"
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
                with logs_path.open("a", encoding="utf-8") as f:
                    f.write("\n\n=== preflight ===\n")
                    f.write(f"Preflight check failed: {exc}\n")

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

                repair_user = (
                    "The render failed.\n"
                    "Here are the logs:\n"
                    f"{logs}\n\n"
                    "Here is the code:\n"
                    f"{scene_path.read_text(encoding='utf-8')}\n\n"
                    "Return a fixed full python file."
                )
                code2 = generate_content(
                    repair_user,
                    system_text=REPAIR_SYSTEM,
                    api_key=api_key,
                    model=text_model,
                )
                code2 = format_python(code2)
                scene_path.write_text(code2, encoding="utf-8")

                state.status = "running"
                state.step = "render"
                state.message = "Rendering MP4 (retry)…"
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
            state.updated_at = time.time()
            write_state(job_dir, state)
            append_event(
                job_dir,
                type_="state",
                payload={"status": state.status, "step": state.step, "message": state.message},
            )

        except GeminiError as exc:
            state.status = "failed"
            state.step = state.step or "code"
            state.error = str(exc)
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
            state.message = "Failed."
            state.updated_at = time.time()
            write_state(job_dir, state)
            append_event(job_dir, type_="state", payload={"status": state.status, "step": state.step, "error": state.error})
