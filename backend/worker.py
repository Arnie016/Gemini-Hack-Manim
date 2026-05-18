from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .artifact_store import publish_job_artifacts
from .job_manager import JobManager
from .job_state import append_event, load_state, write_state
from .render_queue import QueueJob, claim_next_job, complete_job, ensure_queue
from .settings_store import load_settings
from .storage import job_paths


ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "work" / "jobs"


def _text_generation_settings(settings: Dict[str, Any], model: Optional[str] = None) -> tuple[str, Optional[str], str]:
    from .main import _text_generation_settings as main_text_generation_settings

    return main_text_generation_settings(settings, model)


def _render_settings_ratio(ratio: str) -> str:
    from .main import _render_settings_ratio as main_render_settings_ratio

    return main_render_settings_ratio(ratio)


def _render_manim_py(settings: Dict[str, Any]) -> str:
    from .main import _render_manim_py as main_render_manim_py

    return main_render_manim_py(settings)


def _write_share_package(paths) -> Dict[str, Any]:
    from .main import _write_share_package as main_write_share_package

    return main_write_share_package(paths)


def _generate_assets(
    *,
    job_dir: Path,
    image_prompt: str,
    image_mode: str,
    api_key: Optional[str],
    image_model: Optional[str],
    variants: int,
) -> tuple[list[str], list[str], Optional[str], str]:
    from .main import _generate_assets as main_generate_assets

    return main_generate_assets(
        job_dir=job_dir,
        image_prompt=image_prompt,
        image_mode=image_mode,
        api_key=api_key,
        image_model=image_model,
        variants=variants,
    )


def _mark_failed(job: QueueJob, message: str) -> None:
    paths = job_paths(JOBS, job.job_id)
    st = load_state(paths.job_dir, job.job_id)
    st.status = "failed"
    st.step = st.step or "render"
    st.message = "Failed."
    st.error = message
    st.diagnosis = message[:4000]
    st.updated_at = time.time()
    write_state(paths.job_dir, st)
    append_event(paths.job_dir, type_="state", payload={"status": st.status, "step": st.step, "error": st.error})


def run_queue_job(job: QueueJob, *, progress: str = "auto") -> bool:
    payload = job.payload
    settings = load_settings()
    text_provider, api_key, text_model = _text_generation_settings(settings, payload.get("model"))
    paths = job_paths(JOBS, job.job_id)
    paths.job_dir.mkdir(parents=True, exist_ok=True)

    plan_obj = payload.get("plan_obj")
    if not isinstance(plan_obj, dict):
        try:
            plan_obj = json.loads(paths.plan_path.read_text(encoding="utf-8"))
        except Exception as exc:
            _mark_failed(job, f"Queued job is missing a valid plan: {exc}")
            return False

    assets_description = str(payload.get("assets_description") or "")
    bg_candidates = sorted((paths.job_dir / "assets").glob("background*.png"))
    fg_candidates = sorted((paths.job_dir / "assets").glob("foreground*.png"))
    if not assets_description:
        if bg_candidates:
            bg_rel = str(bg_candidates[0].relative_to(paths.job_dir))
            assets_description += f"- background: {bg_rel} (full-frame backdrop, low motion, z_index -10)\n"
        if fg_candidates:
            fg_rel = str(fg_candidates[0].relative_to(paths.job_dir))
            assets_description += f"- foreground: {fg_rel} (small prop/character in lower third)\n"
    if (
        not bg_candidates
        and not fg_candidates
        and payload.get("include_images")
        and str(payload.get("image_prompt") or "").strip()
    ):
        image_api_key = settings.get("api_key") or os.environ.get("GEMINI_API_KEY")
        _bg_rel, _fg_rel, image_warning, desc = _generate_assets(
            job_dir=paths.job_dir,
            image_prompt=str(payload.get("image_prompt") or ""),
            image_mode=str(payload.get("image_mode") or "background"),
            api_key=image_api_key,
            image_model=str(payload.get("image_model") or settings.get("image_model") or ""),
            variants=max(1, int(payload.get("image_variants") or 1)),
        )
        assets_description = desc or assets_description
        if image_warning:
            append_event(paths.job_dir, type_="image_warning", payload={"warning": image_warning})

    if progress != "off":
        print(f"worker: claimed {job.job_id}", file=sys.stderr)

    JobManager().run_approve(
        job_id=job.job_id,
        job_dir=paths.job_dir,
        plan_obj=plan_obj,
        plan_text=str(payload.get("plan_text") or json.dumps(plan_obj, indent=2)),
        assets_description=assets_description,
        render_settings=str(payload.get("render_settings") or _render_settings_ratio(str(payload.get("aspect_ratio") or "9:16"))),
        quality=str(payload.get("quality") or "pql"),
        manim_py=str(payload.get("manim_py") or _render_manim_py(settings)),
        api_key=api_key,
        text_model=text_model,
        text_provider=text_provider,
    )

    st = load_state(paths.job_dir, job.job_id)
    if st.status == "done":
        try:
            _write_share_package(paths)
            publish_job_artifacts(job_id=job.job_id, job_dir=paths.job_dir)
        except Exception as exc:
            st = load_state(paths.job_dir, job.job_id)
            st.diagnosis = ((st.diagnosis + "\n") if st.diagnosis else "") + f"Artifact publish failed: {exc}"
            st.updated_at = time.time()
            write_state(paths.job_dir, st)
            append_event(
                paths.job_dir,
                type_="artifact_publish_error",
                payload={"error": str(exc)},
            )
        if progress != "off":
            print(f"worker: completed {job.job_id}", file=sys.stderr)
        return True

    if progress != "off":
        print(f"worker: failed {job.job_id}: {st.error or st.message}", file=sys.stderr)
    return False


def run_worker(*, once: bool = False, poll: float = 2.0, concurrency: int = 1, progress: str = "auto") -> int:
    if concurrency != 1:
        raise ValueError("The file-backed queue currently supports --concurrency 1 only.")
    ensure_queue()
    while True:
        job = claim_next_job()
        if not job:
            if once:
                if progress != "off":
                    print("worker: no pending jobs", file=sys.stderr)
                return 0
            time.sleep(max(0.2, float(poll)))
            continue
        error = ""
        ok = False
        try:
            ok = run_queue_job(job, progress=progress)
        except Exception as exc:
            error = str(exc)
            _mark_failed(job, f"Worker crashed: {exc}")
        complete_job(job, failed=not ok, error=error)
        if once:
            return 0 if ok else 1
