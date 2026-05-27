from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple


DEFAULT_MAX_MP4_BYTES = 120 * 1024 * 1024


def max_mp4_bytes() -> int:
    try:
        return max(8 * 1024 * 1024, int(os.getenv("NORTHSTAR_MAX_MP4_BYTES", DEFAULT_MAX_MP4_BYTES)))
    except ValueError:
        return DEFAULT_MAX_MP4_BYTES


def max_upload_bytes() -> int:
    try:
        return max(512 * 1024, int(os.getenv("NORTHSTAR_MAX_UPLOAD_BYTES", 10 * 1024 * 1024)))
    except ValueError:
        return 10 * 1024 * 1024


def _crf_for_quality(quality: str) -> str:
    q = (quality or "").strip().lower()
    if q in {"pqh", "high"}:
        return "18"
    if q in {"pqm", "medium"}:
        return "21"
    return "25"


def postprocess_mp4(
    mp4_path: Path,
    *,
    quality: str = "pql",
    logs_path: Optional[Path] = None,
    timeout_s: int = 180,
    max_bytes: Optional[int] = None,
) -> Tuple[bool, str]:
    """Normalize MP4 output for sharing and enforce a hosted size budget."""
    if not mp4_path.exists():
        return False, f"MP4 missing: {mp4_path}"

    max_size = int(max_bytes or max_mp4_bytes())
    original_size = mp4_path.stat().st_size
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        if original_size > max_size:
            return False, f"MP4 is {original_size} bytes, over max {max_size}, and ffmpeg is unavailable."
        return True, "ffmpeg unavailable; kept original MP4."

    tmp = mp4_path.with_name(f".{mp4_path.stem}.postprocess.mp4")
    crf = _crf_for_quality(quality)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(mp4_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        crf,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(tmp),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        msg = ((exc.stdout or "") + "\n" + (exc.stderr or "") + "\nMP4 postprocess timed out").strip()
        if logs_path:
            logs_path.open("a", encoding="utf-8").write(f"\n\n=== mp4 postprocess ===\n{msg}\n")
        return original_size <= max_size, msg

    logs = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    if proc.returncode != 0 or not tmp.exists():
        if logs_path:
            logs_path.open("a", encoding="utf-8").write(f"\n\n=== mp4 postprocess failed ===\n{logs[-4000:]}\n")
        return original_size <= max_size, logs[-4000:] or "ffmpeg postprocess failed"

    new_size = tmp.stat().st_size
    if new_size > max_size:
        tmp.unlink(missing_ok=True)
        msg = f"MP4 is too large after postprocess: {new_size} bytes exceeds max {max_size} bytes."
        if logs_path:
            logs_path.open("a", encoding="utf-8").write(f"\n\n=== mp4 size guard ===\n{msg}\n")
        return False, msg

    tmp.replace(mp4_path)
    msg = f"MP4 postprocessed with H.264 faststart: {original_size} -> {new_size} bytes."
    if logs_path:
        logs_path.open("a", encoding="utf-8").write(f"\n\n=== mp4 postprocess ===\n{msg}\n")
    return True, msg
