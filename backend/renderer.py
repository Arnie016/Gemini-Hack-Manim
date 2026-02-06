from __future__ import annotations

import os
import subprocess
import shutil
from pathlib import Path
from typing import Tuple


def render_with_manim(
    scene_file: Path,
    out_mp4: Path,
    *,
    manim_py: str | None = None,
    quality: str = "pql",
    timeout_s: int = 180,
) -> Tuple[bool, str]:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    # Prefer python3 on macOS (many machines no longer ship `python`).
    manim_py = manim_py or os.getenv("MANIM_PY") or "python3"

    quality_map = {
        "pql": "-ql",
        "low": "-ql",
        "pqm": "-qm",
        "medium": "-qm",
        "pqh": "-qh",
        "high": "-qh",
    }
    quality_flag = quality_map.get(quality.lower(), "-ql")

    cmd = [
        manim_py,
        "-m",
        "manim",
        str(scene_file),
        "GeneratedScene",
        quality_flag,
        "--media_dir",
        str(out_mp4.parent),
        "-o",
        out_mp4.name,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(scene_file.parent),
        )
    except FileNotFoundError as exc:
        return False, f"Executable not found: {exc}"
    except subprocess.TimeoutExpired as exc:
        logs = (exc.stdout or "") + "\n" + (exc.stderr or "")
        return False, logs + "\nRender timed out"

    logs = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode == 0 and not out_mp4.exists():
        candidates = sorted(
            out_mp4.parent.glob(f"videos/**/{out_mp4.name}"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
            reverse=True,
        )
        if candidates:
            out_mp4.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidates[0], out_mp4)
    ok = proc.returncode == 0 and out_mp4.exists()
    return ok, logs


def concat_videos(
    first_video: Path,
    second_video: Path,
    out_video: Path,
    *,
    timeout_s: int = 240,
) -> Tuple[bool, str]:
    """Concatenate two MP4 videos with ffmpeg.

    Fast path uses stream copy concat. Fallback re-encodes if stream copy fails.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False, "ffmpeg not found on PATH"
    if not first_video.exists():
        return False, f"Missing video: {first_video}"
    if not second_video.exists():
        return False, f"Missing video: {second_video}"

    out_video.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_video.parent / f".concat-{out_video.stem}.mp4"
    list_file = out_video.parent / ".concat-inputs.txt"
    logs_all = ""

    def _q(path: Path) -> str:
        # ffmpeg concat demuxer expects quoted POSIX-like paths.
        return str(path.resolve()).replace("'", "'\\''")

    list_file.write_text(
        f"file '{_q(first_video)}'\nfile '{_q(second_video)}'\n",
        encoding="utf-8",
    )

    cmd_copy = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(tmp_out),
    ]
    try:
        proc = subprocess.run(
            cmd_copy,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        logs_all += (proc.stdout or "") + "\n" + (proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        logs_all += (exc.stdout or "") + "\n" + (exc.stderr or "")
        logs_all += "\nffmpeg concat timed out\n"
        proc = None

    if proc is not None and proc.returncode == 0 and tmp_out.exists():
        tmp_out.replace(out_video)
        try:
            list_file.unlink(missing_ok=True)
        except Exception:
            pass
        return True, logs_all

    # Fallback: re-encode concat filter.
    cmd_reencode = [
        ffmpeg,
        "-y",
        "-i",
        str(first_video),
        "-i",
        str(second_video),
        "-filter_complex",
        "[0:v:0][1:v:0]concat=n=2:v=1:a=0[v]",
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(tmp_out),
    ]
    try:
        proc2 = subprocess.run(
            cmd_reencode,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        logs_all += "\n\n=== fallback re-encode ===\n"
        logs_all += (proc2.stdout or "") + "\n" + (proc2.stderr or "")
    except subprocess.TimeoutExpired as exc:
        logs_all += "\n\n=== fallback re-encode timeout ===\n"
        logs_all += (exc.stdout or "") + "\n" + (exc.stderr or "")
        logs_all += "\nffmpeg re-encode timed out\n"
        proc2 = None

    try:
        list_file.unlink(missing_ok=True)
    except Exception:
        pass

    if proc2 is not None and proc2.returncode == 0 and tmp_out.exists():
        tmp_out.replace(out_video)
        return True, logs_all

    return False, logs_all or "ffmpeg concat failed"
