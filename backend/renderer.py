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


def _probe_duration_s(video_path: Path) -> tuple[bool, float, str]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return False, 0.0, "ffprobe not found on PATH"
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as exc:
        logs = (exc.stdout or "") + "\n" + (exc.stderr or "")
        return False, 0.0, logs + "\nffprobe timed out"
    out = (proc.stdout or "").strip()
    logs = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        return False, 0.0, logs or "ffprobe failed"
    try:
        return True, float(out), logs
    except ValueError:
        return False, 0.0, logs or "ffprobe returned invalid duration"


def _encode_trim_segment(
    *,
    ffmpeg: str,
    in_video: Path,
    out_video: Path,
    start_s: float | None = None,
    end_s: float | None = None,
    timeout_s: int = 240,
) -> tuple[bool, str]:
    cmd = [ffmpeg, "-y", "-i", str(in_video)]
    if start_s is not None and start_s > 0:
        cmd += ["-ss", f"{start_s:.3f}"]
    if end_s is not None and end_s > 0:
        if start_s is not None and start_s > 0:
            dur = max(0.05, end_s - start_s)
            cmd += ["-t", f"{dur:.3f}"]
        else:
            cmd += ["-to", f"{end_s:.3f}"]
    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(out_video),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        logs = (exc.stdout or "") + "\n" + (exc.stderr or "")
        return False, logs + "\nffmpeg trim timed out"
    logs = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode == 0 and out_video.exists(), logs


def cut_video_range(
    in_video: Path,
    *,
    start_s: float,
    end_s: float,
    out_video: Path,
    timeout_s: int = 300,
) -> tuple[bool, str]:
    """Remove [start_s, end_s] from a rendered video and output a new mp4."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False, "ffmpeg not found on PATH"
    if not in_video.exists():
        return False, f"Missing video: {in_video}"
    if start_s < 0 or end_s <= start_s:
        return False, "Invalid cut range"

    ok_dur, duration_s, probe_logs = _probe_duration_s(in_video)
    logs_all = probe_logs + "\n"
    if not ok_dur:
        return False, logs_all or "Could not read video duration"
    if start_s >= duration_s:
        return False, f"Cut start {start_s:.2f}s is beyond duration {duration_s:.2f}s"
    end_s = min(end_s, duration_s)
    if end_s - start_s <= 0.02:
        return False, "Cut range too small"

    out_video.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_video.parent / ".cut_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    part_a = tmp_dir / "part-a.mp4"
    part_b = tmp_dir / "part-b.mp4"

    # Case 1: remove from beginning.
    if start_s <= 0.02:
        ok_b, logs_b = _encode_trim_segment(
            ffmpeg=ffmpeg,
            in_video=in_video,
            out_video=part_b,
            start_s=end_s,
            end_s=None,
            timeout_s=timeout_s,
        )
        logs_all += "\n=== cut from start ===\n" + logs_b
        if not ok_b:
            return False, logs_all
        part_b.replace(out_video)
        return True, logs_all

    # Case 2: remove to end.
    if end_s >= duration_s - 0.02:
        ok_a, logs_a = _encode_trim_segment(
            ffmpeg=ffmpeg,
            in_video=in_video,
            out_video=part_a,
            start_s=0.0,
            end_s=start_s,
            timeout_s=timeout_s,
        )
        logs_all += "\n=== cut to end ===\n" + logs_a
        if not ok_a:
            return False, logs_all
        part_a.replace(out_video)
        return True, logs_all

    # Case 3: remove middle range and stitch the two sides.
    ok_a, logs_a = _encode_trim_segment(
        ffmpeg=ffmpeg,
        in_video=in_video,
        out_video=part_a,
        start_s=0.0,
        end_s=start_s,
        timeout_s=timeout_s,
    )
    logs_all += "\n=== part A ===\n" + logs_a
    if not ok_a:
        return False, logs_all

    ok_b, logs_b = _encode_trim_segment(
        ffmpeg=ffmpeg,
        in_video=in_video,
        out_video=part_b,
        start_s=end_s,
        end_s=None,
        timeout_s=timeout_s,
    )
    logs_all += "\n=== part B ===\n" + logs_b
    if not ok_b:
        return False, logs_all

    ok_concat, concat_logs = concat_videos(part_a, part_b, out_video, timeout_s=timeout_s)
    logs_all += "\n=== concat ===\n" + concat_logs
    if not ok_concat:
        return False, logs_all
    return True, logs_all
