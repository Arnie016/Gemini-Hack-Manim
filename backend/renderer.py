from __future__ import annotations

import os
import subprocess
import shutil
from pathlib import Path
from typing import Tuple

from .video_postprocess import postprocess_mp4


def _subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _timeout_logs(exc: subprocess.TimeoutExpired) -> str:
    return _subprocess_text(exc.stdout) + "\n" + _subprocess_text(exc.stderr)


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
        logs = _timeout_logs(exc)
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
    if ok:
        post_ok, post_logs = postprocess_mp4(out_mp4, quality=quality)
        if not post_ok:
            return False, logs + "\n\n=== mp4 postprocess ===\n" + post_logs
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
        logs_all += _timeout_logs(exc)
        logs_all += "\nffmpeg concat timed out\n"
        proc = None

    if proc is not None and proc.returncode == 0 and tmp_out.exists():
        tmp_out.replace(out_video)
        try:
            list_file.unlink(missing_ok=True)
        except Exception:
            pass
        return True, logs_all

    def _has_audio(path: Path) -> bool:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return False
        try:
            probe = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=index",
                    "-of",
                    "csv=p=0",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            return probe.returncode == 0 and bool((probe.stdout or "").strip())
        except Exception:
            return False

    # Fallback: re-encode concat filter.
    first_has_audio = _has_audio(first_video)
    second_has_audio = _has_audio(second_video)
    if first_has_audio and second_has_audio:
        cmd_reencode = [
            ffmpeg,
            "-y",
            "-i",
            str(first_video),
            "-i",
            str(second_video),
            "-filter_complex",
            "[0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1[v][a]",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(tmp_out),
        ]
    else:
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


def cut_video_range(
    input_video: Path,
    *,
    start_s: float,
    end_s: float,
    out_video: Path,
    timeout_s: int = 240,
) -> Tuple[bool, str]:
    """Remove the time range [start_s, end_s] from an MP4 with ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg:
        return False, "ffmpeg not found on PATH"
    if not input_video.exists():
        return False, f"Missing video: {input_video}"

    try:
        start_s = max(0.0, float(start_s))
        end_s = float(end_s)
    except (TypeError, ValueError):
        return False, "Invalid cut timestamps"
    if end_s <= start_s:
        return False, "Invalid cut range"

    out_video.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_video.parent / f".cut-{out_video.stem}.mp4"
    logs_all = ""

    def _run(cmd: list[str]) -> tuple[bool, str]:
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            logs = (exc.stdout or "") + "\n" + (exc.stderr or "")
            return False, logs + "\nffmpeg cut timed out"
        logs = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return proc.returncode == 0, logs

    def _probe_duration() -> float | None:
        if not ffprobe:
            return None
        ok, out = _run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(input_video),
            ]
        )
        if not ok:
            return None
        try:
            value = float(out.strip().splitlines()[-1])
        except (IndexError, ValueError):
            return None
        return value if value > 0 else None

    def _has_audio() -> bool:
        if not ffprobe:
            return False
        ok, out = _run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(input_video),
            ]
        )
        return ok and bool(out.strip())

    duration = _probe_duration()
    has_audio = _has_audio()

    keep_head = start_s > 0.05
    keep_tail = True
    if duration is not None:
        end_s = min(end_s, duration)
        keep_tail = end_s < duration - 0.05

    if not keep_head and not keep_tail:
        return False, "Cut range removes the entire video"

    base_cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_video),
    ]
    encode_args = [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]

    if keep_head and keep_tail:
        if has_audio:
            graph = (
                f"[0:v]trim=start=0:end={start_s:.3f},setpts=PTS-STARTPTS[v0];"
                f"[0:a]atrim=start=0:end={start_s:.3f},asetpts=PTS-STARTPTS[a0];"
                f"[0:v]trim=start={end_s:.3f},setpts=PTS-STARTPTS[v1];"
                f"[0:a]atrim=start={end_s:.3f},asetpts=PTS-STARTPTS[a1];"
                "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]"
            )
            cmd = base_cmd + [
                "-filter_complex",
                graph,
                "-map",
                "[v]",
                "-map",
                "[a]",
            ] + encode_args + ["-c:a", "aac", "-b:a", "192k", str(tmp_out)]
        else:
            graph = (
                f"[0:v]trim=start=0:end={start_s:.3f},setpts=PTS-STARTPTS[v0];"
                f"[0:v]trim=start={end_s:.3f},setpts=PTS-STARTPTS[v1];"
                "[v0][v1]concat=n=2:v=1:a=0[v]"
            )
            cmd = base_cmd + [
                "-filter_complex",
                graph,
                "-map",
                "[v]",
                "-an",
            ] + encode_args + [str(tmp_out)]
    elif keep_head:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(input_video),
            "-t",
            f"{start_s:.3f}",
        ] + encode_args
        if has_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        else:
            cmd += ["-an"]
        cmd.append(str(tmp_out))
    else:
        cmd = [
            ffmpeg,
            "-y",
            "-ss",
            f"{end_s:.3f}",
            "-i",
            str(input_video),
        ] + encode_args
        if has_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        else:
            cmd += ["-an"]
        cmd.append(str(tmp_out))

    ok, logs = _run(cmd)
    logs_all += logs
    if ok and tmp_out.exists():
        tmp_out.replace(out_video)
        return True, logs_all

    try:
        tmp_out.unlink(missing_ok=True)
    except Exception:
        pass
    return False, logs_all or "ffmpeg cut failed"
