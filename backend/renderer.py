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
