from __future__ import annotations

import os
import subprocess
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
    manim_py = manim_py or os.getenv("MANIM_PY", "python")

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
        "-o",
        str(out_mp4),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(scene_file.parent),
        )
    except subprocess.TimeoutExpired as exc:
        logs = (exc.stdout or "") + "\n" + (exc.stderr or "")
        return False, logs + "\nRender timed out"

    logs = (proc.stdout or "") + "\n" + (proc.stderr or "")
    ok = proc.returncode == 0 and out_mp4.exists()
    return ok, logs
