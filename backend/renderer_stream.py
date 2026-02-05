from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Optional


def render_with_manim_stream(
    *,
    scene_file: Path,
    out_mp4: Path,
    logs_path: Path,
    manim_py: Optional[str] = None,
    quality: str = "pql",
    timeout_s: int = 420,
) -> bool:
    """Render with Manim while writing logs incrementally to logs_path."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    logs_path.parent.mkdir(parents=True, exist_ok=True)

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
    quality_flag = quality_map.get((quality or "").lower(), "-ql")

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

    start = time.time()
    with logs_path.open("a", encoding="utf-8") as logf:
        logf.write("\n\n=== manim run ===\n")
        logf.write(" ".join(cmd) + "\n\n")
        logf.flush()

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(scene_file.parent),
                bufsize=1,
                universal_newlines=True,
            )
        except FileNotFoundError as exc:
            logf.write(f"Executable not found: {exc}\n")
            return False

        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                if time.time() - start > timeout_s:
                    proc.kill()
                    logf.write("\nRender timed out\n")
                    return False
        finally:
            try:
                proc.wait(timeout=5)
            except Exception:
                pass

    return proc.returncode == 0 and out_mp4.exists()

