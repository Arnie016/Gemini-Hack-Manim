from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


class TerminalError(Exception):
    pass


ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "work" / "jobs"


def _run_fixed_command(args: list[str], *, timeout: int) -> str:
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    out = ((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")).strip()
    if proc.returncode != 0:
        return f"(exit {proc.returncode})\n{out}".strip()
    return out


def run_diagnostic_check(check: str, *, manim_py: Optional[str] = None) -> str:
    """Run one of the fixed hosted-safe diagnostic probes."""
    name = (check or "all").strip().lower()
    py = manim_py or "python3"

    if name in {"help", "?", "/help"}:
        return "Available diagnostics: all, manim, ffmpeg, jobs, disk"

    if name == "manim":
        return _run_fixed_command([py, "-m", "manim", "--version"], timeout=10)

    if name == "ffmpeg":
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return "ffmpeg not found on PATH"
        return _run_fixed_command([ffmpeg, "-version"], timeout=20)

    if name == "jobs":
        if not JOBS.exists():
            return "No jobs directory found."
        items = sorted(p.name for p in JOBS.iterdir() if not p.name.startswith("."))
        return "\n".join(items[:80]) or "No render jobs found."

    if name == "disk":
        usage = shutil.disk_usage(ROOT)
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        used_pct = ((usage.total - usage.free) / usage.total) * 100 if usage.total else 0
        return f"Disk: {free_gb:.1f} GB free / {total_gb:.1f} GB total ({used_pct:.0f}% used)"

    if name == "all":
        sections = []
        for item in ("manim", "ffmpeg", "disk", "jobs"):
            try:
                result = run_diagnostic_check(item, manim_py=py)
            except Exception as exc:
                result = str(exc)
            sections.append(f"== {item} ==\n{result}")
        return "\n\n".join(sections)

    raise TerminalError(f"Unknown diagnostic '{check}'. Use one of: all, manim, ffmpeg, jobs, disk.")


def run_terminal_command(command: str, *, manim_py: Optional[str] = None) -> str:
    """Run terminal commands from the project root for the middle-panel terminal."""
    cmd = (command or "").strip()
    if not cmd:
        raise TerminalError("Empty command")

    if cmd in {"help", "?", "/help"}:
        return (
            "Local developer terminal commands:\n"
            "- manim --version\n"
            "- ffmpeg -version\n"
            "- ls jobs\n"
            "- plus shell commands from the project root\n"
            "\nHosted mode exposes diagnostics only: all, manim, ffmpeg, jobs, disk\n"
        )

    if cmd == "ls jobs":
        return run_diagnostic_check("jobs", manim_py=manim_py)

    if cmd == "ffmpeg -version":
        return run_diagnostic_check("ffmpeg", manim_py=manim_py)

    if cmd == "manim --version":
        return run_diagnostic_check("manim", manim_py=manim_py)

    denied = [
        "rm -rf /",
        "shutdown",
        "reboot",
        "mkfs",
        ":(){:|:&};:",
    ]
    low = cmd.lower()
    for token in denied:
        if token in low:
            raise TerminalError("Command blocked for safety.")

    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except subprocess.TimeoutExpired:
        raise TerminalError("Command timed out after 60s")
    except Exception as exc:
        raise TerminalError(str(exc))

    out = ((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")).strip()
    if proc.returncode != 0:
        return f"(exit {proc.returncode})\n{out}".strip()
    return out
