from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


class TerminalError(Exception):
    pass


ROOT = Path(__file__).resolve().parents[1]


def run_terminal_command(command: str, *, manim_py: Optional[str] = None) -> str:
    """Run terminal commands from the project root for the middle-panel terminal."""
    cmd = (command or "").strip()
    if not cmd:
        raise TerminalError("Empty command")

    if cmd in {"help", "?", "/help"}:
        return (
            "Allowed commands:\n"
            "- manim --version\n"
            "- ffmpeg -version\n"
            "- ls jobs\n"
            "- plus most shell commands (runs in project root)\n"
        )

    if cmd == "ls jobs":
        proc = subprocess.run(
            ["ls", "-1", "work/jobs"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return (proc.stdout or "") + "\n" + (proc.stderr or "")
        return proc.stdout or ""

    if cmd == "ffmpeg -version":
        proc = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=20)
        return (proc.stdout or "") + "\n" + (proc.stderr or "")

    if cmd == "manim --version":
        py = manim_py or "python3"
        proc = subprocess.run(
            [py, "-m", "manim", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return (proc.stdout or "") + "\n" + (proc.stderr or "")

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
