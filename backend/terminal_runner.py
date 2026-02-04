from __future__ import annotations

import subprocess
from typing import Optional


class TerminalError(Exception):
    pass


def run_terminal_command(command: str, *, manim_py: Optional[str] = None) -> str:
    """Run a very small allow-list of safe commands for the UI terminal panel.

    This is NOT a general purpose shell.
    """
    cmd = (command or "").strip()
    if not cmd:
        raise TerminalError("Empty command")

    # Normalize common aliases.
    if cmd in {"help", "?", "/help"}:
        return (
            "Allowed commands:\n"
            "- manim --version\n"
            "- ffmpeg -version\n"
            "- ls jobs\n"
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

    raise TerminalError("Command not allowed. Type 'help' for options.")

