from __future__ import annotations

from typing import Optional


def format_python(code: str) -> str:
    """Best-effort formatting for generated python.

    Uses Black if available, otherwise applies lightweight whitespace cleanup.
    """
    code = (code or "").replace("\t", "    ").strip() + "\n"
    # Fast path: if black isn't installed, keep code readable-ish.
    try:
        import black  # type: ignore

        mode = black.FileMode(line_length=88)
        return black.format_str(code, mode=mode)
    except Exception:
        # Strip trailing whitespace on each line.
        return "\n".join([ln.rstrip() for ln in code.splitlines()]).rstrip() + "\n"

