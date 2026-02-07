from __future__ import annotations

import re


def strip_markdown_fences(text: str) -> str:
    """Extract Python code if model wrapped output in Markdown fences."""
    raw = (text or "").strip()
    if not raw:
        return ""

    # Prefer explicit fenced blocks.
    m = re.search(r"```(?:python|py)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    if m:
        return (m.group(1) or "").strip()

    # Fallback: strip stray fence lines even when the closing fence is missing.
    cleaned = re.sub(r"(?im)^\s*```(?:python|py)?\s*$", "", raw)
    cleaned = re.sub(r"(?im)^\s*```\s*$", "", cleaned)
    cleaned = cleaned.replace("```python", "").replace("```py", "").replace("```", "")
    return cleaned.strip()


def format_python(code: str) -> str:
    """Best-effort formatting for generated python.

    Uses Black if available, otherwise applies lightweight whitespace cleanup.
    """
    code = strip_markdown_fences(code)
    code = (code or "").replace("\t", "    ").strip() + "\n"
    # Fast path: if black isn't installed, keep code readable-ish.
    try:
        import black  # type: ignore

        mode = black.FileMode(line_length=88)
        return black.format_str(code, mode=mode)
    except Exception:
        # Strip trailing whitespace on each line.
        return "\n".join([ln.rstrip() for ln in code.splitlines()]).rstrip() + "\n"
