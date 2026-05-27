from __future__ import annotations

import ast
import math
import re
from textwrap import dedent


class CodeSanitizationError(ValueError):
    pass


RISKY_MANIM_CALLS = {
    "ThreeDScene",
    "ThreeDAxes",
    "Surface",
    "ParametricSurface",
    "Sphere",
    "Cube",
    "Prism",
    "Torus",
    "OpenGLSurface",
    "OpenGLVMobject",
    "StreamLines",
    "AnimatedStreamLines",
    "ArrowVectorField",
    "VectorField",
}

BLOCKED_RUNTIME_CALLS = {
    "eval",
    "exec",
    "compile",
    "input",
    "open",
    "__import__",
    "os.system",
    "subprocess.Popen",
    "subprocess.run",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "requests.get",
    "requests.post",
    "requests.request",
    "urllib.request.urlopen",
    "Path.read_text",
    "Path.write_text",
    "read_text",
    "write_text",
    "read_bytes",
    "write_bytes",
}

BLOCKED_IMPORT_ROOTS = {
    "os",
    "subprocess",
    "requests",
    "urllib",
    "socket",
    "pathlib",
}

MOBJECT_CALLS = {
    "Text",
    "MarkupText",
    "Dot",
    "Arrow",
    "Axes",
    "NumberPlane",
    "Circle",
    "Rectangle",
    "Square",
    "Line",
    "DashedLine",
    "Polygon",
    "VGroup",
    "ImageMobject",
}

ANIMATION_CALLS = {
    "FadeIn",
    "FadeOut",
    "Create",
    "Write",
    "Transform",
    "ReplacementTransform",
    "LaggedStart",
    "AnimationGroup",
    "GrowArrow",
    "GrowFromCenter",
    "MoveAlongPath",
}


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _call_tail(name: str) -> str:
    return name.rsplit(".", 1)[-1] if name else ""


def _literal_number(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        value = float(node.operand.value)
        return -value if isinstance(node.op, ast.USub) else value
    return None


def _range_iterations(call: ast.Call) -> int | None:
    if _call_tail(_dotted_name(call.func)) != "range":
        return None
    nums = [_literal_number(arg) for arg in call.args[:3]]
    if any(num is None for num in nums):
        return None
    try:
        if len(nums) == 1:
            return max(0, int(nums[0] or 0))
        if len(nums) >= 2:
            start = nums[0] or 0
            stop = nums[1] or 0
            step = nums[2] if len(nums) >= 3 and nums[2] not in (None, 0) else 1
            return max(0, int(math.ceil((stop - start) / step)))
    except Exception:
        return None
    return None


def _first_text_arg_len(call: ast.Call) -> int:
    if not call.args:
        return 0
    arg = call.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return len(arg.value)
    if isinstance(arg, ast.JoinedStr):
        return sum(
            len(part.value)
            for part in arg.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
    return 0


def _kw_number(call: ast.Call, key: str) -> float | None:
    for kw in call.keywords:
        if kw.arg == key:
            return _literal_number(kw.value)
    return None


def manim_render_safety_issues(code: str, *, max_issues: int = 8) -> list[str]:
    """Static preflight for hosted Manim renders.

    The sanitizer checks syntax and the required class. This pass catches code
    shapes that are valid Python but unreliable for a paid hosted render.
    """
    try:
        tree = ast.parse(code or "")
    except SyntaxError as exc:
        return [f"Python syntax error before render: {exc.msg}"]

    issues: list[str] = []
    loop_ranges: list[int] = []
    play_calls = 0
    mobject_calls = 0
    animation_calls = 0
    always_redraw_calls = 0
    updater_calls = 0
    tex_calls = 0

    def add_issue(message: str) -> None:
        if message not in issues and len(issues) < max_issues:
            issues.append(message)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif node.module:
                names = [node.module]
            for name in names:
                root = str(name or "").split(".", 1)[0]
                if root in BLOCKED_IMPORT_ROOTS:
                    add_issue(f"blocked import '{root}' is not allowed in generated Manim code")

        if isinstance(node, (ast.For, ast.comprehension)) and isinstance(node.iter, ast.Call):
            iterations = _range_iterations(node.iter)
            if iterations is not None:
                loop_ranges.append(iterations)
                if iterations > 48:
                    add_issue(f"dense loop range({iterations}) exceeds hosted render budget")

        if not isinstance(node, ast.Call):
            continue

        name = _dotted_name(node.func)
        tail = _call_tail(name)

        if name in BLOCKED_RUNTIME_CALLS or tail in BLOCKED_RUNTIME_CALLS:
            add_issue(f"blocked runtime call '{name or tail}' is not allowed during render")
        if tail in RISKY_MANIM_CALLS:
            add_issue(f"risky Manim construct '{tail}' is disabled for hosted renders")
        if tail in MOBJECT_CALLS:
            mobject_calls += 1
        if tail in ANIMATION_CALLS:
            animation_calls += 1
        if tail == "always_redraw":
            always_redraw_calls += 1
        if tail == "add_updater":
            updater_calls += 1
        if tail in {"Tex", "MathTex"}:
            tex_calls += 1
            if _first_text_arg_len(node) > 120:
                add_issue(f"{tail} block is too large; use short Text labels or split the equation")

        if name == "self.wait" or tail == "wait":
            wait_seconds = _literal_number(node.args[0]) if node.args else None
            if wait_seconds is not None and wait_seconds > 12:
                add_issue(f"single wait({wait_seconds:g}) is too long for hosted scene timing")

        if name == "self.play" or tail == "play":
            play_calls += 1
            if len(node.args) > 8:
                add_issue(f"self.play has {len(node.args)} arguments; split or simplify the animation")
            run_time = _kw_number(node, "run_time")
            if run_time is not None and run_time > 8:
                add_issue(f"self.play run_time={run_time:g} is too long for preview-safe rendering")

    if always_redraw_calls > 1:
        add_issue(f"{always_redraw_calls} always_redraw calls exceed the hosted limit of 1")
    if updater_calls:
        add_issue("add_updater is disabled for hosted renders; use simple Transform or ValueTracker animation")
    if tex_calls > 4:
        add_issue(f"{tex_calls} Tex/MathTex calls are too many; prefer short Text labels")
    if mobject_calls > 96:
        add_issue(f"{mobject_calls} object constructor calls exceed the hosted render budget")
    if animation_calls > 72:
        add_issue(f"{animation_calls} animation calls exceed the hosted render budget")
    if play_calls > 48:
        add_issue(f"{play_calls} self.play calls exceed the hosted render budget")
    if sum(loop_ranges) > 120:
        add_issue(f"loop-generated objects/actions are too dense ({sum(loop_ranges)} total iterations)")

    return issues


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


def sanitize_manim_code(code: str) -> str:
    """Strict sanitizer for model-produced Manim code before rendering.

    - Removes Markdown code fences.
    - Normalizes tabs/newlines/indent baseline.
    - Validates Python syntax.
    - Requires class GeneratedScene(Scene).
    """
    raw = strip_markdown_fences(code)
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = raw.replace("\t", "    ")
    raw = dedent(raw).strip()
    if not raw:
        raise CodeSanitizationError("Generated code is empty.")
    formatted = format_python(raw)
    if "```" in formatted:
        formatted = formatted.replace("```python", "").replace("```py", "").replace("```", "").strip() + "\n"

    try:
        tree = ast.parse(formatted)
    except SyntaxError as exc:
        line = exc.lineno or "?"
        msg = exc.msg or "syntax error"
        raise CodeSanitizationError(f"Invalid Python syntax at line {line}: {msg}") from exc

    generated_scene = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "GeneratedScene":
            generated_scene = node
            break
    if generated_scene is None:
        raise CodeSanitizationError("Missing required class: GeneratedScene(Scene).")

    def _is_scene_base(base: ast.expr) -> bool:
        if isinstance(base, ast.Name):
            return base.id == "Scene"
        if isinstance(base, ast.Attribute):
            return base.attr == "Scene"
        return False

    if not any(_is_scene_base(base) for base in generated_scene.bases):
        raise CodeSanitizationError("GeneratedScene must inherit from Scene.")

    return formatted
