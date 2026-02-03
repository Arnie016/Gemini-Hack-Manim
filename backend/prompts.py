from __future__ import annotations

SCENE_PLAN_SYSTEM = """You are a video director for Manim Community Edition.
Return ONLY valid JSON that matches the schema.
Constraints:
- total_seconds must be <= 20
- scenes count must be between 3 and 6
- avoid more than 4 on-screen elements per scene
- keep language short and clear for short videos
"""

SCENE_PLAN_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"},
        "total_seconds": {"type": "NUMBER"},
        "scenes": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "seconds": {"type": "NUMBER"},
                    "goal": {"type": "STRING"},
                    "elements": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "actions": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "narration": {"type": "STRING"},
                },
                "required": ["seconds", "goal", "elements", "actions", "narration"],
            },
        },
    },
    "required": ["title", "total_seconds", "scenes"],
}


def scene_plan_user_prompt(idea: str) -> str:
    return (
        "Video idea:\n"
        f"{idea}\n\n"
        "Make a short scene plan suitable for Manim."
    )


MANIM_CODE_SYSTEM = """You write Manim Community Edition (ManimCE) python code.
Rules:
- Output ONLY python code (no markdown, no backticks).
- Define exactly: class GeneratedScene(Scene):
- No network calls, no reading external files.
- Use simple primitives: Text, Dot, Arrow, Axes, NumberPlane, ValueTracker, always_redraw.
- Keep runtime <= 20 seconds.
- Prefer Text over LaTeX (avoid MathTex unless necessary).
"""


def manim_code_user_prompt(scene_plan_json: str) -> str:
    return (
        "Create Manim code for this scene plan JSON:\n"
        f"{scene_plan_json}\n"
    )


REPAIR_SYSTEM = """You fix ManimCE python code.
Rules:
- Output ONLY python code (no markdown, no backticks).
- Define exactly: class GeneratedScene(Scene):
- No network calls, no reading external files.
- Keep runtime <= 20 seconds.
- Prefer Text over LaTeX (avoid MathTex unless necessary).
"""
