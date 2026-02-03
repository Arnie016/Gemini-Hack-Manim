from __future__ import annotations

SCENE_PLAN_SYSTEM = """You are a video director for Manim Community Edition.
Return ONLY valid JSON that matches the schema.
Constraints:
- Keep scenes coherent and focused (1 main idea per scene).
- Avoid overcrowding; follow max_objects if provided.
- Use clear, short narration text per scene.
- Include a strong hook, a clear core explanation, and a concise recap.
- Ensure total_seconds equals the sum of scene seconds.
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


def scene_plan_user_prompt(idea: str, director_brief: str = "") -> str:
    brief_block = f"\nDirector brief:\n{director_brief}\n" if director_brief else ""
    return (
        "Video idea:\n"
        f"{idea}\n"
        f"{brief_block}\n"
        "Make a scene plan suitable for Manim."
    )


MANIM_CODE_SYSTEM = """You write Manim Community Edition (ManimCE) python code.
Rules:
- Output ONLY python code (no markdown, no backticks).
- Define exactly: class GeneratedScene(Scene):
- No network calls, no reading external files.
- Use reliable primitives: Text, Dot, Arrow, Axes, NumberPlane, ValueTracker, always_redraw, Circle, Rectangle, Line, VGroup.
- You may use simple animations: FadeIn, FadeOut, Create, Write, Transform, LaggedStart.
- ImageMobject is allowed when assets are provided.
- Prefer Text over LaTeX (avoid MathTex unless necessary).
"""


def manim_code_user_prompt(
    scene_plan_json: str,
    assets_description: str = "",
    render_settings: str = "",
) -> str:
    assets_block = ""
    if assets_description:
        assets_block = f"\nAvailable assets:\n{assets_description}\n"
    settings_block = f"\nRender settings:\n{render_settings}\n" if render_settings else ""
    return (
        "Create Manim code for this scene plan JSON:\n"
        f"{scene_plan_json}\n"
        f"{assets_block}"
        f"{settings_block}"
        "Use assets only if provided. Keep image usage simple (background fill, small prop).\n"
    )


REPAIR_SYSTEM = """You fix ManimCE python code.
Rules:
- Output ONLY python code (no markdown, no backticks).
- Define exactly: class GeneratedScene(Scene):
- No network calls, no reading external files.
- Prefer Text over LaTeX (avoid MathTex unless necessary).
"""
