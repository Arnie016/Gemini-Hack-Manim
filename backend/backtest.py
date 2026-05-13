from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .renderer import render_with_manim
from .science_concepts import SCIENCE_CONCEPTS


@dataclass
class BacktestResult:
    topic: str
    family: str
    ok: bool
    score: int
    code_path: str
    video_path: str
    logs_path: str
    duration: float = 0.0
    size_bytes: int = 0
    issues: List[str] | None = None


def _safe_slug(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")[:80] or "concept"


def _clip_text(text: str, limit: int = 58) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def deterministic_scene_code(concept: Dict[str, str], *, seconds: float = 12.0, aspect_ratio: str = "9:16") -> str:
    topic = _clip_text(concept["topic"], 48)
    family = _clip_text(concept.get("family", "Science"), 32)
    prompt = _clip_text(concept.get("prompt", ""), 68)
    equation = _clip_text(concept.get("equation", ""), 52)
    vertical = aspect_ratio == "9:16"
    width = 7 if vertical else 11
    step_time = max(0.45, min(1.4, seconds / 9))
    wait_time = max(0.15, min(0.8, seconds / 24))
    return f'''from manim import *


class GeneratedScene(Scene):
    def construct(self):
        self.camera.background_color = "#070B12"
        title = Text({topic!r}, font_size=38, weight=BOLD)
        family = Text({family!r}, font_size=22, color=BLUE_B)
        family.next_to(title, UP, buff=0.28)
        hook = Text({prompt!r}, font_size=24, color=GRAY_A, line_spacing=0.85).scale(0.82)
        hook.set_width({width * 0.82})
        hook.next_to(title, DOWN, buff=0.45)
        eq = Text({equation!r}, font_size=30, color=GREEN_B)
        eq.next_to(hook, DOWN, buff=0.55)

        axis = NumberLine(x_range=[-3, 3, 1], length={width * 0.72}, color=GRAY_B)
        axis.shift(DOWN * 2.15)
        dot = Dot(axis.n2p(-2.7), color=BLUE_C)
        path = Line(axis.n2p(-2.7), axis.n2p(2.7), color=BLUE_E).set_stroke(width=3, opacity=0.55)

        phases = VGroup(
            Text("hook", font_size=18, color=GRAY_B),
            Text("mechanism", font_size=18, color=GRAY_B),
            Text("rule", font_size=18, color=GRAY_B),
            Text("takeaway", font_size=18, color=GRAY_B),
        ).arrange(RIGHT, buff=0.35)
        phases.to_edge(DOWN, buff=0.42)

        self.play(FadeIn(family), Write(title), run_time={step_time:.2f})
        self.play(FadeIn(hook, shift=UP * 0.18), run_time={step_time:.2f})
        self.play(Create(axis), Create(path), FadeIn(dot), run_time={step_time:.2f})
        self.play(dot.animate.move_to(axis.n2p(2.7)), run_time={max(1.0, seconds * 0.22):.2f}, rate_func=smooth)
        self.play(FadeIn(eq, shift=UP * 0.12), run_time={step_time:.2f})
        self.play(FadeIn(phases), run_time={step_time:.2f})
        self.wait({wait_time:.2f})
'''


def probe_mp4(path: Path) -> Dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe or not path.exists():
        return {}
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=width,height,codec_name",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        return {}
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {}


def score_video(path: Path, logs: str, *, target_seconds: float) -> tuple[int, float, int, list[str]]:
    issues: list[str] = []
    probe = probe_mp4(path)
    fmt = probe.get("format") if isinstance(probe, dict) else {}
    streams = probe.get("streams") if isinstance(probe, dict) else []
    try:
        duration = float(fmt.get("duration") or 0)
    except Exception:
        duration = 0.0
    try:
        size_bytes = int(fmt.get("size") or (path.stat().st_size if path.exists() else 0))
    except Exception:
        size_bytes = 0

    score = 100
    if not path.exists():
        issues.append("missing_mp4")
        score -= 80
    if duration < max(1.0, target_seconds * 0.35):
        issues.append(f"duration_too_short:{duration:.2f}s")
        score -= 25
    if duration > target_seconds * 1.8:
        issues.append(f"duration_too_long:{duration:.2f}s")
        score -= 10
    if size_bytes < 4096:
        issues.append(f"mp4_too_small:{size_bytes}B")
        score -= 25
    video_stream = next((s for s in streams if s.get("width") and s.get("height")), {})
    if not video_stream:
        issues.append("missing_video_stream")
        score -= 35
    if "Traceback" in logs or "Error" in logs:
        issues.append("render_log_error")
        score -= 20
    if "Rendered GeneratedScene" not in logs:
        issues.append("missing_manim_success_marker")
        score -= 10
    return max(0, score), duration, size_bytes, issues


def iter_concepts(limit: Optional[int] = None, family: Optional[str] = None) -> Iterable[Dict[str, str]]:
    count = 0
    for concept in SCIENCE_CONCEPTS:
        if family and concept.get("family", "").lower() != family.lower():
            continue
        yield concept
        count += 1
        if limit and count >= limit:
            return


def run_local_backtest(
    *,
    out_dir: Path,
    limit: int = 10,
    family: Optional[str] = None,
    render: bool = False,
    manim_py: Optional[str] = None,
    quality: str = "pql",
    seconds: float = 12.0,
    aspect_ratio: str = "9:16",
) -> Dict[str, Any]:
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[BacktestResult] = []
    for idx, concept in enumerate(iter_concepts(limit=limit, family=family), start=1):
        slug = f"{idx:03d}-{_safe_slug(concept['topic'])}"
        job_dir = out_dir / slug
        job_dir.mkdir(parents=True, exist_ok=True)
        code_path = job_dir / "scene.py"
        logs_path = job_dir / "logs.txt"
        video_path = job_dir / "out.mp4"
        code_path.write_text(
            deterministic_scene_code(concept, seconds=seconds, aspect_ratio=aspect_ratio),
            encoding="utf-8",
        )
        ok = True
        logs = "dry run"
        if render:
            ok, logs = render_with_manim(
                code_path,
                video_path,
                manim_py=manim_py,
                quality=quality,
                timeout_s=max(90, int(seconds * 18)),
            )
        logs_path.write_text(logs, encoding="utf-8")
        score, duration, size_bytes, issues = score_video(video_path, logs, target_seconds=seconds) if render else (0, 0.0, 0, ["not_rendered"])
        results.append(
            BacktestResult(
                topic=concept["topic"],
                family=concept.get("family", ""),
                ok=bool(ok and (not render or score >= 70)),
                score=score,
                code_path=str(code_path),
                video_path=str(video_path),
                logs_path=str(logs_path),
                duration=duration,
                size_bytes=size_bytes,
                issues=issues,
            )
        )

    passed = sum(1 for r in results if r.ok)
    report = {
        "ok": passed == len(results) if results else False,
        "mode": "local_manim_render" if render else "code_generation_only",
        "concept_source": "backend.science_concepts.SCIENCE_CONCEPTS",
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "average_score": round(sum(r.score for r in results) / len(results), 1) if render and results else None,
        "elapsed_seconds": round(time.time() - started, 2),
        "out_dir": str(out_dir),
        "results": [asdict(r) for r in results],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# NorthStar Science Backtest",
        "",
        f"- Mode: `{report['mode']}`",
        f"- Total: {report['total']}",
        f"- Passed: {report['passed']}",
        f"- Failed: {report['failed']}",
        f"- Output: `{out_dir}`",
        "",
        "| Score | Status | Concept | Issues |",
        "|---:|---|---|---|",
    ]
    for result in results:
        status = "pass" if result.ok else "fail"
        issues = ", ".join(result.issues or [])
        lines.append(f"| {result.score} | {status} | {result.topic} | {issues} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
