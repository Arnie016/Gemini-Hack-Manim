from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import secrets


@dataclass
class JobPaths:
    job_id: str
    job_dir: Path
    scene_path: Path
    plan_path: Path
    out_mp4: Path
    logs_path: Path


def new_job_id() -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    token = secrets.token_hex(3)
    return f"{stamp}-{token}"


def job_paths(root: Path, job_id: str) -> JobPaths:
    job_dir = root / job_id
    scene_path = job_dir / "scene.py"
    plan_path = job_dir / "plan.json"
    out_mp4 = job_dir / "out.mp4"
    logs_path = job_dir / "logs.txt"
    return JobPaths(
        job_id=job_id,
        job_dir=job_dir,
        scene_path=scene_path,
        plan_path=plan_path,
        out_mp4=out_mp4,
        logs_path=logs_path,
    )
