from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import secrets


@dataclass
class JobPaths:
    job_id: str
    job_dir: Path
    scene_path: Path
    plan_path: Path
    out_mp4: Path
    logs_path: Path


_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def validate_job_id(job_id: str) -> str:
    """Validate a job id before using it as a directory name."""
    value = str(job_id or "").strip()
    if not _JOB_ID_RE.fullmatch(value):
        raise ValueError("Invalid job_id")
    return value


def new_job_id() -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    token = secrets.token_hex(3)
    return f"{stamp}-{token}"


def job_paths(root: Path, job_id: str) -> JobPaths:
    job_id = validate_job_id(job_id)
    root = root.resolve()
    job_dir = (root / job_id).resolve()
    if root != job_dir and root not in job_dir.parents:
        raise ValueError("Invalid job_id")
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
