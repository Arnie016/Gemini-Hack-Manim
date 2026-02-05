from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class JobState:
    job_id: str
    status: str  # created|planned|running|repairing|done|failed
    step: str  # plan|images|code|render|repair|idle
    message: str = ""
    updated_at: float = 0.0
    video_path: str = ""
    plan_path: str = ""
    scene_path: str = ""
    logs_path: str = ""
    error: str = ""


def state_path(job_dir: Path) -> Path:
    return job_dir / "state.json"


def events_path(job_dir: Path) -> Path:
    return job_dir / "events.log"


def load_state(job_dir: Path, job_id: str) -> JobState:
    path = state_path(job_dir)
    if not path.exists():
        return JobState(job_id=job_id, status="created", step="idle", updated_at=time.time())
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return JobState(job_id=job_id, status="created", step="idle", updated_at=time.time())
    return JobState(
        job_id=job_id,
        status=str(data.get("status") or "created"),
        step=str(data.get("step") or "idle"),
        message=str(data.get("message") or ""),
        updated_at=float(data.get("updated_at") or time.time()),
        video_path=str(data.get("video_path") or ""),
        plan_path=str(data.get("plan_path") or ""),
        scene_path=str(data.get("scene_path") or ""),
        logs_path=str(data.get("logs_path") or ""),
        error=str(data.get("error") or ""),
    )


def write_state(job_dir: Path, state: JobState) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    path = state_path(job_dir)
    path.write_text(
        json.dumps(
            {
                "job_id": state.job_id,
                "status": state.status,
                "step": state.step,
                "message": state.message,
                "updated_at": state.updated_at,
                "video_path": state.video_path,
                "plan_path": state.plan_path,
                "scene_path": state.scene_path,
                "logs_path": state.logs_path,
                "error": state.error,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def append_event(job_dir: Path, *, type_: str, payload: Optional[Dict[str, Any]] = None) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    ev = {
        "ts": time.time(),
        "type": type_,
        "payload": payload or {},
    }
    with events_path(job_dir).open("a", encoding="utf-8") as f:
        f.write(json.dumps(ev) + "\n")

