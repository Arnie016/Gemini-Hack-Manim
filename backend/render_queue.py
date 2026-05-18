from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[1]
QUEUE_VERSION = 1
DEFAULT_QUEUE_ROOT = ROOT / "work" / "queue"
PENDING = "pending"
CLAIMED = "claimed"
DONE = "done"
FAILED = "failed"


@dataclass
class QueueJob:
    job_id: str
    payload: Dict[str, Any]
    path: Path
    claimed_path: Optional[Path] = None


def render_mode() -> str:
    mode = (os.getenv("NORTHSTAR_RENDER_MODE") or "inline").strip().lower()
    return mode if mode in {"inline", "queue"} else "inline"


def queue_root(root: Optional[Path] = None) -> Path:
    return (root or DEFAULT_QUEUE_ROOT).resolve()


def ensure_queue(root: Optional[Path] = None) -> Path:
    base = queue_root(root)
    for name in (PENDING, CLAIMED, DONE, FAILED):
        (base / name).mkdir(parents=True, exist_ok=True)
    return base


def _job_filename(job_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(job_id))
    return f"{safe}.json"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def enqueue_render_job(payload: Dict[str, Any], *, root: Optional[Path] = None) -> Path:
    base = ensure_queue(root)
    job_id = str(payload.get("job_id") or "").strip()
    if not job_id:
        raise ValueError("Queue payload requires job_id.")
    data = dict(payload)
    data.update(
        {
            "queue_version": QUEUE_VERSION,
            "queued_at": data.get("queued_at") or time.time(),
            "attempts": int(data.get("attempts") or 0),
        }
    )
    final_path = base / PENDING / _job_filename(job_id)
    tmp_path = final_path.with_suffix(f".{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, final_path)
    return final_path


def claim_next_job(*, root: Optional[Path] = None) -> Optional[QueueJob]:
    base = ensure_queue(root)
    for path in sorted((base / PENDING).glob("*.json")):
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            failed_path = base / FAILED / path.name
            try:
                os.replace(path, failed_path)
            except OSError:
                pass
            continue
        job_id = str(payload.get("job_id") or path.stem)
        attempts = int(payload.get("attempts") or 0) + 1
        payload["attempts"] = attempts
        payload["claimed_at"] = time.time()
        payload["worker_pid"] = os.getpid()
        claimed_path = base / CLAIMED / f"{path.stem}.attempt-{attempts}.{os.getpid()}.json"
        try:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(path, claimed_path)
        except OSError:
            continue
        return QueueJob(job_id=job_id, payload=payload, path=path, claimed_path=claimed_path)
    return None


def complete_job(job: QueueJob, *, failed: bool = False, error: str = "", root: Optional[Path] = None) -> Path:
    base = ensure_queue(root)
    source = job.claimed_path or job.path
    payload = dict(job.payload)
    payload["finished_at"] = time.time()
    if error:
        payload["worker_error"] = error
    dest_dir = base / (FAILED if failed else DONE)
    dest = dest_dir / f"{source.stem}.json"
    tmp = source.with_suffix(".complete.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, source)
        os.replace(source, dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return dest
