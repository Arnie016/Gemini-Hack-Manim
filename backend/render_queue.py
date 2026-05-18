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
    path: Optional[Path] = None
    claimed_path: Optional[Path] = None
    key: str = ""
    claimed_key: str = ""


def render_mode() -> str:
    mode = (os.getenv("NORTHSTAR_RENDER_MODE") or "inline").strip().lower()
    return mode if mode in {"inline", "queue"} else "inline"


def queue_store() -> str:
    raw = (os.getenv("NORTHSTAR_QUEUE_STORE") or os.getenv("NORTHSTAR_ARTIFACT_STORE") or "local").strip().lower()
    if raw in {"s3", "r2", "lightsail"}:
        return "s3"
    return "local"


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


def _queue_bucket() -> str:
    bucket = (os.getenv("NORTHSTAR_QUEUE_BUCKET") or os.getenv("NORTHSTAR_ARTIFACT_BUCKET") or "").strip()
    if not bucket:
        raise ValueError("NORTHSTAR_QUEUE_BUCKET or NORTHSTAR_ARTIFACT_BUCKET is required for S3 queue mode.")
    return bucket


def _queue_prefix() -> str:
    return (os.getenv("NORTHSTAR_QUEUE_PREFIX") or "queue").strip("/ ") or "queue"


def _s3_client():
    try:
        import boto3  # type: ignore
    except ImportError as exc:
        raise RuntimeError("boto3 is required for S3 queue mode. Install requirements.txt.") from exc
    return boto3.client(
        "s3",
        region_name=(os.getenv("AWS_REGION") or "ap-southeast-1").strip() or "ap-southeast-1",
        endpoint_url=os.getenv("NORTHSTAR_ARTIFACT_ENDPOINT_URL") or None,
    )


def _s3_key(folder: str, filename: str) -> str:
    return f"{_queue_prefix()}/{folder}/{filename}"


def _s3_read_json(client, bucket: str, key: str) -> Dict[str, Any]:
    obj = client.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def _s3_put_json(client, bucket: str, key: str, payload: Dict[str, Any]) -> None:
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
        ContentType="application/json",
    )


def _s3_list_keys(client, bucket: str, folder: str) -> list[str]:
    prefix = _s3_key(folder, "")
    keys: list[str] = []
    token: Optional[str] = None
    while True:
        kwargs: Dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)
        for item in resp.get("Contents") or []:
            key = str(item.get("Key") or "")
            if key.endswith(".json"):
                keys.append(key)
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return sorted(keys)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def enqueue_render_job(payload: Dict[str, Any], *, root: Optional[Path] = None) -> Path:
    if queue_store() == "s3":
        client = _s3_client()
        bucket = _queue_bucket()
        job_id = str(payload.get("job_id") or "").strip()
        if not job_id:
            raise ValueError("Queue payload requires job_id.")
        data = dict(payload)
        data.update(
            {
                "queue_version": QUEUE_VERSION,
                "queue_store": "s3",
                "queued_at": data.get("queued_at") or time.time(),
                "attempts": int(data.get("attempts") or 0),
            }
        )
        filename = f"{int(float(data['queued_at']) * 1000)}-{_job_filename(job_id)}"
        key = _s3_key(PENDING, filename)
        _s3_put_json(client, bucket, key, data)
        return Path(key)

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
    if queue_store() == "s3":
        client = _s3_client()
        bucket = _queue_bucket()
        for key in _s3_list_keys(client, bucket, PENDING):
            try:
                payload = _s3_read_json(client, bucket, key)
            except Exception:
                failed_key = key.replace(f"/{PENDING}/", f"/{FAILED}/", 1)
                try:
                    client.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": key}, Key=failed_key)
                    client.delete_object(Bucket=bucket, Key=key)
                except Exception:
                    pass
                continue
            job_id = str(payload.get("job_id") or Path(key).stem)
            attempts = int(payload.get("attempts") or 0) + 1
            payload["attempts"] = attempts
            payload["claimed_at"] = time.time()
            payload["worker_pid"] = os.getpid()
            claimed_key = key.replace(f"/{PENDING}/", f"/{CLAIMED}/", 1).replace(".json", f".attempt-{attempts}.{os.getpid()}.json")
            try:
                _s3_put_json(client, bucket, claimed_key, payload)
                client.delete_object(Bucket=bucket, Key=key)
            except Exception:
                continue
            return QueueJob(job_id=job_id, payload=payload, key=key, claimed_key=claimed_key)
        return None

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
    if queue_store() == "s3":
        client = _s3_client()
        bucket = _queue_bucket()
        source = job.claimed_key or job.key
        payload = dict(job.payload)
        payload["finished_at"] = time.time()
        payload["queue_store"] = "s3"
        if error:
            payload["worker_error"] = error
        filename = Path(source).name if source else f"{_job_filename(job.job_id)}"
        dest = _s3_key(FAILED if failed else DONE, filename)
        _s3_put_json(client, bucket, dest, payload)
        if source:
            try:
                client.delete_object(Bucket=bucket, Key=source)
            except Exception:
                pass
        return Path(dest)

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


def queued_position(job_id: str, *, root: Optional[Path] = None) -> Optional[int]:
    if queue_store() == "s3":
        client = _s3_client()
        bucket = _queue_bucket()
        for idx, key in enumerate(_s3_list_keys(client, bucket, PENDING), start=1):
            if key.endswith(_job_filename(job_id)) or f"-{_job_filename(job_id)}" in key:
                return idx
            try:
                payload = _s3_read_json(client, bucket, key)
            except Exception:
                continue
            if str(payload.get("job_id") or "") == str(job_id):
                return idx
        return None
    base = ensure_queue(root)
    for idx, path in enumerate(sorted((base / PENDING).glob("*.json")), start=1):
        if path.name == _job_filename(job_id):
            return idx
    return None


def completed_job_payload(job_id: str) -> Optional[Dict[str, Any]]:
    if queue_store() != "s3":
        return None
    client = _s3_client()
    bucket = _queue_bucket()
    needle = _job_filename(job_id)
    for folder in (DONE, FAILED):
        for key in reversed(_s3_list_keys(client, bucket, folder)):
            if not (key.endswith(needle) or f"-{needle}" in key or str(job_id) in key):
                continue
            try:
                payload = _s3_read_json(client, bucket, key)
            except Exception:
                return None
            if str(payload.get("job_id") or "") == str(job_id):
                payload["queue_result_key"] = key
                return payload
    return None
