from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional


@dataclass
class PublishedArtifact:
    key: str
    url: str
    bytes: int


class ArtifactStore:
    name = "local"

    def publish_files(self, *, job_id: str, files: Mapping[str, Path]) -> Dict[str, PublishedArtifact]:
        return {}


class LocalArtifactStore(ArtifactStore):
    name = "local"


class S3ArtifactStore(ArtifactStore):
    name = "s3"

    def __init__(
        self,
        *,
        bucket: str,
        region: str,
        prefix: str = "renders",
        endpoint_url: Optional[str] = None,
        public_base_url: Optional[str] = None,
    ) -> None:
        if not bucket:
            raise ValueError("NORTHSTAR_ARTIFACT_BUCKET is required for S3 artifact storage.")
        try:
            import boto3  # type: ignore
        except ImportError as exc:
            raise RuntimeError("boto3 is required for S3 artifact storage. Install requirements.txt.") from exc

        self.bucket = bucket
        self.region = region
        self.prefix = prefix.strip("/ ") or "renders"
        self.public_base_url = (public_base_url or "").rstrip("/")
        self.client = boto3.client("s3", region_name=region, endpoint_url=endpoint_url or None)

    def _key(self, job_id: str, local_path: Path) -> str:
        return f"{self.prefix}/{job_id}/{local_path.name}"

    def _url(self, key: str) -> str:
        if self.public_base_url:
            return f"{self.public_base_url}/{key}"
        return f"s3://{self.bucket}/{key}"

    def publish_files(self, *, job_id: str, files: Mapping[str, Path]) -> Dict[str, PublishedArtifact]:
        published: Dict[str, PublishedArtifact] = {}
        for label, path in files.items():
            if not path.exists() or not path.is_file():
                continue
            key = self._key(job_id, path)
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.client.upload_file(
                str(path),
                self.bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )
            published[label] = PublishedArtifact(key=key, url=self._url(key), bytes=path.stat().st_size)
        return published


def artifact_store_from_env() -> ArtifactStore:
    provider = (os.getenv("NORTHSTAR_ARTIFACT_STORE") or "local").strip().lower()
    if provider in {"", "local", "filesystem", "fs"}:
        return LocalArtifactStore()
    if provider in {"s3", "r2", "lightsail"}:
        return S3ArtifactStore(
            bucket=os.getenv("NORTHSTAR_ARTIFACT_BUCKET", "").strip(),
            region=os.getenv("AWS_REGION", "ap-southeast-1").strip() or "ap-southeast-1",
            prefix=os.getenv("NORTHSTAR_ARTIFACT_PREFIX", "renders"),
            endpoint_url=os.getenv("NORTHSTAR_ARTIFACT_ENDPOINT_URL"),
            public_base_url=os.getenv("NORTHSTAR_ARTIFACT_PUBLIC_BASE_URL"),
        )
    raise ValueError(f"Unsupported NORTHSTAR_ARTIFACT_STORE: {provider}")
