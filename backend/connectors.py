from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List


CONNECTOR_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "google_drive",
        "name": "Google Drive",
        "category": "Sources",
        "context_types": ["docs", "slides", "pdfs", "folders"],
        "animation_use": "Turn lecture notes, cheat sheets, and slide decks into grounded scene plans.",
        "env_vars": ["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET"],
    },
    {
        "id": "google_docs",
        "name": "Google Docs",
        "category": "Sources",
        "context_types": ["documents", "outlines", "scripts"],
        "animation_use": "Extract precise definitions, equations, and story beats from living docs.",
        "env_vars": ["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET"],
    },
    {
        "id": "notion",
        "name": "Notion",
        "category": "Knowledge",
        "context_types": ["pages", "databases", "research notes"],
        "animation_use": "Use course notes, research databases, and prompt packs as animation memory.",
        "env_vars": ["NOTION_API_KEY"],
    },
    {
        "id": "github",
        "name": "GitHub",
        "category": "Code",
        "context_types": ["repos", "issues", "markdown", "examples"],
        "animation_use": "Create demos, technical explainers, and code-to-animation walkthroughs.",
        "env_vars": ["GITHUB_TOKEN"],
    },
    {
        "id": "slack",
        "name": "Slack",
        "category": "Teams",
        "context_types": ["threads", "launch notes", "feedback"],
        "animation_use": "Turn team discussions and feedback threads into short explainer videos.",
        "env_vars": ["SLACK_BOT_TOKEN"],
    },
    {
        "id": "gmail",
        "name": "Gmail",
        "category": "Sources",
        "context_types": ["emails", "attachments", "customer requests"],
        "animation_use": "Convert requests, study questions, or attachment context into animation briefs.",
        "env_vars": ["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET"],
    },
    {
        "id": "supabase",
        "name": "Supabase",
        "category": "Memory",
        "context_types": ["user memory", "render history", "projects"],
        "animation_use": "Store creator projects, source summaries, render metadata, and gallery memory.",
        "env_vars": ["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"],
    },
    {
        "id": "youtube",
        "name": "YouTube Data",
        "category": "Publishing",
        "context_types": ["uploads", "captions", "channel metadata"],
        "animation_use": "Publish finished Shorts/explainers and reuse channel metadata for descriptions.",
        "env_vars": ["YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET"],
    },
    {
        "id": "vimeo",
        "name": "Vimeo",
        "category": "Publishing",
        "context_types": ["video library", "review links"],
        "animation_use": "Publish or review finished animation exports with metadata attached.",
        "env_vars": ["VIMEO_ACCESS_TOKEN"],
    },
    {
        "id": "frameio",
        "name": "Frame.io",
        "category": "Review",
        "context_types": ["review comments", "video assets", "projects"],
        "animation_use": "Bring frame-accurate review notes back into the timeline and repair loop.",
        "env_vars": ["FRAMEIO_TOKEN"],
    },
    {
        "id": "openai_whisper",
        "name": "OpenAI Whisper",
        "category": "Transcription",
        "context_types": ["audio transcripts", "voice notes"],
        "animation_use": "Transcribe lectures or voice notes into grounded animation scripts.",
        "env_vars": ["OPENAI_API_KEY"],
    },
]


def _configured(env_vars: Iterable[str]) -> bool:
    return any(bool(os.getenv(name)) for name in env_vars)


def connector_catalog(enabled: Iterable[str] | None = None) -> List[Dict[str, Any]]:
    enabled_set = set(enabled or [])
    out: List[Dict[str, Any]] = []
    for item in CONNECTOR_CATALOG:
        row = dict(item)
        row["configured"] = _configured(row.get("env_vars") or [])
        row["enabled"] = row["id"] in enabled_set
        row["status"] = "ready" if row["configured"] and row["enabled"] else ("configured" if row["configured"] else "setup_needed")
        out.append(row)
    return out


def connector_context(enabled: Iterable[str] | None = None) -> Dict[str, Any]:
    selected = [c for c in connector_catalog(enabled) if c["enabled"]]
    lines = []
    for item in selected[:10]:
        lines.append(
            f"- {item['name']} ({item['category']}): {item['animation_use']} "
            f"Context types: {', '.join(item['context_types'])}. Status: {item['status']}."
        )
    return {
        "enabled": selected,
        "summary": "\n".join(lines),
        "prompt_contract": (
            "Use connected-context summaries as grounding only. Do not invent private data; "
            "ask the creator to attach or index exact files/messages before citing specifics."
        ),
    }
