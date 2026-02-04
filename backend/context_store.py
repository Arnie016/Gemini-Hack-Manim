from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import re
import secrets
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
MEMORY_PATH = WORK / "memories.json"
SKILLS_DIR = WORK / "skills"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\\s-]", "", text)
    text = re.sub(r"\\s+", "-", text)
    return text or "skill"


def list_memories() -> List[Dict[str, Any]]:
    return _load_json(MEMORY_PATH, [])


def add_memory(title: str, content: str) -> Dict[str, Any]:
    memories = list_memories()
    entry = {
        "id": f"{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}",
        "title": title.strip(),
        "content": content.strip(),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    memories.append(entry)
    _save_json(MEMORY_PATH, memories)
    return entry


def delete_memory(memory_id: str) -> bool:
    memories = list_memories()
    new_memories = [m for m in memories if m.get("id") != memory_id]
    if len(new_memories) == len(memories):
        return False
    _save_json(MEMORY_PATH, new_memories)
    return True


def get_memories_by_ids(ids: Optional[List[str]]) -> List[Dict[str, Any]]:
    if not ids:
        return []
    lookup = set(ids)
    return [m for m in list_memories() if m.get("id") in lookup]


def list_skills() -> List[Dict[str, Any]]:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for path in sorted(SKILLS_DIR.glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        name = lines[0].lstrip("# ").strip() if lines else path.stem
        items.append({"id": path.stem, "name": name, "path": str(path)})
    return items


def save_skill(name: str, content: str, *, skill_id: Optional[str] = None) -> Dict[str, Any]:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    slug = skill_id or _slugify(name)
    filename = f"{slug}.md"
    path = SKILLS_DIR / filename
    if not content.strip():
        content = "Add skill instructions here."
    text = f"# {name.strip()}\\n\\n{content.strip()}\\n"
    path.write_text(text, encoding="utf-8")
    return {"id": slug, "name": name.strip(), "path": str(path)}


def read_skill(skill_id: str) -> Optional[str]:
    path = SKILLS_DIR / f"{skill_id}.md"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def delete_skill(skill_id: str) -> bool:
    path = SKILLS_DIR / f"{skill_id}.md"
    if not path.exists():
        return False
    path.unlink()
    return True


def get_skills_by_ids(ids: Optional[List[str]]) -> List[Dict[str, Any]]:
    if not ids:
        return []
    lookup = set(ids)
    skills = []
    for item in list_skills():
        if item["id"] in lookup:
            content = read_skill(item["id"]) or ""
            skills.append({**item, "content": content})
    return skills
