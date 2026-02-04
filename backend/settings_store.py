from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
CONFIG_PATH = WORK / "config.json"


def load_settings() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_settings(settings: Dict[str, Any]) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def update_settings(patch: Dict[str, Any]) -> Dict[str, Any]:
    settings = load_settings()
    settings.update({k: v for k, v in patch.items() if v is not None})
    save_settings(settings)
    return settings
