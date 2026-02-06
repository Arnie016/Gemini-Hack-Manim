from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
USER_ROOT = ROOT / "work" / "user_files"


def _ensure_root() -> None:
    USER_ROOT.mkdir(parents=True, exist_ok=True)


def _safe_path(rel_path: str) -> Path:
    if rel_path is None:
        raise ValueError("Path required")
    rel_path = str(rel_path).strip()
    if not rel_path:
        raise ValueError("Path required")
    rel = Path(rel_path)
    if rel.is_absolute():
        raise ValueError("Absolute paths are not allowed")
    candidate = (USER_ROOT / rel).resolve()
    if USER_ROOT != candidate and USER_ROOT not in candidate.parents:
        raise ValueError("Invalid path")
    return candidate


def list_tree() -> Dict[str, Any]:
    _ensure_root()

    def build(node: Path) -> Dict[str, Any]:
        rel = "" if node == USER_ROOT else str(node.relative_to(USER_ROOT))
        if node.is_dir():
            children = sorted(node.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            return {
                "type": "dir",
                "name": "workspace" if node == USER_ROOT else node.name,
                "path": rel,
                "children": [build(child) for child in children],
            }
        return {
            "type": "file",
            "name": node.name,
            "path": rel,
        }

    return build(USER_ROOT)


def create_folder(rel_path: str) -> str:
    _ensure_root()
    path = _safe_path(rel_path)
    # Keep folder semantics explicit in the UI/API: avoid names that look like files.
    if path.suffix:
        raise ValueError("Folder names cannot include a file extension (example: file.py)")
    path.mkdir(parents=True, exist_ok=True)
    return str(path.relative_to(USER_ROOT))


def write_file(rel_path: str, content: str, overwrite: bool = False) -> str:
    _ensure_root()
    path = _safe_path(rel_path)
    if path.exists() and not overwrite:
        raise FileExistsError("File already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content or "", encoding="utf-8")
    return str(path.relative_to(USER_ROOT))


def read_file(rel_path: str) -> str:
    _ensure_root()
    path = _safe_path(rel_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("File not found")
    return path.read_text(encoding="utf-8")


def delete_path(rel_path: str) -> None:
    _ensure_root()
    path = _safe_path(rel_path)
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_file():
                child.unlink()
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()
        path.rmdir()
    elif path.exists():
        path.unlink()
    else:
        raise FileNotFoundError("Path not found")


def rename_path(from_rel: str, to_rel: str, *, overwrite: bool = False) -> Dict[str, str]:
    """Rename (move) a file or folder inside USER_ROOT."""
    _ensure_root()
    src = _safe_path(from_rel)
    dst = _safe_path(to_rel)
    if not src.exists():
        raise FileNotFoundError("Source not found")
    if src.is_dir() and dst.suffix:
        raise ValueError("Folder names cannot include a file extension (example: file.py)")
    if dst.exists() and not overwrite:
        raise FileExistsError("Destination already exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    return {
        "from": str(Path(from_rel).as_posix()),
        "to": str(Path(to_rel).as_posix()),
    }
