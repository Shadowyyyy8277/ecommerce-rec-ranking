"""Path and file IO utilities.

All project paths are resolved relative to the repository root unless an
absolute path is supplied. This keeps the project portable on Windows, macOS,
and Linux.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    """Return the absolute project root directory."""
    return PROJECT_ROOT


def resolve_path(path_like: str | Path) -> Path:
    """Resolve a config path relative to the project root.

    Parameters
    ----------
    path_like:
        Relative or absolute filesystem path.
    """
    path = Path(path_like)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def ensure_dir(path_like: str | Path) -> Path:
    """Create a directory if it does not exist and return its Path."""
    path = resolve_path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(config_path: str | Path = "configs/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration from the project config file."""
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path_like: str | Path) -> Dict[str, Any]:
    """Read a JSON file using UTF-8 encoding."""
    path = resolve_path(path_like)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Dict[str, Any], path_like: str | Path) -> Path:
    """Write a dictionary to JSON with stable formatting."""
    path = resolve_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path
