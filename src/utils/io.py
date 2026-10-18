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

import functools
import os

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


def to_relative_path(path_like: str | Path) -> str:
    """Convert a path to a forward-slash relative path string from project root."""
    path = Path(path_like)
    try:
        rel = path.resolve().relative_to(PROJECT_ROOT.resolve())
        return rel.as_posix()
    except ValueError:
        return path.as_posix()


def ensure_dir(path_like: str | Path) -> Path:
    """Create a directory if it does not exist and return its Path."""
    path = resolve_path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


@functools.lru_cache(maxsize=16)
def _load_yaml_cached(resolved_str_path: str) -> Dict[str, Any]:
    with open(resolved_str_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clear_config_cache() -> None:
    """Clear cached configuration files."""
    _load_yaml_cached.cache_clear()


def load_config(config_path: str | Path | None = None, use_cache: bool = True) -> Dict[str, Any]:
    """Load YAML configuration with caching and environment variable override support."""
    if config_path is None:
        config_path = os.environ.get("REC_CONFIG_PATH", "configs/config.yaml")
    path = resolve_path(config_path)
    if use_cache:
        # Return a shallow copy of dict so caller modifications don't mutate cache
        cached = _load_yaml_cached(str(path.resolve()))
        return yaml.safe_load(yaml.dump(cached))
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
