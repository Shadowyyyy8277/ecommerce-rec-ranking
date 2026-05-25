"""Dataset orchestration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal

import pandas as pd

from src.data.generate_synthetic_data import SyntheticDataSpec, generate_synthetic_data, save_synthetic_data
from src.utils.io import resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def ensure_raw_dataset(
    config: Dict[str, Any],
    scale: Literal["quick", "full"] | None = None,
    force: bool = False,
) -> Path:
    """Ensure the raw interactions CSV exists, generating it when needed.

    Parameters
    ----------
    config:
        Project configuration dictionary.
    scale:
        Dataset scale. If omitted, follows project.quick_mode.
    force:
        Regenerate even when the raw file already exists.
    """
    raw_path = resolve_path(config["paths"]["raw_data"])
    if raw_path.exists() and not force:
        logger.info("Raw dataset already exists: %s", raw_path)
        return raw_path

    chosen_scale = scale or ("quick" if config["project"].get("quick_mode", True) else "full")
    data_cfg = config["data"][chosen_scale]
    spec = SyntheticDataSpec(
        n_users=int(data_cfg["n_users"]),
        n_items=int(data_cfg["n_items"]),
        n_interactions=int(data_cfg["n_interactions"]),
        n_categories=int(config["data"]["n_categories"]),
        n_brands=int(config["data"]["n_brands"]),
        seed=int(config["project"]["seed"]),
    )
    df = generate_synthetic_data(spec)
    return save_synthetic_data(df, raw_path)


def load_raw_interactions(config: Dict[str, Any]) -> pd.DataFrame:
    """Load raw interactions from the configured CSV path."""
    raw_path = resolve_path(config["paths"]["raw_data"])
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}. Run python scripts/run_demo_data.py first.")
    return pd.read_csv(raw_path)
