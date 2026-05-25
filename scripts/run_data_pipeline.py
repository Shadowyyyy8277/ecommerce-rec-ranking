from __future__ import annotations

import _bootstrap  # noqa: F401

from src.data.make_dataset import ensure_raw_dataset
from src.features.build_features import build_and_save_datasets
from src.utils.io import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


def main() -> None:
    """Run raw data generation when needed and build processed features."""
    config = load_config()
    set_global_seed(int(config["project"]["seed"]))
    ensure_raw_dataset(config, scale="quick" if config["project"].get("quick_mode", True) else "full", force=False)
    feature_config = build_and_save_datasets(config)
    logger.info("Feature pipeline completed with %s model features.", len(feature_config["all_features"]))


if __name__ == "__main__":
    main()
