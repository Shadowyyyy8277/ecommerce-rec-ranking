from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse

from src.data.make_dataset import ensure_raw_dataset
from src.utils.io import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


def main() -> None:
    """Generate demo synthetic data."""
    parser = argparse.ArgumentParser(description="Generate synthetic e-commerce data.")
    parser.add_argument("--full", action="store_true", help="Generate full-scale data instead of quick demo data.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing raw interactions CSV.")
    args = parser.parse_args()

    config = load_config()
    set_global_seed(int(config["project"]["seed"]))
    scale = "full" if args.full else "quick"
    path = ensure_raw_dataset(config, scale=scale, force=args.force)
    logger.info("Raw data is ready: %s", path)


if __name__ == "__main__":
    main()
