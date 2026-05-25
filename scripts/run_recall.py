from __future__ import annotations

import _bootstrap  # noqa: F401

import pandas as pd

from src.recall.item2vec_recall import build_item2vec_recall
from src.recall.itemcf_recall import build_itemcf_recall
from src.recall.popular_recall import build_popular_recall
from src.utils.io import resolve_path, load_config
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


def main() -> None:
    """Run multi-channel recall and save recall candidate files."""
    config = load_config()
    set_global_seed(int(config["project"]["seed"]))
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    train_path = processed_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError("Processed train.csv not found. Run python scripts/run_data_pipeline.py first.")
    train_df = pd.read_csv(train_path)

    users = set(train_df["user_id"].unique())
    for split_name in ["valid", "test"]:
        split_path = processed_dir / f"{split_name}.csv"
        if split_path.exists():
            users.update(pd.read_csv(split_path, usecols=["user_id"])["user_id"].unique())
    users = sorted(users)

    recall_dir = resolve_path(config["paths"]["recall_dir"])
    recall_dir.mkdir(parents=True, exist_ok=True)
    top_k = int(config["recall"]["top_k"])

    logger.info("Running popular recall...")
    popular_df = build_popular_recall(train_df, users=users, top_k=top_k)
    popular_df.to_csv(recall_dir / "popular_recall.csv", index=False, encoding="utf-8")

    logger.info("Running ItemCF recall...")
    itemcf_df = build_itemcf_recall(train_df, users=users, top_k=int(config["recall"].get("itemcf_top_k", top_k)))
    itemcf_df.to_csv(recall_dir / "itemcf_recall.csv", index=False, encoding="utf-8")

    logger.info("Running Item2Vec recall...")
    item2vec_df = build_item2vec_recall(
        train_df,
        users=users,
        top_k=int(config["recall"].get("item2vec_top_k", top_k)),
        embedding_dim=int(config["recall"].get("embedding_dim", 32)),
        epochs=int(config["recall"].get("item2vec_epochs", 2)),
        seed=int(config["project"]["seed"]),
    )
    item2vec_df.to_csv(recall_dir / "item2vec_recall.csv", index=False, encoding="utf-8")

    merged = pd.concat([popular_df, itemcf_df, item2vec_df], ignore_index=True)
    merged.to_csv(recall_dir / "merged_recall.csv", index=False, encoding="utf-8")
    logger.info("Recall completed: popular=%s itemcf=%s item2vec=%s merged=%s", len(popular_df), len(itemcf_df), len(item2vec_df), len(merged))


if __name__ == "__main__":
    main()
