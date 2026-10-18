from __future__ import annotations

import _bootstrap  # noqa: F401

import concurrent.futures
from typing import List

import pandas as pd

from src.recall.item2vec_recall import build_item2vec_recall
from src.recall.itemcf_recall import build_itemcf_recall
from src.recall.popular_recall import build_popular_recall
from src.utils.io import load_config, resolve_path
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


def main() -> None:
    """Run multi-channel recall and save recall candidate files.

    Supports concurrent channel execution (Popular, ItemCF, Item2Vec) via ThreadPoolExecutor.
    """
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
    user_list = sorted(users)

    recall_dir = resolve_path(config["paths"]["recall_dir"])
    recall_dir.mkdir(parents=True, exist_ok=True)
    top_k = int(config["recall"]["top_k"])

    def _run_popular() -> pd.DataFrame:
        logger.info("Running Popular recall channel...")
        df = build_popular_recall(train_df, users=user_list, top_k=top_k)
        df.to_csv(recall_dir / "popular_recall.csv", index=False, encoding="utf-8")
        logger.info("Popular recall finished with %s candidates.", len(df))
        return df

    def _run_itemcf() -> pd.DataFrame:
        logger.info("Running ItemCF recall channel...")
        df = build_itemcf_recall(
            train_df,
            users=user_list,
            top_k=int(config["recall"].get("itemcf_top_k", top_k)),
        )
        df.to_csv(recall_dir / "itemcf_recall.csv", index=False, encoding="utf-8")
        logger.info("ItemCF recall finished with %s candidates.", len(df))
        return df

    def _run_item2vec() -> pd.DataFrame:
        logger.info("Running Item2Vec recall channel...")
        df = build_item2vec_recall(
            train_df,
            users=user_list,
            top_k=int(config["recall"].get("item2vec_top_k", top_k)),
            embedding_dim=int(config["recall"].get("embedding_dim", 32)),
            epochs=int(config["recall"].get("item2vec_epochs", 2)),
            seed=int(config["project"]["seed"]),
        )
        df.to_csv(recall_dir / "item2vec_recall.csv", index=False, encoding="utf-8")
        logger.info("Item2Vec recall finished with %s candidates.", len(df))
        return df

    parallel = bool(config["recall"].get("parallel", True))
    if parallel:
        logger.info("Executing recall channels concurrently in thread pool...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            fut_pop = executor.submit(_run_popular)
            fut_cf = executor.submit(_run_itemcf)
            fut_i2v = executor.submit(_run_item2vec)
            popular_df = fut_pop.result()
            itemcf_df = fut_cf.result()
            item2vec_df = fut_i2v.result()
    else:
        logger.info("Executing recall channels sequentially...")
        popular_df = _run_popular()
        itemcf_df = _run_itemcf()
        item2vec_df = _run_item2vec()

    merged = pd.concat([popular_df, itemcf_df, item2vec_df], ignore_index=True)
    # Deduplicate keeping highest recall_score per (user_id, item_id)
    merged = (
        merged.sort_values(["user_id", "recall_score"], ascending=[True, False])
        .drop_duplicates(subset=["user_id", "item_id"], keep="first")
        .reset_index(drop=True)
    )
    merged.to_csv(recall_dir / "merged_recall.csv", index=False, encoding="utf-8")
    logger.info(
        "All recall channels completed: popular=%s, itemcf=%s, item2vec=%s, merged (deduped)=%s",
        len(popular_df),
        len(itemcf_df),
        len(item2vec_df),
        len(merged),
    )


if __name__ == "__main__":
    main()
