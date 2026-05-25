"""Popular item recall."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

BEHAVIOR_WEIGHTS = {"view": 1.0, "click": 3.0, "cart": 4.0, "fav": 4.0, "buy": 5.0}


def build_popular_recall(train_df: pd.DataFrame, users: Iterable[str] | None = None, top_k: int = 50) -> pd.DataFrame:
    """Recall globally popular items while excluding each user's history."""
    df = train_df.copy()
    df["_weight"] = df["behavior_type"].map(BEHAVIOR_WEIGHTS).fillna(1.0)
    popularity = (
        df.groupby("item_id")["_weight"]
        .sum()
        .sort_values(ascending=False)
        .reset_index(name="recall_score")
    )
    popular_items = popularity["item_id"].tolist()
    score_map = popularity.set_index("item_id")["recall_score"].to_dict()
    user_list = list(users) if users is not None else sorted(df["user_id"].unique())
    user_history = df.groupby("user_id")["item_id"].apply(set).to_dict()

    rows = []
    for user_id in user_list:
        history = user_history.get(user_id, set())
        count = 0
        for item_id in popular_items:
            if item_id in history:
                continue
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "recall_score": float(score_map[item_id]),
                    "recall_type": "popular",
                }
            )
            count += 1
            if count >= top_k:
                break
    return pd.DataFrame(rows)
