"""Popular item recall."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd

DEFAULT_BEHAVIOR_WEIGHTS: Dict[str, float] = {
    "view": 1.0,
    "click": 3.0,
    "cart": 4.0,
    "fav": 4.0,
    "buy": 5.0,
}
# Backward compatibility alias
BEHAVIOR_WEIGHTS = DEFAULT_BEHAVIOR_WEIGHTS


def build_popular_recall(
    train_df: pd.DataFrame,
    users: Optional[Iterable[str]] = None,
    top_k: int = 50,
    behavior_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Recall globally popular items while excluding each user's history.

    Parameters
    ----------
    train_df:
        DataFrame containing user interactions with columns `user_id`, `item_id`, `behavior_type`.
    users:
        Optional iterable of user IDs to generate recommendations for. Defaults to all unique users.
    top_k:
        Maximum number of candidate items to recall per user.
    behavior_weights:
        Optional dictionary mapping behavior types (view, click, cart, fav, buy) to numeric weights.
    """
    weights = behavior_weights or DEFAULT_BEHAVIOR_WEIGHTS
    weights_series = train_df["behavior_type"].map(weights).fillna(1.0)

    # Compute popularity score without modifying train_df
    popularity = (
        train_df.assign(_weight=weights_series)
        .groupby("item_id")["_weight"]
        .sum()
        .sort_values(ascending=False)
        .reset_index(name="recall_score")
    )
    popular_items = popularity["item_id"].tolist()
    score_map = popularity.set_index("item_id")["recall_score"].to_dict()

    user_list = list(users) if users is not None else sorted(train_df["user_id"].unique())
    user_history = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

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
