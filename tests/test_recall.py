from __future__ import annotations

import pandas as pd

from src.recall.item2vec_recall import build_item2vec_recall
from src.recall.itemcf_recall import build_itemcf_recall, build_itemcf_similarity
from src.recall.popular_recall import build_popular_recall


def _sample_interactions():
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "item_id": ["i1", "i2", "i2", "i3", "i1", "i3"],
            "behavior_type": ["buy", "click", "view", "cart", "buy", "buy"],
            "timestamp": [
                "2025-01-01 00:00:00",
                "2025-01-01 01:00:00",
                "2025-01-01 02:00:00",
                "2025-01-01 03:00:00",
                "2025-01-01 04:00:00",
                "2025-01-01 05:00:00",
            ],
        }
    )


def test_popular_recall_basic():
    df = _sample_interactions()
    popular = build_popular_recall(df, users=["u1", "u2"], top_k=2)

    assert not popular.empty
    assert set(popular["user_id"].unique()) == {"u1", "u2"}
    assert "recall_score" in popular.columns
    assert "recall_type" in popular.columns
    # Check that u1's history (i1, i2) is excluded
    u1_recs = popular[popular["user_id"] == "u1"]["item_id"].tolist()
    assert "i1" not in u1_recs
    assert "i2" not in u1_recs


def test_popular_recall_custom_weights():
    df = _sample_interactions()
    custom_weights = {"view": 100.0, "click": 1.0, "cart": 1.0, "fav": 1.0, "buy": 1.0}
    popular = build_popular_recall(df, users=["u1"], top_k=2, behavior_weights=custom_weights)
    # i2 had a view with weight 100, but u1 already interacted with i2. i3 has cart(1)+buy(1)=2.
    u1_items = popular[popular["user_id"] == "u1"]["item_id"].tolist()
    assert "i3" in u1_items


def test_itemcf_similarity_and_recall():
    df = _sample_interactions()
    sim = build_itemcf_similarity(df, top_k=5)
    assert isinstance(sim, dict)
    assert len(sim) > 0

    recs = build_itemcf_recall(df, users=["u1", "u2"], top_k=2)
    assert isinstance(recs, pd.DataFrame)
    assert "user_id" in recs.columns
    assert "item_id" in recs.columns


def test_item2vec_recall():
    df = _sample_interactions()
    recs = build_item2vec_recall(df, users=["u1"], top_k=2, embedding_dim=8, epochs=1, seed=42)
    assert isinstance(recs, pd.DataFrame)
    if not recs.empty:
        assert "user_id" in recs.columns
        assert "item_id" in recs.columns
        assert "recall_score" in recs.columns
