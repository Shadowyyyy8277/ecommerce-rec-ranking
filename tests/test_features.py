from __future__ import annotations

import pandas as pd

from src.features.build_features import create_feature_splits


def _mini_config():
    return {
        "split": {"train_ratio": 0.6, "valid_ratio": 0.2, "test_ratio": 0.2, "split_by_time": True},
        "project": {"seed": 42},
    }


def _mini_df():
    rows = []
    behaviors = ["view", "click", "view", "cart", "fav", "buy", "view", "click", "view", "buy"]
    for idx, behavior in enumerate(behaviors):
        rows.append(
            {
                "user_id": f"u_{idx % 3}",
                "item_id": f"i_{idx % 4}",
                "category_id": f"c_{idx % 2}",
                "brand_id": f"b_{idx % 2}",
                "behavior_type": behavior,
                "timestamp": f"2025-01-01 00:{idx:02d}:00",
                "item_price": 10.0 + idx,
                "user_age_bucket": "25-34",
                "user_gender": "M",
                "city_level": "tier_2",
            }
        )
    return pd.DataFrame(rows)


def test_features_have_required_columns():
    train, valid, test, feature_config = create_feature_splits(_mini_df(), _mini_config())
    required = {
        "user_total_views",
        "user_total_clicks",
        "user_total_buys",
        "item_total_views",
        "item_total_clicks",
        "category_ctr",
        "user_category_ctr",
        "hour",
        "day_of_week",
        "is_weekend",
        "price_bucket",
        "label",
    }
    assert required.issubset(train.columns)
    assert required.issubset(valid.columns)
    assert required.issubset(test.columns)
    assert feature_config["target_col"] == "label"


def test_behavior_type_not_model_feature_and_label_exists():
    train, _, _, feature_config = create_feature_splits(_mini_df(), _mini_config())
    assert "label" in train.columns
    assert "behavior_type" not in feature_config["all_features"]
    assert "label" not in feature_config["all_features"]
