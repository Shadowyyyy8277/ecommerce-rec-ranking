from __future__ import annotations

from pathlib import Path
import pandas as pd
import torch

from src.models.deepfm import DeepFM
from src.models.train_deepfm import train_deepfm_model


def _synthetic_mini_train_valid():
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"] * 10,
            "item_id": ["i1", "i2", "i3", "i4"] * 10,
            "category_id": ["c1", "c2"] * 20,
            "brand_id": ["b1", "b2"] * 20,
            "city_level": ["tier_1", "tier_2"] * 20,
            "price": [10.0, 20.0, 30.0, 40.0] * 10,
            "ctr": [0.1, 0.2, 0.3, 0.4] * 10,
            "label": [0, 1, 0, 1] * 10,
        }
    )
    feature_config = {
        "sparse_features": ["user_id", "item_id", "category_id", "brand_id", "city_level"],
        "dense_features": ["price", "ctr"],
        "target_col": "label",
    }
    config = {
        "project": {"seed": 42},
        "training": {
            "deepfm": {
                "epochs": 2,
                "batch_size": 16,
                "embedding_dim": 8,
                "lr": 0.01,
                "lr_scheduler": "cosine",
            }
        },
    }
    return df, df.copy(), feature_config, config


def test_deepfm_training_with_lr_scheduler(tmp_path):
    train_df, valid_df, feature_config, config = _synthetic_mini_train_valid()
    model_path = tmp_path / "deepfm_test.pt"

    trained_path, metrics = train_deepfm_model(train_df, valid_df, feature_config, config, model_path)
    assert Path(trained_path).exists()
    assert "auc" in metrics
    assert "logloss" in metrics
