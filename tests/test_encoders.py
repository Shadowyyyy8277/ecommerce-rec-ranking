from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.encoders import (
    build_category_encoders,
    fit_dense_scaler,
    transform_deepfm_features,
    transform_sparse_dense_dataframe,
)


def _sample_feature_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "item_id": ["i1", "i2", "i3"],
            "city_level": ["tier_1", "tier_2", "tier_1"],
            "price": [10.0, 20.0, 30.0],
            "ctr": [0.1, 0.2, 0.3],
        }
    )


def test_build_category_encoders():
    df = _sample_feature_df()
    sparse_cols = ["city_level", "user_id"]
    encoders = build_category_encoders(df, sparse_cols)

    assert "city_level" in encoders
    assert "user_id" in encoders
    assert encoders["city_level"]["tier_1"] == 1
    assert encoders["city_level"]["tier_2"] == 2


def test_transform_sparse_dense_dataframe():
    df = _sample_feature_df()
    sparse_cols = ["city_level"]
    dense_cols = ["price", "ctr"]
    encoders = build_category_encoders(df, sparse_cols)

    transformed = transform_sparse_dense_dataframe(df, sparse_cols, dense_cols, encoders)
    assert "city_level" in transformed.columns
    assert "price" in transformed.columns
    assert "ctr" in transformed.columns
    assert transformed["city_level"].iloc[0] == 1.0

    # Test unknown category mapping to 0
    test_df = pd.DataFrame({"city_level": ["tier_99"], "price": [50.0], "ctr": [0.5]})
    transformed_test = transform_sparse_dense_dataframe(test_df, sparse_cols, dense_cols, encoders)
    assert transformed_test["city_level"].iloc[0] == 0.0


def test_fit_dense_scaler_and_deepfm_transform():
    df = _sample_feature_df()
    sparse_cols = ["city_level"]
    dense_cols = ["price", "ctr"]

    scaler = fit_dense_scaler(df, dense_cols)
    assert len(scaler["mean"]) == 2
    assert len(scaler["std"]) == 2

    encoders = build_category_encoders(df, sparse_cols)
    sparse_x, dense_x = transform_deepfm_features(df, sparse_cols, dense_cols, encoders, scaler)

    assert sparse_x.shape == (3, 1)
    assert dense_x.shape == (3, 2)
    assert np.allclose(dense_x.mean(axis=0), 0.0, atol=1e-5)
