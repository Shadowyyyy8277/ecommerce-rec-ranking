"""LightGBM CTR ranking model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd

from src.evaluation.metrics import binary_classification_metrics


def train_lightgbm_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_config: Dict[str, Any],
    config: Dict[str, Any],
    model_path: str | Path,
) -> Tuple[Path, Dict[str, float]]:
    """Train a LightGBM classifier with integer-encoded categorical features.

    Integer encoding avoids very large one-hot matrices and keeps the CPU demo lightweight.
    """
    from lightgbm import LGBMClassifier

    sparse_features = feature_config["sparse_features"]
    dense_features = feature_config["dense_features"]
    target_col = feature_config["target_col"]
    lgb_cfg = config["training"]["lightgbm"]

    category_maps = _fit_category_maps(train_df, sparse_features)
    x_train = _transform_features(train_df, sparse_features, dense_features, category_maps)
    x_valid = _transform_features(valid_df, sparse_features, dense_features, category_maps)

    model = LGBMClassifier(
        n_estimators=int(lgb_cfg.get("n_estimators", 100)),
        learning_rate=float(lgb_cfg.get("learning_rate", 0.05)),
        num_leaves=int(lgb_cfg.get("num_leaves", 31)),
        objective="binary",
        random_state=int(config["project"]["seed"]),
        n_jobs=1,
        verbose=-1,
        force_col_wise=True,
        max_bin=63,
        min_child_samples=50,
    )
    model.fit(x_train, train_df[target_col])
    valid_pred = model.predict_proba(x_valid)[:, 1]
    metrics = binary_classification_metrics(valid_df[target_col].to_numpy(), valid_pred)
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "category_maps": category_maps,
            "sparse_features": sparse_features,
            "dense_features": dense_features,
        },
        path,
    )
    return path, metrics


def predict_lightgbm(model_path: str | Path, df: pd.DataFrame, feature_config: Dict[str, Any]) -> pd.Series:
    """Predict CTR probabilities with a saved LightGBM model bundle."""
    bundle = joblib.load(model_path)
    sparse_features = bundle.get("sparse_features", feature_config["sparse_features"])
    dense_features = bundle.get("dense_features", feature_config["dense_features"])
    x = _transform_features(df, sparse_features, dense_features, bundle["category_maps"])
    return pd.Series(bundle["model"].predict_proba(x)[:, 1], index=df.index)


def _fit_category_maps(df: pd.DataFrame, sparse_features: list[str]) -> Dict[str, Dict[str, int]]:
    """Fit category-to-code maps using train data."""
    maps: Dict[str, Dict[str, int]] = {}
    for col in sparse_features:
        values = sorted(df[col].fillna("unknown").astype(str).unique().tolist())
        maps[col] = {value: idx + 1 for idx, value in enumerate(values)}
    return maps


def _transform_features(
    df: pd.DataFrame,
    sparse_features: list[str],
    dense_features: list[str],
    category_maps: Dict[str, Dict[str, int]],
) -> pd.DataFrame:
    """Transform sparse fields to integer codes and keep dense fields numeric."""
    transformed = pd.DataFrame(index=df.index)
    for col in sparse_features:
        transformed[col] = df[col].fillna("unknown").astype(str).map(category_maps[col]).fillna(0).astype("float32")
    for col in dense_features:
        transformed[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float32")
    return transformed[sparse_features + dense_features]
