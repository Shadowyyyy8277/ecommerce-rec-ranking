"""Logistic Regression baseline for CTR ranking."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluation.metrics import binary_classification_metrics


def train_lr_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_config: Dict[str, Any],
    config: Dict[str, Any],
    model_path: str | Path,
) -> Tuple[Path, Dict[str, float]]:
    """Train a sparse+dense Logistic Regression CTR model."""
    sparse_features = feature_config["sparse_features"]
    dense_features = feature_config["dense_features"]
    target_col = feature_config["target_col"]
    model = Pipeline(
        steps=[
            ("preprocess", _build_preprocessor(sparse_features, dense_features)),
            (
                "model",
                LogisticRegression(
                    max_iter=int(config["training"]["lr"].get("max_iter", 1000)),
                    solver="liblinear",
                    random_state=int(config["project"]["seed"]),
                ),
            ),
        ]
    )
    model.fit(train_df[sparse_features + dense_features], train_df[target_col])
    valid_pred = model.predict_proba(valid_df[sparse_features + dense_features])[:, 1]
    metrics = binary_classification_metrics(valid_df[target_col].to_numpy(), valid_pred)
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path, metrics


def predict_lr(model_path: str | Path, df: pd.DataFrame, feature_config: Dict[str, Any]) -> pd.Series:
    """Predict CTR probabilities with a saved Logistic Regression model."""
    model = joblib.load(model_path)
    features = feature_config["sparse_features"] + feature_config["dense_features"]
    return pd.Series(model.predict_proba(df[features])[:, 1], index=df.index)


def _build_preprocessor(sparse_features: list[str], dense_features: list[str]) -> ColumnTransformer:
    """Build a sklearn preprocessor compatible with multiple sklearn versions."""
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # pragma: no cover - older sklearn
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
    return ColumnTransformer(
        transformers=[
            ("sparse", encoder, sparse_features),
            ("dense", StandardScaler(), dense_features),
        ]
    )
