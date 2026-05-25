"""FastAPI online recommendation service."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features.build_features import build_inference_features
from src.utils.io import load_config, read_json, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

API_DESCRIPTION = """
Online serving API for the e-commerce recommendation ranking demo.

This service provides a lightweight recommendation endpoint for portfolio and
local demonstration purposes. It loads the selected best model from the local
model registry, ranks recalled candidates, and returns Top-K recommended items.

The current project uses simulated user behavior data and does not include any
real business data.
"""

app = FastAPI(
    title="E-commerce Recommendation Ranking API",
    version="2.0",
    description=API_DESCRIPTION,
)


class RecommendRequest(BaseModel):
    """Request body for recommendation endpoint."""

    user_id: str = Field(
        ...,
        description="User ID, for example: u_00001",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of recommended items to return.",
    )


class RecommendationItem(BaseModel):
    """One recommendation item."""

    item_id: str = Field(..., description="Recommended item ID.")
    score: float = Field(..., description="Predicted CTR/ranking score.")
    reason: str = Field(..., description="Short explanation for demo display.")


class RecommendResponse(BaseModel):
    """Recommendation response."""

    user_id: str
    recommendations: List[RecommendationItem]


@app.get(
    "/health",
    tags=["service"],
    summary="Check service health",
)
def health() -> Dict[str, Any]:
    """Return service health and model availability."""
    registry = _load_registry()
    model_loaded = bool(registry and resolve_path(registry.get("model_path", "")).exists())
    return {"status": "ok", "model_loaded": model_loaded}


@app.post(
    "/recommend",
    response_model=RecommendResponse,
    tags=["recommendation"],
    summary="Recommend Top-K items for a user",
)
def recommend(request: RecommendRequest) -> RecommendResponse:
    """Recommend Top-K items for a user.

    If a trained model is missing, the service falls back to popular recall so
    the API remains available for demos and tests.
    """
    config = load_config()
    top_k = request.top_k or int(config["serving"].get("default_top_k", 10))

    registry = _load_registry()
    feature_config = _load_feature_config()
    train_df = _load_train_df()

    if registry and feature_config and train_df is not None:
        model_path = resolve_path(registry.get("model_path", ""))

        if model_path.exists():
            try:
                recs = _recommend_with_model(
                    user_id=request.user_id,
                    top_k=top_k,
                    registry=registry,
                    model_path=model_path,
                    feature_config=feature_config,
                    train_df=train_df,
                    config=config,
                )
                if recs:
                    return RecommendResponse(user_id=request.user_id, recommendations=recs)

            except Exception as exc:
                logger.warning(
                    "Model recommendation failed; falling back to popular recall. "
                    "user_id=%s, top_k=%s, error=%s",
                    request.user_id,
                    top_k,
                    exc,
                )

    fallback_recs = _fallback_popular(request.user_id, top_k)
    return RecommendResponse(user_id=request.user_id, recommendations=fallback_recs)


def _recommend_with_model(
    user_id: str,
    top_k: int,
    registry: Dict[str, Any],
    model_path: Path,
    feature_config: Dict[str, Any],
    train_df: pd.DataFrame,
    config: Dict[str, Any],
) -> List[RecommendationItem]:
    """Rank recall candidates using the trained best model."""
    recall_path = resolve_path(config["paths"]["recall_dir"]) / "merged_recall.csv"

    if recall_path.exists():
        recall_df = pd.read_csv(recall_path)
        candidates = recall_df[recall_df["user_id"] == user_id].copy()
    else:
        candidates = pd.DataFrame()

    if candidates.empty:
        popular_items = _popular_items_from_train(train_df, top_k * 5)
        candidates = pd.DataFrame(
            {
                "user_id": user_id,
                "item_id": popular_items,
                "recall_score": 1.0,
                "recall_type": "popular",
            }
        )

    features = build_inference_features(
        candidates,
        train_df,
        config=config,
        feature_config=feature_config,
    )

    features["score"] = _predict_by_model(
        registry["model_type"],
        model_path,
        features,
        feature_config,
    )

    ranked = (
        features.groupby(["user_id", "item_id"], as_index=False)["score"]
        .max()
        .sort_values("score", ascending=False)
        .head(top_k)
    )

    model_name = str(registry.get("best_model") or registry.get("model_type") or "model")

    recommendations: List[RecommendationItem] = []
    for rank, row in enumerate(ranked.itertuples(index=False), start=1):
        score = float(row.score)
        recommendations.append(
            RecommendationItem(
                item_id=str(row.item_id),
                score=score,
                reason=_build_recommend_reason(
                    rank=rank,
                    score=score,
                    model_name=model_name,
                ),
            )
        )

    return recommendations


def _fallback_popular(user_id: str, top_k: int) -> List[RecommendationItem]:
    """Return popular fallback recommendations when no model is available."""
    config = load_config()
    popular_path = resolve_path(config["paths"]["recall_dir"]) / "popular_recall.csv"

    if popular_path.exists():
        recall_df = pd.read_csv(popular_path)
        user_recs = recall_df[recall_df["user_id"] == user_id].head(top_k)

        if user_recs.empty:
            user_recs = recall_df.head(top_k)

        return [
            RecommendationItem(
                item_id=str(row.item_id),
                score=float(row.recall_score),
                reason="fallback: popular recall is used because the trained ranking model is unavailable",
            )
            for row in user_recs.itertuples(index=False)
        ]

    train_df = _load_train_df()
    items = _popular_items_from_train(train_df, top_k) if train_df is not None else []

    return [
        RecommendationItem(
            item_id=str(item_id),
            score=0.0,
            reason="fallback: no trained model or recall file is available; using popular items",
        )
        for item_id in items
    ]


def _build_recommend_reason(rank: int, score: float, model_name: str) -> str:
    """Build a concise, less template-like recommendation reason."""
    model_label = _format_model_name(model_name)

    if rank == 1:
        return f"{model_label} predicted the highest CTR among recalled candidates."

    if score >= 0.25:
        return "High CTR score after multi-channel recall and ranking."

    if score >= 0.20:
        return "Selected from recalled candidates with a stable ranking score."

    return "Kept in Top-K after CTR ranking over the merged recall set."


def _format_model_name(model_name: str) -> str:
    """Format model name for user-facing recommendation reasons."""
    normalized = str(model_name).strip().lower()

    name_map = {
        "lr": "LR",
        "lightgbm": "LightGBM",
        "lgbm": "LightGBM",
        "deepfm": "DeepFM",
    }

    return name_map.get(normalized, str(model_name))


def _popular_items_from_train(train_df: pd.DataFrame | None, top_k: int) -> List[str]:
    """Compute popular items from train data."""
    if train_df is None or train_df.empty:
        return []
    return train_df["item_id"].value_counts().head(top_k).index.astype(str).tolist()


def _predict_by_model(
    model_type: str,
    model_path: Path,
    df: pd.DataFrame,
    feature_config: Dict[str, Any],
) -> pd.Series:
    """Predict CTR with a saved model using lazy imports."""
    if model_type == "lr":
        from src.models.train_lr import predict_lr

        return predict_lr(model_path, df, feature_config)

    if model_type == "lightgbm":
        from src.models.train_lightgbm import predict_lightgbm

        return predict_lightgbm(model_path, df, feature_config)

    if model_type == "deepfm":
        from src.models.train_deepfm import predict_deepfm

        return predict_deepfm(model_path, df, feature_config)

    raise ValueError(f"Unsupported model_type: {model_type}")


def _load_registry() -> Dict[str, Any] | None:
    """Load local model selection registry if it exists."""
    config = load_config()
    path = resolve_path(config["paths"]["model_registry"])

    if not path.exists():
        return None

    try:
        raw_text = path.read_text(encoding="utf-8").strip()
        if raw_text in {"", "{}"}:
            return None
        return read_json(path)

    except Exception as exc:
        logger.warning("Failed to read model registry: %s", exc)
        return None


def _load_feature_config() -> Dict[str, Any] | None:
    """Load processed feature metadata."""
    config = load_config()
    path = resolve_path(config["paths"]["processed_dir"]) / "feature_columns.json"

    if not path.exists():
        return None

    try:
        return read_json(path)

    except Exception as exc:
        logger.warning("Failed to read feature config: %s", exc)
        return None


@lru_cache(maxsize=1)
def _load_train_df() -> pd.DataFrame | None:
    """Load train split for metadata and fallback recommendations."""
    config = load_config()
    path = resolve_path(config["paths"]["processed_dir"]) / "train.csv"

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)

    except Exception as exc:
        logger.warning("Failed to read train data: %s", exc)
        return None