"""Feature engineering for CTR ranking.

This module explicitly avoids leakage by fitting statistics on the train split
only for valid/test data and by using cumulative historical statistics for train
rows so the current row cannot see its own label.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.data.make_dataset import load_raw_interactions
from src.utils.io import resolve_path, write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

POSITIVE_BEHAVIORS = {"click", "cart", "fav", "buy"}
SPARSE_FEATURES = [
    "user_id",
    "item_id",
    "category_id",
    "brand_id",
    "user_age_bucket",
    "user_gender",
    "city_level",
    "hour",
    "day_of_week",
    "is_weekend",
    "price_bucket",
]
DENSE_FEATURES = [
    "item_price",
    "user_total_views",
    "user_total_clicks",
    "user_total_buys",
    "user_ctr",
    "user_cvr",
    "item_total_views",
    "item_total_clicks",
    "item_total_buys",
    "item_ctr",
    "item_cvr",
    "category_ctr",
    "category_cvr",
    "user_category_views",
    "user_category_clicks",
    "user_category_ctr",
]
STAT_FEATURES = [feature for feature in DENSE_FEATURES if feature != "item_price"]


def build_and_save_datasets(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the full data processing and feature engineering pipeline."""
    df = load_raw_interactions(config)
    train_df, valid_df, test_df, feature_config = create_feature_splits(df, config)

    processed_dir = resolve_path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_dir / "train.csv", index=False, encoding="utf-8")
    valid_df.to_csv(processed_dir / "valid.csv", index=False, encoding="utf-8")
    test_df.to_csv(processed_dir / "test.csv", index=False, encoding="utf-8")
    write_json(feature_config, processed_dir / "feature_columns.json")
    logger.info(
        "Saved processed datasets: train=%s valid=%s test=%s",
        train_df.shape,
        valid_df.shape,
        test_df.shape,
    )
    return feature_config


def create_feature_splits(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Create time-based train/valid/test splits and leakage-safe features."""
    df = add_label(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    train_raw, valid_raw, test_raw = split_by_time(df, config)
    price_edges = compute_price_bucket_edges(train_raw["item_price"])

    train_base = add_basic_features(train_raw, price_edges)
    valid_base = add_basic_features(valid_raw, price_edges)
    test_base = add_basic_features(test_raw, price_edges)

    train_features = add_cumulative_train_statistics(train_base)
    train_stats = compute_train_statistics(train_base)
    valid_features = apply_train_statistics(valid_base, train_stats)
    test_features = apply_train_statistics(test_base, train_stats)

    train_features = finalize_feature_frame(train_features)
    valid_features = finalize_feature_frame(valid_features)
    test_features = finalize_feature_frame(test_features)

    feature_config = build_feature_config(price_edges)
    validate_no_leakage_columns(feature_config)
    return train_features, valid_features, test_features, feature_config


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary CTR label from behavior_type without using it as a model input."""
    result = df.copy()
    result["label"] = result["behavior_type"].isin(POSITIVE_BEHAVIORS).astype(int)
    return result


def split_by_time(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset by timestamp according to configured ratios."""
    if not config["split"].get("split_by_time", True):
        raise ValueError("This project requires split_by_time=true to avoid leakage.")
    n = len(df)
    train_end = int(n * float(config["split"]["train_ratio"]))
    valid_end = train_end + int(n * float(config["split"]["valid_ratio"]))
    return (
        df.iloc[:train_end].copy().reset_index(drop=True),
        df.iloc[train_end:valid_end].copy().reset_index(drop=True),
        df.iloc[valid_end:].copy().reset_index(drop=True),
    )


def compute_price_bucket_edges(prices: Iterable[float], n_buckets: int = 5) -> List[float]:
    """Fit price-bucket quantile edges on train prices only."""
    values = np.asarray(list(prices), dtype=float)
    if len(values) == 0:
        return []
    quantiles = np.linspace(0, 1, n_buckets + 1)[1:-1]
    edges = np.quantile(values, quantiles)
    return [float(x) for x in np.unique(np.round(edges, 6))]


def add_basic_features(df: pd.DataFrame, price_edges: List[float]) -> pd.DataFrame:
    """Add timestamp and price bucket features."""
    result = df.copy()
    ts = pd.to_datetime(result["timestamp"])
    result["hour"] = ts.dt.hour.astype(int)
    result["day_of_week"] = ts.dt.dayofweek.astype(int)
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
    result["price_bucket"] = np.digitize(result["item_price"].astype(float), price_edges).astype(int)
    return result


def add_cumulative_train_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative historical statistics for train rows only.

    The current sample is subtracted from each cumulative count. Therefore a row
    never sees its own behavior or label.
    """
    result = df.sort_values("timestamp").copy().reset_index(drop=True)
    result = _add_behavior_indicators(result)

    _add_group_counts(result, ["user_id"], "user")
    _add_group_counts(result, ["item_id"], "item")
    _add_group_counts(result, ["category_id"], "category")
    _add_group_counts(result, ["user_id", "category_id"], "user_category")

    result["user_ctr"] = _safe_rate(result["user_total_clicks"], result["user_total_views"] + result["user_total_clicks"])
    result["user_cvr"] = _safe_rate(result["user_total_buys"], result["user_total_clicks"])
    result["item_ctr"] = _safe_rate(result["item_total_clicks"], result["item_total_views"] + result["item_total_clicks"])
    result["item_cvr"] = _safe_rate(result["item_total_buys"], result["item_total_clicks"])
    result["category_ctr"] = _safe_rate(result["category_total_clicks"], result["category_total_views"] + result["category_total_clicks"])
    result["category_cvr"] = _safe_rate(result["category_total_buys"], result["category_total_clicks"])
    result["user_category_ctr"] = _safe_rate(
        result["user_category_total_clicks"],
        result["user_category_total_views"] + result["user_category_total_clicks"],
    )
    result["user_category_views"] = result["user_category_total_views"]
    result["user_category_clicks"] = result["user_category_total_clicks"]

    drop_cols = [
        "_is_view",
        "_is_click",
        "_is_buy",
        "category_total_views",
        "category_total_clicks",
        "category_total_buys",
        "user_category_total_views",
        "user_category_total_clicks",
        "user_category_total_buys",
    ]
    return result.drop(columns=[c for c in drop_cols if c in result.columns])


def compute_train_statistics(train_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compute train-only aggregate statistics for valid/test/inference."""
    base = _add_behavior_indicators(train_df.copy())
    return {
        "user": _aggregate_counts(base, ["user_id"], "user"),
        "item": _aggregate_counts(base, ["item_id"], "item"),
        "category": _aggregate_counts(base, ["category_id"], "category"),
        "user_category": _aggregate_counts(base, ["user_id", "category_id"], "user_category"),
    }


def apply_train_statistics(df: pd.DataFrame, stats: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Apply train-only statistics to a non-train split."""
    result = df.copy()
    result = result.merge(stats["user"], on="user_id", how="left")
    result = result.merge(stats["item"], on="item_id", how="left")
    result = result.merge(stats["category"][["category_id", "category_ctr", "category_cvr"]], on="category_id", how="left")
    result = result.merge(
        stats["user_category"][["user_id", "category_id", "user_category_views", "user_category_clicks", "user_category_ctr"]],
        on=["user_id", "category_id"],
        how="left",
    )
    return result


def finalize_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values and normalize sparse feature representation."""
    result = df.copy()
    if "label" not in result.columns:
        result["label"] = 0
    for col in SPARSE_FEATURES:
        if col not in result.columns:
            result[col] = "unknown"
        result[col] = result[col].fillna("unknown").astype(str)
    for col in DENSE_FEATURES:
        if col not in result.columns:
            result[col] = 0.0
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0).astype(float)
    result["label"] = pd.to_numeric(result["label"], errors="coerce").fillna(0).astype(int)
    return result


def build_feature_config(price_edges: List[float]) -> Dict[str, Any]:
    """Build feature column metadata saved with processed data."""
    return {
        "target_col": "label",
        "sparse_features": SPARSE_FEATURES,
        "dense_features": DENSE_FEATURES,
        "all_features": SPARSE_FEATURES + DENSE_FEATURES,
        "excluded_features": ["behavior_type", "timestamp", "label"],
        "price_bucket_edges": price_edges,
        "label_definition": "1 if behavior_type in ['click', 'cart', 'fav', 'buy'] else 0",
    }


def validate_no_leakage_columns(feature_config: Dict[str, Any]) -> None:
    """Validate that leakage-prone columns are not used as model inputs."""
    all_features = set(feature_config["all_features"])
    forbidden = {"behavior_type", "label", "timestamp"}
    leaked = all_features.intersection(forbidden)
    if leaked:
        raise ValueError(f"Leakage-prone columns found in model inputs: {sorted(leaked)}")


def build_inference_features(
    candidates_df: pd.DataFrame,
    train_df: pd.DataFrame,
    config: Dict[str, Any] | None = None,
    feature_config: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build model features for arbitrary user-item candidate pairs.

    Candidate features use metadata and aggregate statistics derived from train
    data only. This function is used by offline Top-K evaluation, FastAPI, and
    Streamlit.
    """
    if candidates_df.empty:
        return finalize_feature_frame(candidates_df.copy())

    base = candidates_df.copy()
    base = base.drop_duplicates(subset=["user_id", "item_id", "recall_type"] if "recall_type" in base.columns else ["user_id", "item_id"])
    train_sorted = train_df.copy()
    train_sorted["timestamp"] = pd.to_datetime(train_sorted["timestamp"])
    train_sorted = train_sorted.sort_values("timestamp")

    user_meta_cols = ["user_id", "user_age_bucket", "user_gender", "city_level"]
    item_meta_cols = ["item_id", "category_id", "brand_id", "item_price"]
    user_meta = train_sorted[user_meta_cols].drop_duplicates("user_id", keep="last")
    item_meta = train_sorted[item_meta_cols].drop_duplicates("item_id", keep="last")

    result = base.merge(user_meta, on="user_id", how="left")
    result = result.merge(item_meta, on="item_id", how="left")
    result["user_age_bucket"] = result["user_age_bucket"].fillna("unknown")
    result["user_gender"] = result["user_gender"].fillna("unknown")
    result["city_level"] = result["city_level"].fillna("unknown")
    result["category_id"] = result["category_id"].fillna("unknown")
    result["brand_id"] = result["brand_id"].fillna("unknown")
    result["item_price"] = pd.to_numeric(result["item_price"], errors="coerce").fillna(float(train_sorted["item_price"].median()))
    result["behavior_type"] = "unknown"
    result["label"] = 0
    result["timestamp"] = pd.Timestamp.now().replace(microsecond=0)

    price_edges = feature_config.get("price_bucket_edges", []) if feature_config else compute_price_bucket_edges(train_sorted["item_price"])
    result = add_basic_features(result, price_edges)
    train_stats = compute_train_statistics(train_sorted)
    result = apply_train_statistics(result, train_stats)
    return finalize_feature_frame(result)


def _add_behavior_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add helper indicator columns for behavior aggregation."""
    result = df.copy()
    result["_is_view"] = (result["behavior_type"] == "view").astype(int)
    result["_is_click"] = result["label"].astype(int)
    result["_is_buy"] = (result["behavior_type"] == "buy").astype(int)
    return result


def _add_group_counts(df: pd.DataFrame, group_cols: List[str], prefix: str) -> None:
    """Add shifted cumulative counts in-place for a group."""
    df[f"{prefix}_total_views"] = df.groupby(group_cols)["_is_view"].cumsum() - df["_is_view"]
    df[f"{prefix}_total_clicks"] = df.groupby(group_cols)["_is_click"].cumsum() - df["_is_click"]
    df[f"{prefix}_total_buys"] = df.groupby(group_cols)["_is_buy"].cumsum() - df["_is_buy"]


def _aggregate_counts(df: pd.DataFrame, group_cols: List[str], prefix: str) -> pd.DataFrame:
    """Aggregate train statistics for one entity type."""
    grouped = (
        df.groupby(group_cols)
        .agg(
            total_views=("_is_view", "sum"),
            total_clicks=("_is_click", "sum"),
            total_buys=("_is_buy", "sum"),
        )
        .reset_index()
    )

    if prefix == "category":
        grouped["category_ctr"] = _safe_rate(grouped["total_clicks"], grouped["total_views"] + grouped["total_clicks"])
        grouped["category_cvr"] = _safe_rate(grouped["total_buys"], grouped["total_clicks"])
        return grouped[group_cols + ["category_ctr", "category_cvr"]]

    if prefix == "user_category":
        grouped["user_category_views"] = grouped["total_views"]
        grouped["user_category_clicks"] = grouped["total_clicks"]
        grouped["user_category_ctr"] = _safe_rate(grouped["total_clicks"], grouped["total_views"] + grouped["total_clicks"])
        return grouped[group_cols + ["user_category_views", "user_category_clicks", "user_category_ctr"]]

    grouped[f"{prefix}_total_views"] = grouped["total_views"]
    grouped[f"{prefix}_total_clicks"] = grouped["total_clicks"]
    grouped[f"{prefix}_total_buys"] = grouped["total_buys"]
    grouped[f"{prefix}_ctr"] = _safe_rate(grouped["total_clicks"], grouped["total_views"] + grouped["total_clicks"])
    grouped[f"{prefix}_cvr"] = _safe_rate(grouped["total_buys"], grouped["total_clicks"])
    return grouped[
        group_cols
        + [
            f"{prefix}_total_views",
            f"{prefix}_total_clicks",
            f"{prefix}_total_buys",
            f"{prefix}_ctr",
            f"{prefix}_cvr",
        ]
    ]


def _safe_rate(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray, smoothing: float = 1.0) -> pd.Series:
    """Return a smoothed rate that is zero when numerator is zero."""
    numerator_series = pd.Series(numerator, dtype=float)
    denominator_series = pd.Series(denominator, dtype=float)
    return numerator_series / (denominator_series + smoothing)
