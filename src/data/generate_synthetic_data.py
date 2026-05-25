"""Synthetic e-commerce behavior data generation.

The generated dataset is designed for recommendation/ranking experiments. It
contains long-tail item popularity, user/category preferences, price sensitivity,
and behavior-type imbalance where views dominate and purchases are rare.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

BEHAVIOR_TYPES = ["view", "click", "cart", "fav", "buy"]
AGE_BUCKETS = ["18-24", "25-34", "35-44", "45+"]
GENDERS = ["M", "F", "unknown"]
CITY_LEVELS = ["tier_1", "tier_2", "tier_3", "tier_4", "tier_5"]


@dataclass
class SyntheticDataSpec:
    """Configuration for synthetic data generation."""

    n_users: int
    n_items: int
    n_interactions: int
    n_categories: int
    n_brands: int
    seed: int = 42


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def generate_synthetic_data(spec: SyntheticDataSpec) -> pd.DataFrame:
    """Generate a synthetic e-commerce interaction dataset.

    Parameters
    ----------
    spec:
        Dataset scale and random seed configuration.

    Returns
    -------
    pandas.DataFrame
        Interaction table with required raw columns.
    """
    rng = np.random.default_rng(spec.seed)
    logger.info(
        "Generating synthetic data: users=%s items=%s interactions=%s",
        spec.n_users,
        spec.n_items,
        spec.n_interactions,
    )

    users = _generate_users(spec.n_users, rng)
    items = _generate_items(spec.n_items, spec.n_categories, spec.n_brands, rng)
    item_popularity = _long_tail_popularity(spec.n_items, rng)
    items["popularity_weight"] = item_popularity

    category_item_indices: Dict[int, np.ndarray] = {}
    category_item_probs: Dict[int, np.ndarray] = {}
    for category_idx in range(spec.n_categories):
        indices = np.where(items["category_idx"].to_numpy() == category_idx)[0]
        if len(indices) == 0:
            indices = np.array([int(rng.integers(0, spec.n_items))])
        weights = item_popularity[indices]
        weights = weights / weights.sum()
        category_item_indices[category_idx] = indices
        category_item_probs[category_idx] = weights

    user_activity_weights = _long_tail_popularity(spec.n_users, rng)
    user_activity_weights = user_activity_weights / user_activity_weights.sum()
    user_indices = rng.choice(spec.n_users, size=spec.n_interactions, p=user_activity_weights)

    start_ts = pd.Timestamp("2025-01-01 00:00:00")
    seconds = np.sort(rng.integers(0, 120 * 24 * 3600, size=spec.n_interactions))

    rows: List[dict] = []
    price_mean = items["item_price"].mean()
    price_std = items["item_price"].std() + 1e-6

    for row_idx, user_idx in enumerate(user_indices):
        user = users.iloc[int(user_idx)]
        category_idx = _sample_category_for_user(user, spec.n_categories, rng)
        item_pool = category_item_indices[category_idx]
        item_probs = category_item_probs[category_idx]
        item_idx = int(rng.choice(item_pool, p=item_probs))
        item = items.iloc[item_idx]

        behavior = _sample_behavior(user, item, price_mean, price_std, rng)
        ts = start_ts + pd.Timedelta(seconds=int(seconds[row_idx]))
        rows.append(
            {
                "user_id": user["user_id"],
                "item_id": item["item_id"],
                "category_id": item["category_id"],
                "brand_id": item["brand_id"],
                "behavior_type": behavior,
                "timestamp": ts.isoformat(),
                "item_price": round(float(item["item_price"]), 2),
                "user_age_bucket": user["user_age_bucket"],
                "user_gender": user["user_gender"],
                "city_level": user["city_level"],
            }
        )

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    logger.info("Generated dataset shape=%s behavior_counts=%s", df.shape, df["behavior_type"].value_counts().to_dict())
    return df


def save_synthetic_data(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save synthetic interactions to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Synthetic data saved to %s", path)
    return path


def _generate_users(n_users: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate user demographic profiles and latent preferences."""
    age_probs = np.array([0.28, 0.38, 0.22, 0.12])
    gender_probs = np.array([0.48, 0.48, 0.04])
    city_probs = np.array([0.16, 0.24, 0.30, 0.20, 0.10])
    users = pd.DataFrame(
        {
            "user_id": [f"u_{i:06d}" for i in range(n_users)],
            "user_age_bucket": rng.choice(AGE_BUCKETS, size=n_users, p=age_probs),
            "user_gender": rng.choice(GENDERS, size=n_users, p=gender_probs),
            "city_level": rng.choice(CITY_LEVELS, size=n_users, p=city_probs),
        }
    )
    age_sensitivity = {"18-24": 1.25, "25-34": 1.05, "35-44": 0.9, "45+": 0.85}
    city_sensitivity = {"tier_1": 0.75, "tier_2": 0.9, "tier_3": 1.05, "tier_4": 1.18, "tier_5": 1.25}
    users["price_sensitivity"] = [
        age_sensitivity[a] * city_sensitivity[c] * rng.lognormal(0, 0.12)
        for a, c in zip(users["user_age_bucket"], users["city_level"])
    ]
    return users


def _generate_items(
    n_items: int, n_categories: int, n_brands: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Generate item metadata with category-specific price ranges."""
    category_probs = _long_tail_popularity(n_categories, rng)
    category_probs = category_probs / category_probs.sum()
    category_idx = rng.choice(n_categories, size=n_items, p=category_probs)
    brand_idx = rng.integers(0, n_brands, size=n_items)
    category_price_center = rng.lognormal(mean=3.5, sigma=0.65, size=n_categories)
    prices = category_price_center[category_idx] * rng.lognormal(mean=0, sigma=0.45, size=n_items)
    prices = np.clip(prices, 9.9, 9999.0)
    return pd.DataFrame(
        {
            "item_id": [f"i_{i:06d}" for i in range(n_items)],
            "category_idx": category_idx,
            "category_id": [f"c_{c:03d}" for c in category_idx],
            "brand_id": [f"b_{b:03d}" for b in brand_idx],
            "item_price": prices,
        }
    )


def _long_tail_popularity(n: int, rng: np.random.Generator, alpha: float = 1.15) -> np.ndarray:
    """Create a shuffled Zipf-like popularity distribution."""
    ranks = np.arange(1, n + 1, dtype=float)
    weights = 1.0 / np.power(ranks, alpha)
    rng.shuffle(weights)
    return weights / weights.sum()


def _sample_category_for_user(user: pd.Series, n_categories: int, rng: np.random.Generator) -> int:
    """Sample a category based on demographic preferences and randomness."""
    age_offset = AGE_BUCKETS.index(user["user_age_bucket"])
    gender_offset = 1 if user["user_gender"] == "F" else 0
    city_offset = CITY_LEVELS.index(user["city_level"])
    preferred_groups = {
        (age_offset * 3 + gender_offset) % 10,
        (age_offset * 5 + city_offset) % 10,
        (gender_offset * 7 + city_offset) % 10,
    }
    if rng.random() < 0.78:
        group = int(rng.choice(list(preferred_groups)))
        candidates = np.arange(group, n_categories, 10)
        return int(rng.choice(candidates))
    return int(rng.integers(0, n_categories))


def _sample_behavior(
    user: pd.Series,
    item: pd.Series,
    price_mean: float,
    price_std: float,
    rng: np.random.Generator,
) -> str:
    """Sample behavior type with realistic imbalance and price effects."""
    category_idx = int(item["category_idx"])
    age_idx = AGE_BUCKETS.index(user["user_age_bucket"])
    gender_factor = 0.08 if user["user_gender"] == "F" and category_idx % 3 == 0 else 0.0
    city_factor = 0.06 if user["city_level"] in {"tier_1", "tier_2"} and category_idx % 5 == 0 else 0.0
    category_match = 0.35 if category_idx % 4 == age_idx % 4 else -0.05
    price_z = (float(item["item_price"]) - price_mean) / price_std
    popularity_factor = 0.18 * np.log1p(float(item["popularity_weight"]) * 10000)
    price_factor = -0.55 * float(user["price_sensitivity"]) * price_z
    logit = -1.55 + category_match + gender_factor + city_factor + popularity_factor + price_factor
    engage_prob = 0.015 + 0.42 * float(sigmoid(logit))

    if rng.random() > engage_prob:
        return "view"

    buy_strength = float(sigmoid(-0.15 - 0.7 * price_z + category_match))
    p_buy = 0.025 + 0.075 * buy_strength
    p_cart = 0.12 + 0.06 * buy_strength
    p_fav = 0.08
    p_click = max(0.0, 1.0 - p_buy - p_cart - p_fav)
    return str(rng.choice(["click", "cart", "fav", "buy"], p=[p_click, p_cart, p_fav, p_buy]))
