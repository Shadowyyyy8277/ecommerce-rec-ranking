"""Category encoders and numeric normalizers for ranking models.

Provides shared feature encoding across models (LightGBM, DeepFM, etc.),
eliminating duplicate encoding logic and ensuring consistent preprocessing.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def build_category_encoders(df: pd.DataFrame, sparse_features: List[str]) -> Dict[str, Dict[str, int]]:
    """Fit category-to-integer code mappings using training data.

    Unknown/unseen categories during inference will map to code 0.
    Known categories receive positive 1-indexed integers (1, 2, ...).
    """
    encoders: Dict[str, Dict[str, int]] = {}
    for col in sparse_features:
        values = sorted(df[col].fillna("unknown").astype(str).unique().tolist())
        encoders[col] = {value: idx + 1 for idx, value in enumerate(values)}
    return encoders


def transform_sparse_dense_dataframe(
    df: pd.DataFrame,
    sparse_features: List[str],
    dense_features: List[str],
    category_maps: Dict[str, Dict[str, int]],
) -> pd.DataFrame:
    """Transform sparse fields to integer codes and numeric fields to float32 DataFrame.

    Used by tree-based models like LightGBM.
    """
    transformed = pd.DataFrame(index=df.index)
    for col in sparse_features:
        mapping = category_maps[col]
        transformed[col] = df[col].fillna("unknown").astype(str).map(mapping).fillna(0).astype("float32")
    for col in dense_features:
        transformed[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float32")
    return transformed[sparse_features + dense_features]


def fit_dense_scaler(df: pd.DataFrame, dense_features: List[str]) -> Dict[str, List[float]]:
    """Fit mean/std normalization statistics for dense continuous features."""
    dense = df[dense_features].astype(float).to_numpy(dtype=np.float32)
    mean = dense.mean(axis=0)
    std = dense.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist()}


def transform_deepfm_features(
    df: pd.DataFrame,
    sparse_features: List[str],
    dense_features: List[str],
    encoders: Dict[str, Dict[str, int]],
    scaler: Dict[str, List[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform DataFrame into sparse integer array and z-score normalized dense array."""
    sparse_arrays = []
    for col in sparse_features:
        mapping = encoders[col]
        sparse_arrays.append(df[col].fillna("unknown").astype(str).map(mapping).fillna(0).astype(int).to_numpy())
    sparse_x = np.vstack(sparse_arrays).T.astype(np.int64)

    dense = df[dense_features].astype(float).to_numpy(dtype=np.float32)
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    dense_x = (dense - mean) / std
    return sparse_x, dense_x.astype(np.float32)
