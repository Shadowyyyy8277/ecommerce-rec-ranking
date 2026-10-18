"""Offline metrics for CTR ranking and Top-K recommendation."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def binary_classification_metrics(y_true: Sequence[int], y_pred: Sequence[float]) -> Dict[str, float]:
    """Compute AUC and LogLoss for binary CTR prediction."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.clip(np.asarray(y_pred).astype(float), 1e-7, 1 - 1e-7)
    if len(np.unique(y_true_arr)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true_arr, y_pred_arr))
    loss = float(log_loss(y_true_arr, y_pred_arr, labels=[0, 1]))
    return {"auc": auc, "logloss": loss}


def precision_at_k(recommended: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Compute Precision@K for one user."""
    if k <= 0:
        return 0.0
    recs = list(recommended)[:k]
    if not recs:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for item in recs if item in relevant_set)
    return hits / float(k)


def recall_at_k(recommended: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Compute Recall@K for one user."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    recs = list(recommended)[:k]
    hits = sum(1 for item in recs if item in relevant_set)
    return hits / float(len(relevant_set))


def ndcg_at_k(recommended: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Compute binary NDCG@K for one user."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    recs = list(recommended)[:k]
    dcg = 0.0
    for rank, item in enumerate(recs, start=1):
        if item in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Compute HitRate@K for one user."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    recs = list(recommended)[:k]
    return float(any(item in relevant_set for item in recs))


def mrr_at_k(recommended: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Compute Mean Reciprocal Rank (MRR) at K for one user."""
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    for rank, item in enumerate(list(recommended)[:k], start=1):
        if item in relevant_set:
            return 1.0 / float(rank)
    return 0.0


def evaluate_topk(
    recommendations: pd.DataFrame,
    positives: pd.DataFrame,
    k_list: List[int],
    user_col: str = "user_id",
    item_col: str = "item_id",
    score_col: str = "score",
) -> Dict[str, float]:
    """Evaluate Top-K recommendation metrics over users with positives."""
    if recommendations.empty or positives.empty:
        return {f"{metric}@{k}": 0.0 for k in k_list for metric in ["precision", "recall", "ndcg", "hit_rate", "mrr"]}

    recs = recommendations.sort_values([user_col, score_col], ascending=[True, False])
    rec_map = recs.groupby(user_col)[item_col].apply(list).to_dict()
    pos_map = positives.groupby(user_col)[item_col].apply(lambda x: set(x)).to_dict()

    metrics: Dict[str, float] = {}
    for k in k_list:
        precision_values = []
        recall_values = []
        ndcg_values = []
        hit_values = []
        mrr_values = []
        for user_id, relevant in pos_map.items():
            recommended = rec_map.get(user_id, [])
            precision_values.append(precision_at_k(recommended, relevant, k))
            recall_values.append(recall_at_k(recommended, relevant, k))
            ndcg_values.append(ndcg_at_k(recommended, relevant, k))
            hit_values.append(hit_rate_at_k(recommended, relevant, k))
            mrr_values.append(mrr_at_k(recommended, relevant, k))
        metrics[f"precision@{k}"] = float(np.mean(precision_values)) if precision_values else 0.0
        metrics[f"recall@{k}"] = float(np.mean(recall_values)) if recall_values else 0.0
        metrics[f"ndcg@{k}"] = float(np.mean(ndcg_values)) if ndcg_values else 0.0
        metrics[f"hit_rate@{k}"] = float(np.mean(hit_values)) if hit_values else 0.0
        metrics[f"mrr@{k}"] = float(np.mean(mrr_values)) if mrr_values else 0.0
    return metrics


def average_inference_latency_ms(predict_fn, sample_df: pd.DataFrame, repeat: int = 3) -> float:
    """Measure average per-row inference latency in milliseconds."""
    if sample_df.empty:
        return 0.0
    durations = []
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        predict_fn(sample_df)
        durations.append(time.perf_counter() - start)
    return float(np.mean(durations) * 1000.0 / len(sample_df))


def model_size_mb(model_path: str | Path) -> float:
    """Return model file size in MB."""
    path = Path(model_path)
    if not path.exists():
        return 0.0
    return float(path.stat().st_size / (1024 * 1024))
