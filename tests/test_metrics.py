from __future__ import annotations

import numpy as np

from src.evaluation.metrics import binary_classification_metrics, hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k


def test_topk_metrics():
    recommended = ["i1", "i2", "i3"]
    relevant = {"i2", "i4"}
    assert precision_at_k(recommended, relevant, 2) == 0.5
    assert recall_at_k(recommended, relevant, 2) == 0.5
    assert hit_rate_at_k(recommended, relevant, 1) == 0.0
    assert hit_rate_at_k(recommended, relevant, 2) == 1.0
    assert 0.0 < ndcg_at_k(recommended, relevant, 3) <= 1.0


def test_binary_classification_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = binary_classification_metrics(y_true, y_pred)
    assert metrics["auc"] == 1.0
    assert metrics["logloss"] < 0.3
