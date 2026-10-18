from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    average_inference_latency_ms,
    binary_classification_metrics,
    evaluate_topk,
    hit_rate_at_k,
    model_size_mb,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_topk_metrics():
    recommended = ["i1", "i2", "i3"]
    relevant = {"i2", "i4"}
    assert precision_at_k(recommended, relevant, 2) == 0.5
    assert recall_at_k(recommended, relevant, 2) == 0.5
    assert hit_rate_at_k(recommended, relevant, 1) == 0.0
    assert hit_rate_at_k(recommended, relevant, 2) == 1.0
    assert 0.0 < ndcg_at_k(recommended, relevant, 3) <= 1.0


def test_mrr_metric():
    recommended = ["i1", "i2", "i3"]
    # Hit at rank 2 -> MRR is 1/2 = 0.5
    assert mrr_at_k(recommended, {"i2"}, 3) == 0.5
    # Hit at rank 1 -> MRR is 1/1 = 1.0
    assert mrr_at_k(recommended, {"i1"}, 3) == 1.0
    # No hit within k=1 -> MRR is 0.0
    assert mrr_at_k(recommended, {"i2"}, 1) == 0.0
    # Empty relevant -> MRR is 0.0
    assert mrr_at_k(recommended, set(), 3) == 0.0


def test_evaluate_topk_aggregation():
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "item_id": ["i1", "i2", "i3", "i4"],
            "score": [0.9, 0.8, 0.7, 0.6],
        }
    )
    positives = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "item_id": ["i1", "i4"],
        }
    )
    metrics = evaluate_topk(recommendations, positives, [2])
    assert "precision@2" in metrics
    assert "recall@2" in metrics
    assert "ndcg@2" in metrics
    assert "hit_rate@2" in metrics
    assert "mrr@2" in metrics
    assert metrics["hit_rate@2"] == 1.0


def test_evaluate_topk_empty():
    empty_df = pd.DataFrame()
    metrics = evaluate_topk(empty_df, empty_df, [5, 10])
    assert metrics["precision@5"] == 0.0
    assert metrics["mrr@10"] == 0.0


def test_binary_classification_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = binary_classification_metrics(y_true, y_pred)
    assert metrics["auc"] == 1.0
    assert metrics["logloss"] < 0.3


def test_latency_and_size(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    latency = average_inference_latency_ms(lambda x: x["a"] * 2, df, repeat=2)
    assert latency >= 0.0

    dummy_file = tmp_path / "model.bin"
    dummy_file.write_bytes(b"0" * 1024 * 1024)
    size = model_size_mb(dummy_file)
    assert 0.99 <= size <= 1.01
