from __future__ import annotations

import _bootstrap  # noqa: F401

from typing import Any, Dict

import pandas as pd

from src.evaluation.metrics import average_inference_latency_ms, binary_classification_metrics, evaluate_topk, model_size_mb
from src.features.build_features import build_inference_features
from src.utils.io import load_config, read_json, resolve_path, write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Evaluate the selected best model on the test set and recall candidates."""
    config = load_config()
    registry_path = resolve_path(config["paths"]["model_registry"])
    if not registry_path.exists() or registry_path.read_text(encoding="utf-8").strip() in {"", "{}"}:
        raise SystemExit("No trained model registry found. Please run python scripts/run_train_all.py first.")

    registry = read_json(registry_path)
    model_path = resolve_path(registry["model_path"])
    if not model_path.exists():
        raise SystemExit(f"Best model file not found: {model_path}. Please run python scripts/run_train_all.py first.")

    processed_dir = resolve_path(config["paths"]["processed_dir"])
    test_path = processed_dir / "test.csv"
    train_path = processed_dir / "train.csv"
    feature_path = processed_dir / "feature_columns.json"
    if not test_path.exists() or not train_path.exists() or not feature_path.exists():
        raise SystemExit("Processed data not found. Please run python scripts/run_data_pipeline.py first.")

    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)
    feature_config = read_json(feature_path)
    metrics_dir = resolve_path(config["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    pred = _predict_by_model(registry["model_type"], model_path, test_df, feature_config)
    ctr_metrics = binary_classification_metrics(test_df[feature_config["target_col"]].to_numpy(), pred)

    sample_df = test_df.head(min(512, len(test_df))).copy()
    latency = average_inference_latency_ms(lambda frame: _predict_by_model(registry["model_type"], model_path, frame, feature_config), sample_df)
    size_mb = model_size_mb(model_path)

    topk_metrics = {}
    recall_path = resolve_path(config["paths"]["recall_dir"]) / "merged_recall.csv"
    if recall_path.exists():
        recall_df = pd.read_csv(recall_path)
        candidates = recall_df[recall_df["user_id"].isin(test_df["user_id"].unique())].copy()
        if not candidates.empty:
            candidate_features = build_inference_features(candidates, train_df, config=config, feature_config=feature_config)
            candidate_features["score"] = _predict_by_model(registry["model_type"], model_path, candidate_features, feature_config)
            ranked = (
                candidate_features.groupby(["user_id", "item_id"], as_index=False)["score"]
                .max()
                .sort_values(["user_id", "score"], ascending=[True, False])
            )
            positives = test_df[test_df[feature_config["target_col"]] == 1][["user_id", "item_id"]]
            topk_metrics = evaluate_topk(ranked, positives, [int(k) for k in config["evaluation"]["top_k"]])
        else:
            logger.warning("Recall candidates are empty for test users; Top-K metrics will be zeros.")
            positives = test_df[test_df[feature_config["target_col"]] == 1][["user_id", "item_id"]]
            topk_metrics = evaluate_topk(pd.DataFrame(), positives, [int(k) for k in config["evaluation"]["top_k"]])
    else:
        logger.warning("Recall file not found: %s. Run python scripts/run_recall.py to enable Top-K evaluation.", recall_path)

    metrics: Dict[str, Any] = {
        "best_model": registry["best_model"],
        "test_auc": ctr_metrics["auc"],
        "test_logloss": ctr_metrics["logloss"],
        "topk": topk_metrics,
        "average_inference_latency_ms": latency,
        "model_size_mb": size_mb,
        "model_path": str(model_path),
    }
    write_json(metrics, metrics_dir / "metrics.json")

    flat_metrics = {"metric": [], "value": []}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_metrics["metric"].append(sub_key)
                flat_metrics["value"].append(sub_value)
        elif key not in {"model_path"}:
            flat_metrics["metric"].append(key)
            flat_metrics["value"].append(value)
    pd.DataFrame(flat_metrics).to_csv(metrics_dir / "metrics.csv", index=False, encoding="utf-8")

    comparison_rows = []
    for model_name, valid_metrics in registry.get("all_valid_metrics", {}).items():
        comparison_rows.append(
            {
                "model": model_name,
                "valid_auc": valid_metrics.get("auc"),
                "valid_logloss": valid_metrics.get("logloss"),
                "test_auc": ctr_metrics["auc"] if model_name == registry["best_model"] else None,
                "test_logloss": ctr_metrics["logloss"] if model_name == registry["best_model"] else None,
                "is_best": model_name == registry["best_model"],
            }
        )
    pd.DataFrame(comparison_rows).to_csv(metrics_dir / "model_comparison.csv", index=False, encoding="utf-8")
    logger.info("Evaluation completed. Metrics saved to %s", metrics_dir)


def _predict_by_model(model_type: str, model_path, df: pd.DataFrame, feature_config: Dict[str, Any]) -> pd.Series:
    """Predict with a saved model using lazy imports."""
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


if __name__ == "__main__":
    main()
