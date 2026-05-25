from __future__ import annotations

import _bootstrap  # noqa: F401

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.utils.io import load_config, read_json, resolve_path, write_json
from src.utils.logger import get_logger
from src.utils.mlflow_utils import configure_mlflow, safe_log_artifact, safe_log_metrics, safe_log_params, safe_mlflow_start_run
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


def main() -> None:
    """Train all ranking models and select the best by validation AUC."""
    config = load_config()
    set_global_seed(int(config["project"]["seed"]))
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    train_path = processed_dir / "train.csv"
    valid_path = processed_dir / "valid.csv"
    feature_path = processed_dir / "feature_columns.json"
    if not train_path.exists() or not valid_path.exists() or not feature_path.exists():
        raise FileNotFoundError("Processed data not found. Run python scripts/run_data_pipeline.py first.")

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    feature_config = read_json(feature_path)
    model_dir = resolve_path(config["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = resolve_path(config["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    configure_mlflow(config)
    model_specs = [
        ("lr", model_dir / "lr_model.joblib"),
        ("lightgbm", model_dir / "lightgbm_model.joblib"),
        ("deepfm", model_dir / "deepfm_model.pt"),
    ]
    results: Dict[str, Dict[str, Any]] = {}
    failures: Dict[str, str] = {}

    for model_name, model_path in model_specs:
        logger.info("Training model: %s", model_name)
        try:
            with safe_mlflow_start_run(run_name=model_name):
                safe_log_params({"model_name": model_name, "target_col": feature_config["target_col"]})
                trained_path, metrics = _train_one_model(model_name, train_df, valid_df, feature_config, config, model_path)
                safe_log_metrics({f"valid_{k}": v for k, v in metrics.items()})
                safe_log_artifact(trained_path)
            results[model_name] = {"model_path": str(Path(trained_path).relative_to(resolve_path("."))), "valid_metrics": metrics}
            logger.info("Model %s finished: %s", model_name, metrics)
        except Exception as exc:
            logger.exception("Model %s failed but training will continue.", model_name)
            failures[model_name] = str(exc)

    if not results:
        failure_path = metrics_dir / "training_failures.json"
        write_json({"failures": failures, "created_at": datetime.now().isoformat()}, failure_path)
        raise SystemExit("All models failed. See reports/metrics/training_failures.json for details.")

    best_model = _select_best_model(results)
    best_info = results[best_model]
    registry = {
        "best_model": best_model,
        "model_path": best_info["model_path"],
        "model_type": best_model,
        "feature_columns": feature_config["all_features"],
        "sparse_features": feature_config["sparse_features"],
        "dense_features": feature_config["dense_features"],
        "valid_metrics": best_info["valid_metrics"],
        "all_valid_metrics": {name: info["valid_metrics"] for name, info in results.items()},
        "training_failures": failures,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(registry, config["paths"]["model_registry"])

    comparison_rows = []
    for name, info in results.items():
        row = {"model": name, **info["valid_metrics"], "status": "success"}
        comparison_rows.append(row)
    for name, reason in failures.items():
        comparison_rows.append({"model": name, "auc": np.nan, "logloss": np.nan, "status": f"failed: {reason}"})
    pd.DataFrame(comparison_rows).to_csv(metrics_dir / "model_comparison.csv", index=False, encoding="utf-8")
    logger.info("Best model=%s registry saved to %s", best_model, resolve_path(config["paths"]["model_registry"]))


def _train_one_model(model_name: str, train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_config: Dict[str, Any], config: Dict[str, Any], model_path: Path):
    """Dispatch one model training function lazily so failures are isolated."""
    if model_name == "lr":
        from src.models.train_lr import train_lr_model

        return train_lr_model(train_df, valid_df, feature_config, config, model_path)
    if model_name == "lightgbm":
        from src.models.train_lightgbm import train_lightgbm_model

        return train_lightgbm_model(train_df, valid_df, feature_config, config, model_path)
    if model_name == "deepfm":
        from src.models.train_deepfm import train_deepfm_model

        return train_deepfm_model(train_df, valid_df, feature_config, config, model_path)
    raise ValueError(f"Unknown model: {model_name}")


def _select_best_model(results: Dict[str, Dict[str, Any]]) -> str:
    """Select the best model by validation AUC, then by lower LogLoss."""
    def score(item):
        _, info = item
        metrics = info["valid_metrics"]
        auc = metrics.get("auc", float("nan"))
        logloss = metrics.get("logloss", float("inf"))
        auc_score = auc if np.isfinite(auc) else -1.0
        return (auc_score, -logloss)

    return max(results.items(), key=score)[0]


if __name__ == "__main__":
    main()
