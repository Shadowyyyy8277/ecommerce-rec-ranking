"""Model Registry for extensible training and prediction dispatch.

Allows plug-and-play addition of new CTR ranking models without modifying
orchestration scripts or serving API logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Registry mapping model_name -> dict of loader / train / predict callables
_MODEL_TRAINERS: Dict[str, Callable[..., Tuple[Path, Dict[str, float]]]] = {}
_MODEL_PREDICTORS: Dict[str, Callable[..., pd.Series]] = {}


def register_model_trainer(model_name: str, trainer_fn: Callable[..., Tuple[Path, Dict[str, float]]]) -> None:
    """Register a training function for a given model type."""
    _MODEL_TRAINERS[model_name] = trainer_fn


def register_model_predictor(model_name: str, predictor_fn: Callable[..., pd.Series]) -> None:
    """Register a prediction function for a given model type."""
    _MODEL_PREDICTORS[model_name] = predictor_fn


def _init_default_models() -> None:
    """Lazily bind default models into the registry."""
    if _MODEL_TRAINERS:
        return

    def _train_lr(*args, **kwargs):
        from src.models.train_lr import train_lr_model

        return train_lr_model(*args, **kwargs)

    def _predict_lr(*args, **kwargs):
        from src.models.train_lr import predict_lr

        return predict_lr(*args, **kwargs)

    def _train_lgbm(*args, **kwargs):
        from src.models.train_lightgbm import train_lightgbm_model

        return train_lightgbm_model(*args, **kwargs)

    def _predict_lgbm(*args, **kwargs):
        from src.models.train_lightgbm import predict_lightgbm

        return predict_lightgbm(*args, **kwargs)

    def _train_deepfm(*args, **kwargs):
        from src.models.train_deepfm import train_deepfm_model

        return train_deepfm_model(*args, **kwargs)

    def _predict_deepfm(*args, **kwargs):
        from src.models.train_deepfm import predict_deepfm

        return predict_deepfm(*args, **kwargs)

    register_model_trainer("lr", _train_lr)
    register_model_predictor("lr", _predict_lr)

    register_model_trainer("lightgbm", _train_lgbm)
    register_model_predictor("lightgbm", _predict_lgbm)

    register_model_trainer("deepfm", _train_deepfm)
    register_model_predictor("deepfm", _predict_deepfm)


def list_supported_models() -> List[str]:
    """Return a list of all registered model identifiers."""
    _init_default_models()
    return sorted(_MODEL_TRAINERS.keys())


def train_model(
    model_name: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_config: Dict[str, Any],
    config: Dict[str, Any],
    model_path: str | Path,
) -> Tuple[Path, Dict[str, float]]:
    """Dispatch training for the specified model using the registry."""
    _init_default_models()
    trainer = _MODEL_TRAINERS.get(model_name)
    if trainer is None:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models are: {list_supported_models()}"
        )
    return trainer(train_df, valid_df, feature_config, config, model_path)


def predict_model(
    model_name: str,
    model_path: str | Path,
    df: pd.DataFrame,
    feature_config: Dict[str, Any],
) -> pd.Series:
    """Dispatch prediction for the specified model using the registry."""
    _init_default_models()
    predictor = _MODEL_PREDICTORS.get(model_name)
    if predictor is None:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models are: {list_supported_models()}"
        )
    return predictor(model_path, df, feature_config)
