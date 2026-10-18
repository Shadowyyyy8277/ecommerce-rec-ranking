from __future__ import annotations

import pytest

from src.models.registry import list_supported_models, predict_model, train_model


def test_list_supported_models():
    models = list_supported_models()
    assert "lr" in models
    assert "lightgbm" in models
    assert "deepfm" in models


def test_unsupported_model_raises():
    with pytest.raises(ValueError, match="Unsupported model 'unsupported_xyz'"):
        list_supported_models()
        predict_model("unsupported_xyz", "dummy_path", None, {})
