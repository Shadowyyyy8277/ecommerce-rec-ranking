"""Training and prediction helpers for DeepFM."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.evaluation.metrics import binary_classification_metrics
from src.features.encoders import (
    build_category_encoders,
    fit_dense_scaler,
    transform_deepfm_features,
)
from src.models.deepfm import DeepFM
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_deepfm_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_config: Dict[str, Any],
    config: Dict[str, Any],
    model_path: str | Path,
) -> Tuple[Path, Dict[str, float]]:
    """Train a PyTorch DeepFM model and save a checkpoint."""
    torch.manual_seed(int(config["project"]["seed"]))
    if not torch.cuda.is_available():
        torch.set_num_threads(1)
    sparse_features = feature_config["sparse_features"]
    dense_features = feature_config["dense_features"]
    target_col = feature_config["target_col"]
    deepfm_cfg = config["training"]["deepfm"]

    encoders = build_category_encoders(train_df, sparse_features)
    scaler = fit_dense_scaler(train_df, dense_features)
    train_sparse, train_dense = transform_deepfm_features(train_df, sparse_features, dense_features, encoders, scaler)
    valid_sparse, valid_dense = transform_deepfm_features(valid_df, sparse_features, dense_features, encoders, scaler)
    y_train = train_df[target_col].to_numpy(dtype=np.float32)
    y_valid = valid_df[target_col].to_numpy(dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training DeepFM on device=%s", device)
    field_dims = [len(encoders[col]) + 1 for col in sparse_features]
    model = DeepFM(field_dims, len(dense_features), embedding_dim=int(deepfm_cfg.get("embedding_dim", 16))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(deepfm_cfg.get("lr", 0.001)))
    criterion = torch.nn.BCEWithLogitsLoss()

    dataset = TensorDataset(
        torch.tensor(train_sparse, dtype=torch.long),
        torch.tensor(train_dense, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=int(deepfm_cfg.get("batch_size", 512)), shuffle=True)

    total_epochs = int(deepfm_cfg.get("epochs", 3))
    scheduler_name = str(deepfm_cfg.get("lr_scheduler", "cosine")).lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs), eta_min=1e-6)
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, total_epochs // 2), gamma=0.5)
    else:
        scheduler = None

    best_metrics: Dict[str, float] = {"auc": float("nan"), "logloss": float("inf")}
    best_state = None
    for epoch in range(total_epochs):
        model.train()
        losses = []
        progress = tqdm(loader, desc=f"deepfm epoch {epoch + 1}", leave=False)
        for sparse_batch, dense_batch, y_batch in progress:
            sparse_batch = sparse_batch.to(device)
            dense_batch = dense_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(sparse_batch, dense_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            progress.set_postfix({"loss": f"{np.mean(losses):.4f}"})

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = float(deepfm_cfg.get("lr", 0.001))

        valid_pred = predict_deepfm_arrays(model, valid_sparse, valid_dense, device=device)
        metrics = binary_classification_metrics(y_valid, valid_pred)
        logger.info("DeepFM epoch=%s lr=%.6f loss=%.4f valid=%s", epoch + 1, current_lr, float(np.mean(losses)), metrics)
        if _is_better(metrics, best_metrics):
            best_metrics = metrics
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "field_dims": field_dims,
        "embedding_dim": int(deepfm_cfg.get("embedding_dim", 16)),
        "sparse_features": sparse_features,
        "dense_features": dense_features,
        "encoders": encoders,
        "scaler": scaler,
        "feature_config": feature_config,
        "valid_metrics": best_metrics,
    }
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path, best_metrics


def predict_deepfm(model_path: str | Path, df: pd.DataFrame, feature_config: Dict[str, Any] | None = None) -> pd.Series:
    """Predict CTR probabilities with a saved DeepFM checkpoint."""
    checkpoint = _torch_load_checkpoint(model_path)
    sparse_features = checkpoint["sparse_features"]
    dense_features = checkpoint["dense_features"]
    sparse_x, dense_x = transform_deepfm_features(df, sparse_features, dense_features, checkpoint["encoders"], checkpoint["scaler"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepFM(checkpoint["field_dims"], len(dense_features), embedding_dim=int(checkpoint["embedding_dim"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    preds = predict_deepfm_arrays(model, sparse_x, dense_x, device=device)
    return pd.Series(preds, index=df.index)

# Encoder and scaler utilities are imported from src.features.encoders


def predict_deepfm_arrays(model: DeepFM, sparse_x: np.ndarray, dense_x: np.ndarray, device: torch.device) -> np.ndarray:
    """Predict DeepFM probabilities from numpy arrays."""
    model.eval()
    preds = []
    dataset = TensorDataset(torch.tensor(sparse_x, dtype=torch.long), torch.tensor(dense_x, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=4096, shuffle=False)
    with torch.no_grad():
        for sparse_batch, dense_batch in loader:
            logits = model(sparse_batch.to(device), dense_batch.to(device))
            preds.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=np.float32)


def _is_better(candidate: Dict[str, float], current: Dict[str, float]) -> bool:
    """Compare validation metrics using AUC first and LogLoss second."""
    cand_auc = candidate.get("auc", float("nan"))
    curr_auc = current.get("auc", float("nan"))
    if np.isfinite(cand_auc) and (not np.isfinite(curr_auc) or cand_auc > curr_auc):
        return True
    if not np.isfinite(cand_auc) and not np.isfinite(curr_auc):
        return candidate.get("logloss", float("inf")) < current.get("logloss", float("inf"))
    return False


def _torch_load_checkpoint(model_path: str | Path) -> Dict[str, Any]:
    """Load a checkpoint while supporting newer Torch defaults."""
    try:
        return torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        return torch.load(model_path, map_location="cpu")
