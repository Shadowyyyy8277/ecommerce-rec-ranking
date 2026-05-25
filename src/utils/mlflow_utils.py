"""Windows-safe MLflow helpers.

MLflow is used only for experiment logging in this project. All helpers catch
MLflow failures and convert them into warnings so model training and local file
outputs still complete.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from src.utils.io import resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_mlflow_tracking_uri(config: Dict[str, Any]) -> str:
    """Return a local MLflow tracking URI in file URI format.

    Path.as_uri() is the key Windows-safe conversion. A local path such as a
    drive-based Windows path becomes a file URI, avoiding scheme confusion.
    """
    mlruns_dir = resolve_path(config["paths"]["mlruns_dir"])
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    return mlruns_dir.resolve().as_uri()


def configure_mlflow(config: Dict[str, Any]) -> bool:
    """Configure MLflow tracking and experiment name.

    Returns True when configuration succeeds and False otherwise. The function
    intentionally does not configure any model-registration feature.
    """
    try:
        import mlflow

        tracking_uri = get_mlflow_tracking_uri(config)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config["training"]["experiment_name"])
        logger.info("MLflow tracking URI configured: %s", tracking_uri)
        return True
    except Exception as exc:  # pragma: no cover - depends on local MLflow setup
        logger.warning("MLflow configuration failed and will be skipped: %s", exc)
        return False


@contextmanager
def safe_mlflow_start_run(run_name: Optional[str] = None) -> Iterator[Optional[Any]]:
    """Safely start an MLflow run.

    If MLflow is not importable or start_run fails, the context yields None.
    Exceptions raised inside the context body are not swallowed.
    """
    mlflow_module = None
    run_cm = None
    run = None
    try:
        import mlflow as mlflow_module  # type: ignore[no-redef]

        run_cm = mlflow_module.start_run(run_name=run_name)
        run = run_cm.__enter__()
    except Exception as exc:  # pragma: no cover - depends on local MLflow setup
        logger.warning("MLflow run start failed and will be skipped: %s", exc)
        run_cm = None
        run = None
    try:
        yield run
    finally:
        if run_cm is not None:
            try:
                run_cm.__exit__(None, None, None)
            except Exception as exc:  # pragma: no cover
                logger.warning("MLflow run close failed: %s", exc)


def safe_log_params(params: Dict[str, Any]) -> None:
    """Log MLflow parameters without interrupting training on failure."""
    try:
        import mlflow

        clean_params = {k: _safe_param_value(v) for k, v in params.items()}
        mlflow.log_params(clean_params)
    except Exception as exc:  # pragma: no cover
        logger.warning("MLflow parameter logging failed: %s", exc)


def safe_log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log MLflow metrics without interrupting training on failure."""
    try:
        import mlflow

        numeric_metrics = {}
        for key, value in metrics.items():
            try:
                if value is not None:
                    numeric_metrics[key] = float(value)
            except (TypeError, ValueError):
                continue
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics, step=step)
    except Exception as exc:  # pragma: no cover
        logger.warning("MLflow metric logging failed: %s", exc)


def safe_log_artifact(path_like: str | Path) -> None:
    """Log a single MLflow artifact if available."""
    try:
        import mlflow

        path = Path(path_like)
        if path.exists():
            mlflow.log_artifact(str(path))
    except Exception as exc:  # pragma: no cover
        logger.warning("MLflow artifact logging failed: %s", exc)


def _safe_param_value(value: Any) -> str | int | float | bool:
    """Convert complex parameter values to strings for MLflow logging."""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
