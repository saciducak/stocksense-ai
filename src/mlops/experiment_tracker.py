"""MLflow-based experiment tracking for model training runs.

Provides a clean wrapper around MLflow's tracking API for:
- Hyperparameter logging
- Training/validation metric tracking
- Model artifact management
- Experiment organization

This module addresses the job posting requirement:
    - "LLMOps altyapısı kurmak; deney takibi"

Why MLflow?
    1. Open-source and vendor-agnostic (vs. W&B which is paid)
    2. Integrated model registry for versioning
    3. Industry-standard for experiment tracking
    4. Native PyTorch model logging support
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

import mlflow
import mlflow.pytorch

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """MLflow experiment tracking wrapper.

    Simplifies the MLflow API into a clean interface for:
    - Starting/ending experiment runs
    - Logging parameters, metrics, and artifacts
    - Organizing experiments by name

    Example:
        >>> tracker = ExperimentTracker(experiment_name="stocksense-v1")
        >>> with tracker.start_run(run_name="transformer-lr001"):
        ...     tracker.log_params({"lr": 0.001, "epochs": 100})
        ...     for epoch in range(100):
        ...         tracker.log_metrics({"train_loss": loss}, step=epoch)
        ...     tracker.log_model(model, "transformer")
    """

    def __init__(
        self,
        experiment_name: str = "stocksense-ai",
        tracking_uri: str = "mlruns",
    ) -> None:
        """Initialize experiment tracker.

        Args:
            experiment_name: Name of the MLflow experiment.
            tracking_uri: Path or URI for MLflow tracking server.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._active_run: Optional[mlflow.ActiveRun] = None

        # Configure MLflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        logger.info(
            f"ExperimentTracker initialized: "
            f"experiment='{experiment_name}', uri='{tracking_uri}'"
        )

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> Generator[mlflow.ActiveRun, None, None]:
        """Start a new MLflow run as a context manager.

        Args:
            run_name: Human-readable name for this run.
            tags: Optional tags for run organization.

        Yields:
            Active MLflow run object.

        Example:
            >>> with tracker.start_run(run_name="experiment-1"):
            ...     tracker.log_params({"lr": 0.001})
        """
        try:
            run = mlflow.start_run(run_name=run_name, tags=tags)
            self._active_run = run
            logger.info(f"Started MLflow run: '{run_name}' (ID: {run.info.run_id[:8]}...)")
            yield run
        except Exception as e:
            logger.error(f"Error during MLflow run: {e}")
            raise
        finally:
            mlflow.end_run()
            self._active_run = None
            logger.info(f"Ended MLflow run: '{run_name}'")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters for the current run.

        Args:
            params: Dictionary of parameter names and values.
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
        logger.info(f"  Logged {len(params)} parameters")

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics for the current run.

        Args:
            metrics: Dictionary of metric names and values.
            step: Optional training step number for time-series metrics.
        """
        mlflow.log_metrics(metrics, step=step)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
    ) -> None:
        """Log a PyTorch model as an MLflow artifact.

        Args:
            model: PyTorch model to log.
            artifact_path: Artifact directory name within the run.
            registered_model_name: If provided, register model in registry.
        """
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
        )
        logger.info(f"  Logged model to '{artifact_path}'")

    def log_artifact(self, file_path: str, artifact_path: Optional[str] = None) -> None:
        """Log a file as an artifact.

        Useful for logging plots, config files, prediction CSVs, etc.

        Args:
            file_path: Local path to file.
            artifact_path: Optional subdirectory in artifact store.
        """
        mlflow.log_artifact(file_path, artifact_path=artifact_path)
        logger.info(f"  Logged artifact: {Path(file_path).name}")

    def log_figure(self, figure: Any, filename: str) -> None:
        """Log a matplotlib/plotly figure as an artifact.

        Args:
            figure: Matplotlib or plotly figure.
            filename: Filename for the saved figure.
        """
        mlflow.log_figure(figure, filename)
        logger.info(f"  Logged figure: {filename}")

    def log_dict(self, data: dict, filename: str) -> None:
        """Log a dictionary as a JSON artifact.

        Args:
            data: Dictionary to save.
            filename: JSON filename.
        """
        mlflow.log_dict(data, filename)

    def get_best_run(
        self,
        metric: str = "val_loss",
        ascending: bool = True,
    ) -> Optional[dict]:
        """Find the best run based on a metric.

        Args:
            metric: Metric name to compare.
            ascending: If True, lower is better (loss). If False, higher is better.

        Returns:
            Dictionary with best run info, or None if no runs found.
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return None

        order = "ASC" if ascending else "DESC"
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )

        if runs.empty:
            return None

        best = runs.iloc[0]
        return {
            "run_id": best["run_id"],
            "metric_value": best.get(f"metrics.{metric}"),
            "params": {
                k.replace("params.", ""): v
                for k, v in best.items()
                if k.startswith("params.")
            },
        }

    def get_run_history(self, max_results: int = 20) -> list[dict]:
        """Get recent run history for the experiment.

        Args:
            max_results: Maximum number of runs to return.

        Returns:
            List of run summaries.
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return []

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=max_results,
        )

        return [
            {
                "run_id": row["run_id"][:8],
                "run_name": row.get("tags.mlflow.runName", "unnamed"),
                "status": row["status"],
                "start_time": str(row.get("start_time", "")),
            }
            for _, row in runs.iterrows()
        ]
