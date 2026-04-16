"""MLflow Model Registry for model versioning and lifecycle management.

Provides structured model versioning with stage transitions:
    None → Staging → Production → Archived

This module addresses the job posting requirement:
    - "Model versiyonlama"

Key capability: Compare any two model versions on metrics and
make informed promotion decisions.
"""

from dataclasses import dataclass
from typing import Any, Optional

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelVersion:
    """Represents a registered model version.

    Attributes:
        name: Registered model name.
        version: Version number.
        stage: Current stage (None, Staging, Production, Archived).
        run_id: MLflow run that produced this version.
        description: Version description.
        metrics: Performance metrics from training.
    """

    name: str
    version: int
    stage: str
    run_id: str
    description: str = ""
    metrics: dict[str, float] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for API/logging."""
        return {
            "name": self.name,
            "version": self.version,
            "stage": self.stage,
            "run_id": self.run_id[:8] + "...",
            "metrics": self.metrics,
        }


class ModelRegistry:
    """Model versioning and lifecycle management via MLflow Registry.

    Supports:
    - Registering new model versions from training runs
    - Stage transitions (Staging → Production)
    - Version comparison for promotion decisions
    - Loading models by version or stage

    Example:
        >>> registry = ModelRegistry(model_name="stocksense-predictor")
        >>> # Register a new version
        >>> version = registry.register(run_id="abc123", description="v2 with GELU")
        >>> # Promote to staging
        >>> registry.transition_stage(version.version, "Staging")
        >>> # Compare with production
        >>> comparison = registry.compare_versions(v1=2, v2=1)
        >>> # Promote to production
        >>> registry.transition_stage(version.version, "Production")
    """

    VALID_STAGES = {"None", "Staging", "Production", "Archived"}

    def __init__(
        self,
        model_name: str = "stocksense-predictor",
        tracking_uri: str = "mlruns",
    ) -> None:
        """Initialize Model Registry.

        Args:
            model_name: Name for the registered model.
            tracking_uri: MLflow tracking server URI.
        """
        self.model_name = model_name
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

        # Ensure model is registered
        try:
            self.client.get_registered_model(model_name)
        except mlflow.exceptions.MlflowException:
            self.client.create_registered_model(
                model_name,
                description="StockSense AI prediction model",
            )
            logger.info(f"Created registered model: '{model_name}'")

        logger.info(f"ModelRegistry initialized for '{model_name}'")

    def register(
        self,
        run_id: str,
        artifact_path: str = "model",
        description: str = "",
    ) -> ModelVersion:
        """Register a new model version from a training run.

        Args:
            run_id: MLflow run ID containing the model artifact.
            artifact_path: Path within run artifacts where model is stored.
            description: Description of this version's changes.

        Returns:
            ModelVersion object for the newly registered version.
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"

        result = mlflow.register_model(
            model_uri=model_uri,
            name=self.model_name,
        )

        # Update description
        if description:
            self.client.update_model_version(
                name=self.model_name,
                version=result.version,
                description=description,
            )

        version = ModelVersion(
            name=self.model_name,
            version=int(result.version),
            stage=result.current_stage,
            run_id=run_id,
            description=description,
        )

        logger.info(
            f"  Registered model version {version.version} "
            f"(run: {run_id[:8]}...)"
        )
        return version

    def transition_stage(
        self,
        version: int,
        stage: str,
        archive_existing: bool = True,
    ) -> None:
        """Transition a model version to a new stage.

        Args:
            version: Model version number.
            stage: Target stage (Staging, Production, Archived).
            archive_existing: If True, archive existing models in target stage.

        Raises:
            ValueError: If stage is invalid.
        """
        if stage not in self.VALID_STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {self.VALID_STAGES}")

        self.client.transition_model_version_stage(
            name=self.model_name,
            version=str(version),
            stage=stage,
            archive_existing_versions=archive_existing,
        )

        logger.info(f"  Model v{version} → {stage}")

    def get_production_model(self) -> Optional[Any]:
        """Load the current production model.

        Returns:
            Loaded PyTorch model, or None if no production model exists.
        """
        try:
            model_uri = f"models:/{self.model_name}/Production"
            model = mlflow.pytorch.load_model(model_uri)
            logger.info("Loaded production model")
            return model
        except Exception as e:
            logger.warning(f"No production model found: {e}")
            return None

    def get_staging_model(self) -> Optional[Any]:
        """Load the current staging model.

        Returns:
            Loaded PyTorch model, or None if no staging model exists.
        """
        try:
            model_uri = f"models:/{self.model_name}/Staging"
            model = mlflow.pytorch.load_model(model_uri)
            logger.info("Loaded staging model")
            return model
        except Exception as e:
            logger.warning(f"No staging model found: {e}")
            return None

    def compare_versions(
        self,
        v1: int,
        v2: int,
    ) -> dict[str, Any]:
        """Compare two model versions by their training metrics.

        Useful for promotion decisions: compare staging vs production.

        Args:
            v1: First version number.
            v2: Second version number.

        Returns:
            Comparison dictionary with metrics for both versions.
        """
        def _get_version_metrics(version: int) -> dict:
            mv = self.client.get_model_version(self.model_name, str(version))
            run = self.client.get_run(mv.run_id)
            return {
                "version": version,
                "stage": mv.current_stage,
                "metrics": run.data.metrics,
                "params": run.data.params,
            }

        v1_data = _get_version_metrics(v1)
        v2_data = _get_version_metrics(v2)

        # Determine which is better for each metric
        improvements: dict[str, str] = {}
        for metric in set(v1_data["metrics"]) & set(v2_data["metrics"]):
            v1_val = v1_data["metrics"][metric]
            v2_val = v2_data["metrics"][metric]
            if "loss" in metric or "mse" in metric or "mae" in metric:
                # Lower is better
                winner = f"v{v1}" if v1_val < v2_val else f"v{v2}"
            else:
                # Higher is better
                winner = f"v{v1}" if v1_val > v2_val else f"v{v2}"
            improvements[metric] = winner

        comparison = {
            "v1": v1_data,
            "v2": v2_data,
            "improvements": improvements,
        }

        logger.info(f"  Compared v{v1} vs v{v2}: {improvements}")
        return comparison

    def list_versions(self) -> list[ModelVersion]:
        """List all versions of the registered model.

        Returns:
            List of ModelVersion objects sorted by version number.
        """
        versions = self.client.search_model_versions(f"name='{self.model_name}'")

        return [
            ModelVersion(
                name=self.model_name,
                version=int(v.version),
                stage=v.current_stage,
                run_id=v.run_id,
                description=v.description or "",
            )
            for v in sorted(versions, key=lambda x: int(x.version), reverse=True)
        ]

    def get_config(self) -> dict:
        """Return registry configuration for logging."""
        versions = self.list_versions()
        return {
            "model_name": self.model_name,
            "total_versions": len(versions),
            "production_versions": sum(1 for v in versions if v.stage == "Production"),
            "staging_versions": sum(1 for v in versions if v.stage == "Staging"),
        }
