"""Custom evaluation metrics for stock prediction models.

Provides financial-domain-specific metrics beyond standard ML metrics,
including directional accuracy and risk-adjusted error measurements.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class PredictionMetrics:
    """Container for all prediction evaluation metrics."""

    mse: float
    rmse: float
    mae: float
    mape: float
    r_squared: float
    direction_accuracy: float

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary for MLflow logging."""
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "r_squared": self.r_squared,
            "direction_accuracy": self.direction_accuracy,
        }

    def __str__(self) -> str:
        """Human-readable metrics summary."""
        return (
            f"MSE: {self.mse:.6f} | RMSE: {self.rmse:.6f} | MAE: {self.mae:.6f} | "
            f"MAPE: {self.mape:.2f}% | R²: {self.r_squared:.4f} | "
            f"Direction Acc: {self.direction_accuracy:.2f}%"
        )


def mean_squared_error(y_true: NDArray, y_pred: NDArray) -> float:
    """Calculate Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: NDArray, y_pred: NDArray) -> float:
    """Calculate Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: NDArray, y_pred: NDArray) -> float:
    """Calculate Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_absolute_percentage_error(y_true: NDArray, y_pred: NDArray) -> float:
    """Calculate Mean Absolute Percentage Error.

    Handles zero values by adding a small epsilon.
    """
    epsilon = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)


def r_squared(y_true: NDArray, y_pred: NDArray) -> float:
    """Calculate R-squared (coefficient of determination).

    Measures how well the model explains variance in the data.
    R² = 1 means perfect prediction, R² = 0 means no better than mean.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - (ss_res / ss_tot))


def direction_accuracy(y_true: NDArray, y_pred: NDArray) -> float:
    """Calculate directional accuracy (trend prediction accuracy).

    Measures the percentage of times the model correctly predicts
    the direction of price movement (up or down).

    This is arguably the most important metric in financial forecasting,
    since a trader cares more about direction than exact value.

    Args:
        y_true: Actual values as a 1D array.
        y_pred: Predicted values as a 1D array.

    Returns:
        Percentage of correct direction predictions (0-100).
    """
    if len(y_true) < 2:
        return 0.0

    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    return float(np.mean(true_direction == pred_direction) * 100)


def calculate_all_metrics(y_true: NDArray, y_pred: NDArray) -> PredictionMetrics:
    """Calculate all prediction metrics at once.

    Args:
        y_true: Array of actual values.
        y_pred: Array of predicted values.

    Returns:
        PredictionMetrics dataclass with all computed metrics.

    Example:
        >>> metrics = calculate_all_metrics(actual_prices, predicted_prices)
        >>> print(metrics)
        >>> mlflow.log_metrics(metrics.to_dict())
    """
    return PredictionMetrics(
        mse=mean_squared_error(y_true, y_pred),
        rmse=root_mean_squared_error(y_true, y_pred),
        mae=mean_absolute_error(y_true, y_pred),
        mape=mean_absolute_percentage_error(y_true, y_pred),
        r_squared=r_squared(y_true, y_pred),
        direction_accuracy=direction_accuracy(y_true, y_pred),
    )
