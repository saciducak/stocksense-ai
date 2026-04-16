"""Model evaluation script with detailed metrics and visualizations.

Usage:
    python scripts/evaluate.py --config configs/config.yaml
    python scripts/evaluate.py --config configs/config.yaml --model lstm
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.price_fetcher import PriceFetcher
from src.data.feature_engineer import FeatureEngineer
from src.data.preprocessor import DataPreprocessor
from src.models.lstm_model import LSTMPredictor
from src.models.transformer_model import TransformerPredictor
from src.utils.logger import get_logger
from src.utils.metrics import calculate_all_metrics

logger = get_logger(__name__)


def load_model(model_type: str, num_features: int, config: dict) -> torch.nn.Module:
    """Load a trained model from disk."""
    forecast_horizon = config["data"]["forecast_horizon"]
    model_path = Path("models") / model_type / "model.pt"

    if model_type == "lstm":
        model_cfg = config["models"]["lstm"]
        model = LSTMPredictor(
            input_size=num_features,
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
            bidirectional=model_cfg["bidirectional"],
            forecast_horizon=forecast_horizon,
        )
    elif model_type == "transformer":
        model_cfg = config["models"]["transformer"]
        model = TransformerPredictor(
            input_size=num_features,
            d_model=model_cfg["d_model"],
            nhead=model_cfg["nhead"],
            num_encoder_layers=model_cfg["num_encoder_layers"],
            dim_feedforward=model_cfg["dim_feedforward"],
            dropout=model_cfg["dropout"],
            forecast_horizon=forecast_horizon,
        )
    else:
        raise ValueError(f"Unknown model: {model_type}")

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        logger.info(f"  Loaded weights from {model_path}")
    else:
        logger.warning(f"  No saved weights found at {model_path}, using random initialization")

    model.eval()
    return model


def main() -> None:
    """Run model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate StockSense AI models")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, default="all", choices=["lstm", "transformer", "all"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Prepare data (same pipeline as training)
    data_cfg = config["data"]
    feat_cfg = config["features"]["technical_indicators"]

    fetcher = PriceFetcher(
        tickers=data_cfg["tickers"],
        period=data_cfg["period"],
        interval=data_cfg["interval"],
    )
    stock_data = fetcher.fetch()

    engineer = FeatureEngineer(
        sma_periods=feat_cfg["sma_periods"],
        ema_periods=feat_cfg["ema_periods"],
        rsi_period=feat_cfg["rsi_period"],
    )
    features = engineer.compute_all(stock_data.data)

    preprocessor = DataPreprocessor(
        sequence_length=data_cfg["sequence_length"],
        forecast_horizon=data_cfg["forecast_horizon"],
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
    )
    split = preprocessor.fit_transform(stock_data.data, additional_features=features)
    num_features = split.train.num_features

    # Evaluate models
    model_types = ["lstm", "transformer"] if args.model == "all" else [args.model]

    logger.info("=" * 60)
    logger.info("MODEL EVALUATION REPORT")
    logger.info("=" * 60)

    results: dict[str, dict] = {}

    for model_type in model_types:
        logger.info(f"\n--- {model_type.upper()} ---")

        model = load_model(model_type, num_features, config)

        # Generate predictions on test set
        test_X = torch.tensor(split.test.X)
        test_y = split.test.y

        with torch.no_grad():
            predictions = model(test_X).numpy()

        # Inverse transform
        pred_original = preprocessor.inverse_transform_target(predictions.flatten())
        true_original = preprocessor.inverse_transform_target(test_y.flatten())

        # Calculate metrics
        metrics = calculate_all_metrics(true_original, pred_original)
        results[model_type] = metrics.to_dict()

        logger.info(f"  {metrics}")

    # Comparison table
    if len(results) > 1:
        logger.info("\n" + "=" * 60)
        logger.info("COMPARISON")
        logger.info("=" * 60)
        for name, m in results.items():
            logger.info(
                f"  {name:15s} | MSE: {m['mse']:.6f} | "
                f"Dir Acc: {m['direction_accuracy']:.1f}% | "
                f"R²: {m['r_squared']:.4f}"
            )

    logger.info("\n✅ Evaluation complete")


if __name__ == "__main__":
    main()
