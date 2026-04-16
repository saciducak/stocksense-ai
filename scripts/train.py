"""Model training script with MLflow experiment tracking.

Usage:
    python scripts/train.py --config configs/config.yaml --model transformer
    python scripts/train.py --config configs/config.yaml --model lstm
    python scripts/train.py --config configs/config.yaml --model all
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.price_fetcher import PriceFetcher
from src.data.feature_engineer import FeatureEngineer
from src.data.preprocessor import DataPreprocessor
from src.models.lstm_model import LSTMPredictor
from src.models.transformer_model import TransformerPredictor
from src.mlops.experiment_tracker import ExperimentTracker
from src.utils.logger import get_logger
from src.utils.metrics import calculate_all_metrics

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_data(config: dict) -> tuple:
    """Fetch, engineer features, and preprocess data.

    Returns:
        Tuple of (split_dataset, preprocessor, num_features).
    """
    data_cfg = config["data"]
    feat_cfg = config["features"]["technical_indicators"]

    # Fetch stock data
    logger.info("=" * 60)
    logger.info("PHASE 1: Data Preparation")
    logger.info("=" * 60)

    fetcher = PriceFetcher(
        tickers=data_cfg["tickers"],
        period=data_cfg["period"],
        interval=data_cfg["interval"],
    )
    stock_data = fetcher.fetch()

    # Compute technical indicators
    engineer = FeatureEngineer(
        sma_periods=feat_cfg["sma_periods"],
        ema_periods=feat_cfg["ema_periods"],
        rsi_period=feat_cfg["rsi_period"],
        macd_fast=feat_cfg["macd"]["fast"],
        macd_slow=feat_cfg["macd"]["slow"],
        macd_signal=feat_cfg["macd"]["signal"],
        bb_period=feat_cfg["bollinger"]["period"],
        bb_std=feat_cfg["bollinger"]["std_dev"],
    )
    features = engineer.compute_all(stock_data.data)

    # Preprocess and create sequences
    preprocessor = DataPreprocessor(
        sequence_length=data_cfg["sequence_length"],
        forecast_horizon=data_cfg["forecast_horizon"],
        target_column="Close",
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
    )
    split = preprocessor.fit_transform(stock_data.data, additional_features=features)

    return split, preprocessor, split.train.num_features


def create_dataloaders(split, batch_size: int) -> tuple:
    """Create PyTorch DataLoaders from split dataset."""
    train_ds = TensorDataset(
        torch.tensor(split.train.X), torch.tensor(split.train.y),
    )
    val_ds = TensorDataset(
        torch.tensor(split.val.X), torch.tensor(split.val.y),
    )
    test_ds = TensorDataset(
        torch.tensor(split.test.X), torch.tensor(split.test.y),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def create_model(model_type: str, num_features: int, config: dict) -> nn.Module:
    """Create a model based on type and configuration."""
    forecast_horizon = config["data"]["forecast_horizon"]

    if model_type == "lstm":
        model_cfg = config["models"]["lstm"]
        return LSTMPredictor(
            input_size=num_features,
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
            bidirectional=model_cfg["bidirectional"],
            forecast_horizon=forecast_horizon,
        )
    elif model_type == "transformer":
        model_cfg = config["models"]["transformer"]
        return TransformerPredictor(
            input_size=num_features,
            d_model=model_cfg["d_model"],
            nhead=model_cfg["nhead"],
            num_encoder_layers=model_cfg["num_encoder_layers"],
            dim_feedforward=model_cfg["dim_feedforward"],
            dropout=model_cfg["dropout"],
            forecast_horizon=forecast_horizon,
            activation=model_cfg["activation"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    tracker: ExperimentTracker,
    model_type: str,
) -> nn.Module:
    """Train the model with early stopping and MLflow tracking.

    Args:
        model: PyTorch model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training configuration.
        tracker: MLflow experiment tracker.
        model_type: Model type name for logging.

    Returns:
        Trained model with best weights restored.
    """
    train_cfg = config["training"]

    # Device setup
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    model = model.to(device)
    logger.info(f"Training on device: {device}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"],
    )

    criterion = nn.MSELoss()

    # Early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    logger.info("=" * 60)
    logger.info(f"PHASE 2: Training ({model_type})")
    logger.info("=" * 60)

    for epoch in range(train_cfg["epochs"]):
        # ─── Training ─────────────────────────────────────────────
        model.train()
        train_losses: list[float] = []

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()

            # Gradient clipping
            if train_cfg.get("gradient_clip"):
                nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg["gradient_clip"],
                )

            optimizer.step()
            train_losses.append(loss.item())

        # ─── Validation ───────────────────────────────────────────
        model.eval()
        val_losses: list[float] = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        # Log to MLflow
        tracker.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }, step=epoch)

        scheduler.step()

        # Log every 10 epochs
        if epoch % 10 == 0 or epoch == train_cfg["epochs"] - 1:
            logger.info(
                f"  Epoch {epoch:3d}/{train_cfg['epochs']} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

        # Early stopping
        if val_loss < best_val_loss - train_cfg["early_stopping"]["min_delta"]:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= train_cfg["early_stopping"]["patience"]:
                logger.info(
                    f"  ⏹ Early stopping at epoch {epoch} "
                    f"(patience: {train_cfg['early_stopping']['patience']})"
                )
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    logger.info(f"  ✅ Best validation loss: {best_val_loss:.6f}")
    return model


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    preprocessor: DataPreprocessor,
    tracker: ExperimentTracker,
) -> dict:
    """Evaluate model on test set and log metrics.

    Args:
        model: Trained model.
        test_loader: Test data loader.
        preprocessor: For inverse scaling.
        tracker: MLflow tracker for metric logging.

    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: Evaluation")
    logger.info("=" * 60)

    device = next(model.parameters()).device
    model.eval()

    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            predictions = model(batch_X).cpu().numpy()
            targets = batch_y.numpy()

            all_predictions.append(predictions)
            all_targets.append(targets)

    predictions = np.concatenate(all_predictions).flatten()
    targets = np.concatenate(all_targets).flatten()

    # Inverse transform to original scale
    predictions_original = preprocessor.inverse_transform_target(predictions)
    targets_original = preprocessor.inverse_transform_target(targets)

    # Calculate metrics
    metrics = calculate_all_metrics(targets_original, predictions_original)
    logger.info(f"  Test Metrics: {metrics}")

    # Log to MLflow
    tracker.log_metrics(metrics.to_dict())

    return metrics.to_dict()


def main() -> None:
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train StockSense AI models")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, default="transformer",
                        choices=["lstm", "transformer", "all"])
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    torch.manual_seed(config["project"]["seed"])
    np.random.seed(config["project"]["seed"])

    # Prepare data
    split, preprocessor, num_features = prepare_data(config)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        split, config["training"]["batch_size"],
    )

    # Initialize MLflow tracker
    tracker = ExperimentTracker(
        experiment_name=config["mlops"]["experiment_name"],
        tracking_uri=config["mlops"]["tracking_uri"],
    )

    # Determine which models to train
    model_types = ["lstm", "transformer"] if args.model == "all" else [args.model]

    for model_type in model_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training: {model_type.upper()}")
        logger.info(f"{'='*60}\n")

        # Create model
        model = create_model(model_type, num_features, config)

        # Start MLflow run
        with tracker.start_run(run_name=f"{model_type}-training"):
            # Log parameters
            tracker.log_params({
                **model.get_config(),
                "sequence_length": config["data"]["sequence_length"],
                "forecast_horizon": config["data"]["forecast_horizon"],
                "batch_size": config["training"]["batch_size"],
                "learning_rate": config["training"]["learning_rate"],
                "epochs": config["training"]["epochs"],
            })

            # Train
            model = train_model(
                model, train_loader, val_loader, config, tracker, model_type,
            )

            # Evaluate
            metrics = evaluate_model(model, test_loader, preprocessor, tracker)

            # Save model
            tracker.log_model(model, artifact_path=f"{model_type}_model")

            # Save model locally
            save_dir = Path("models") / model_type
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "model.pt")
            logger.info(f"  💾 Model saved to {save_dir}/model.pt")

    logger.info("\n🎉 Training complete! View results: make mlflow-ui")


if __name__ == "__main__":
    main()
