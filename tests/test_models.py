"""Tests for prediction models."""

import pytest
import torch

from src.models.lstm_model import LSTMPredictor
from src.models.transformer_model import TransformerPredictor, PositionalEncoding, create_causal_mask
from src.models.ensemble import EnsemblePredictor

import numpy as np


# ─── LSTM Tests ───────────────────────────────────────────────────────────────

class TestLSTMPredictor:
    """Test LSTM model architecture and forward pass."""

    @pytest.fixture
    def model(self) -> LSTMPredictor:
        """Create LSTM model for testing."""
        return LSTMPredictor(
            input_size=10,
            hidden_size=64,
            num_layers=2,
            forecast_horizon=5,
        )

    def test_output_shape(self, model: LSTMPredictor) -> None:
        """Output should have shape (batch, forecast_horizon)."""
        x = torch.randn(8, 60, 10)
        output = model(x)
        assert output.shape == (8, 5)

    def test_single_sample(self, model: LSTMPredictor) -> None:
        """Should work with single sample."""
        x = torch.randn(1, 60, 10)
        output = model(x)
        assert output.shape == (1, 5)

    def test_different_sequence_lengths(self, model: LSTMPredictor) -> None:
        """Should handle different sequence lengths."""
        for seq_len in [30, 60, 120]:
            x = torch.randn(4, seq_len, 10)
            output = model(x)
            assert output.shape == (4, 5)

    def test_get_config(self, model: LSTMPredictor) -> None:
        """Config should contain key hyperparameters."""
        config = model.get_config()
        assert config["model_type"] == "LSTM"
        assert "hidden_size" in config
        assert "total_params" in config
        assert config["total_params"] > 0

    def test_gradient_flow(self, model: LSTMPredictor) -> None:
        """Gradients should flow through the model."""
        x = torch.randn(4, 60, 10)
        y = torch.randn(4, 5)

        output = model(x)
        loss = torch.nn.MSELoss()(output, y)
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


# ─── Transformer Tests ────────────────────────────────────────────────────────

class TestTransformerPredictor:
    """Test Transformer model architecture."""

    @pytest.fixture
    def model(self) -> TransformerPredictor:
        """Create Transformer model for testing."""
        return TransformerPredictor(
            input_size=10,
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            dim_feedforward=128,
            forecast_horizon=5,
        )

    def test_output_shape(self, model: TransformerPredictor) -> None:
        """Output should have shape (batch, forecast_horizon)."""
        x = torch.randn(8, 60, 10)
        output = model(x)
        assert output.shape == (8, 5)

    def test_single_sample(self, model: TransformerPredictor) -> None:
        """Should work with single sample."""
        x = torch.randn(1, 60, 10)
        output = model(x)
        assert output.shape == (1, 5)

    def test_attention_weights(self, model: TransformerPredictor) -> None:
        """Should extract attention weights for interpretability."""
        x = torch.randn(2, 60, 10)
        weights = model.get_attention_weights(x)

        assert len(weights) == 2  # num_encoder_layers
        assert weights[0].shape[0] == 2  # batch_size

    def test_causal_mask(self) -> None:
        """Causal mask should be upper triangular with -inf."""
        mask = create_causal_mask(10)
        assert mask.shape == (10, 10)
        # Diagonal should be 0
        assert mask[0, 0] == 0
        # Upper triangle should be -inf
        assert mask[0, 1] == float("-inf")

    def test_get_config(self, model: TransformerPredictor) -> None:
        """Config should contain Transformer-specific params."""
        config = model.get_config()
        assert config["model_type"] == "Transformer"
        assert config["d_model"] == 64
        assert config["nhead"] == 4
        assert config["num_layers"] == 2


class TestPositionalEncoding:
    """Test positional encoding module."""

    def test_output_shape(self) -> None:
        """PE should maintain input shape."""
        pe = PositionalEncoding(d_model=64)
        x = torch.randn(4, 60, 64)
        output = pe(x)
        assert output.shape == x.shape

    def test_adds_positional_info(self) -> None:
        """Output should differ from input (positional info added)."""
        pe = PositionalEncoding(d_model=64, dropout=0.0)
        x = torch.zeros(1, 10, 64)
        output = pe(x)
        # Should not be all zeros anymore
        assert not torch.allclose(output, x)


# ─── Ensemble Tests ───────────────────────────────────────────────────────────

class TestEnsemblePredictor:
    """Test ensemble model."""

    def test_predict(self) -> None:
        """Predict should return EnsemblePrediction."""
        ensemble = EnsemblePredictor(forecast_horizon=5)
        result = ensemble.predict(
            lstm_pred=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            transformer_pred=np.array([1.1, 2.1, 3.1, 4.1, 5.1]),
            sentiment_score=0.5,
        )

        assert len(result.prediction) == 5
        assert 0 <= result.confidence <= 1
        assert "lstm" in result.weights_used

    def test_weights_validation(self) -> None:
        """Should raise error if weights don't sum to 1."""
        with pytest.raises(ValueError):
            EnsemblePredictor(weights={"lstm": 0.5, "transformer": 0.3, "sentiment": 0.1})

    def test_confidence_high_agreement(self) -> None:
        """High agreement between models should give high confidence."""
        ensemble = EnsemblePredictor()
        result = ensemble.predict(
            lstm_pred=np.array([1.0, 1.0, 1.0]),
            transformer_pred=np.array([1.0, 1.0, 1.0]),
        )
        assert result.confidence > 0.8

    def test_confidence_low_agreement(self) -> None:
        """Low agreement should give lower confidence."""
        ensemble = EnsemblePredictor()
        result = ensemble.predict(
            lstm_pred=np.array([1.0, 2.0, 3.0]),
            transformer_pred=np.array([10.0, 20.0, 30.0]),
        )
        assert result.confidence < 0.5

    def test_to_dict(self) -> None:
        """Result should be serializable."""
        ensemble = EnsemblePredictor()
        result = ensemble.predict(
            lstm_pred=np.array([1.0, 2.0]),
            transformer_pred=np.array([1.1, 2.1]),
        )
        d = result.to_dict()
        assert "prediction" in d
        assert "confidence" in d
