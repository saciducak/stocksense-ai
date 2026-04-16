"""Tests for inference optimization modules."""

import numpy as np
import pytest
import torch

from src.models.lstm_model import LSTMPredictor
from src.models.transformer_model import TransformerPredictor
from src.optimization.quantizer import ModelQuantizer
from src.optimization.benchmark import InferenceBenchmark, BenchmarkComparison


# ─── Quantizer Tests ──────────────────────────────────────────────────────────

class TestModelQuantizer:
    """Test model quantization."""

    @pytest.fixture
    def lstm_model(self) -> LSTMPredictor:
        """Create a small LSTM for testing."""
        return LSTMPredictor(input_size=5, hidden_size=32, num_layers=1, forecast_horizon=3)

    def test_dynamic_quantization(self, lstm_model: LSTMPredictor) -> None:
        """Quantized model should be smaller than original."""
        quantizer = ModelQuantizer(output_dir="/tmp/test_quantized")
        quantized = quantizer.quantize_dynamic(lstm_model)

        orig_size = quantizer._get_model_size(lstm_model)
        quant_size = quantizer._get_model_size(quantized)

        # Quantized should be smaller (or equal for very small models)
        assert quant_size <= orig_size

    def test_quantized_output_shape(self, lstm_model: LSTMPredictor) -> None:
        """Quantized model should produce same output shape."""
        quantizer = ModelQuantizer(output_dir="/tmp/test_quantized")
        quantized = quantizer.quantize_dynamic(lstm_model)

        x = torch.randn(4, 30, 5)
        output = quantized(x)
        assert output.shape == (4, 3)

    def test_model_size_calculation(self, lstm_model: LSTMPredictor) -> None:
        """Model size should be positive."""
        quantizer = ModelQuantizer()
        size = quantizer._get_model_size(lstm_model)
        assert size > 0


# ─── Benchmark Tests ──────────────────────────────────────────────────────────

class TestInferenceBenchmark:
    """Test inference benchmarking suite."""

    def test_benchmark_single_model(self) -> None:
        """Should benchmark a single PyTorch model."""
        model = LSTMPredictor(input_size=5, hidden_size=16, num_layers=1, forecast_horizon=3)
        model.eval()

        bench = InferenceBenchmark(num_warmup=2, num_iterations=5)
        bench.add_pytorch_model("test_lstm", model, input_shape=(1, 30, 5))

        comparison = bench.run_all()
        assert len(comparison.results) == 1
        assert comparison.results[0].mean_latency_ms > 0
        assert comparison.results[0].throughput_rps > 0

    def test_multiple_models(self) -> None:
        """Should benchmark multiple models."""
        model1 = LSTMPredictor(input_size=5, hidden_size=16, num_layers=1, forecast_horizon=3)
        model2 = TransformerPredictor(
            input_size=5, d_model=32, nhead=4, num_encoder_layers=1, forecast_horizon=3,
        )
        model1.eval()
        model2.eval()

        bench = InferenceBenchmark(num_warmup=2, num_iterations=5)
        bench.add_pytorch_model("lstm", model1, input_shape=(1, 30, 5))
        bench.add_pytorch_model("transformer", model2, input_shape=(1, 30, 5))

        comparison = bench.run_all()
        assert len(comparison.results) == 2

    def test_summary_table(self) -> None:
        """Summary table should be a non-empty string."""
        comparison = BenchmarkComparison(results=[])
        table = comparison.summary_table()
        assert isinstance(table, str)
        assert "Benchmark" in table

    def test_multiple_batch_sizes(self) -> None:
        """Should test with different batch sizes."""
        model = LSTMPredictor(input_size=5, hidden_size=16, num_layers=1, forecast_horizon=3)
        model.eval()

        bench = InferenceBenchmark(num_warmup=2, num_iterations=3)
        bench.add_pytorch_model("test", model, input_shape=(1, 30, 5))

        comparison = bench.run_all(batch_sizes=[1, 4])
        assert len(comparison.results) == 2
