"""Inference benchmark suite for comparing model variants.

Provides comprehensive performance comparison across:
    - PyTorch FP32 (baseline)
    - PyTorch INT8 (quantized)
    - ONNX Runtime (optimized)

Measures latency, throughput, memory, and model size
to make informed deployment decisions.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run.

    Attributes:
        name: Model variant name.
        mean_latency_ms: Average inference time in milliseconds.
        std_latency_ms: Standard deviation of latency.
        p50_ms: Median latency.
        p95_ms: 95th percentile latency.
        p99_ms: 99th percentile latency.
        throughput_rps: Requests per second.
        model_size_mb: Model size in megabytes.
        batch_size: Batch size used for benchmarking.
    """

    name: str
    mean_latency_ms: float
    std_latency_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_rps: float
    model_size_mb: float
    batch_size: int

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/API."""
        return {
            "name": self.name,
            "mean_latency_ms": round(self.mean_latency_ms, 3),
            "p95_latency_ms": round(self.p95_ms, 3),
            "p99_latency_ms": round(self.p99_ms, 3),
            "throughput_rps": round(self.throughput_rps, 1),
            "model_size_mb": round(self.model_size_mb, 2),
            "batch_size": self.batch_size,
        }


@dataclass
class BenchmarkComparison:
    """Comparison across all benchmarked model variants.

    Attributes:
        results: Individual benchmark results.
        baseline: Name of the baseline model for comparison.
    """

    results: list[BenchmarkResult] = field(default_factory=list)
    baseline: str = "pytorch_fp32"

    def summary_table(self) -> str:
        """Generate a formatted comparison table.

        Returns:
            String table comparing all model variants.
        """
        header = (
            f"{'Model':<20} {'Latency(ms)':<14} {'P95(ms)':<10} "
            f"{'RPS':<10} {'Size(MB)':<10} {'Speedup':<10}"
        )
        separator = "─" * len(header)

        lines = ["\n📊 Inference Benchmark Results", separator, header, separator]

        # Find baseline latency for speedup calculation
        baseline_latency = None
        for r in self.results:
            if r.name == self.baseline:
                baseline_latency = r.mean_latency_ms
                break

        for r in self.results:
            speedup = ""
            if baseline_latency and r.mean_latency_ms > 0:
                sp = baseline_latency / r.mean_latency_ms
                speedup = f"{sp:.2f}x"

            line = (
                f"{r.name:<20} {r.mean_latency_ms:<14.3f} {r.p95_ms:<10.3f} "
                f"{r.throughput_rps:<10.1f} {r.model_size_mb:<10.2f} {speedup:<10}"
            )
            lines.append(line)

        lines.append(separator)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert comparison to dictionary."""
        return {
            "results": [r.to_dict() for r in self.results],
            "baseline": self.baseline,
        }


class InferenceBenchmark:
    """Comprehensive inference benchmarking suite.

    Benchmarks multiple model variants (PyTorch FP32, INT8 quantized,
    ONNX Runtime) and produces a detailed comparison report.

    Example:
        >>> bench = InferenceBenchmark(num_warmup=10, num_iterations=100)
        >>> bench.add_pytorch_model("fp32", model, input_shape=(1, 60, 20))
        >>> bench.add_pytorch_model("int8", quantized_model, input_shape=(1, 60, 20))
        >>> bench.add_onnx_model("onnx", onnx_session, input_shape=(1, 60, 20))
        >>> comparison = bench.run_all()
        >>> print(comparison.summary_table())
    """

    def __init__(
        self,
        num_warmup: int = 10,
        num_iterations: int = 100,
    ) -> None:
        """Initialize benchmark suite.

        Args:
            num_warmup: Warmup iterations before timing.
            num_iterations: Timed iterations for measurement.
        """
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations
        self._models: list[dict[str, Any]] = []

    def add_pytorch_model(
        self,
        name: str,
        model: nn.Module,
        input_shape: tuple[int, ...],
    ) -> None:
        """Register a PyTorch model for benchmarking.

        Args:
            name: Model variant name.
            model: PyTorch model.
            input_shape: Shape of input tensor (including batch dim).
        """
        model.eval()
        model_size = self._get_model_size_mb(model)

        def predict_fn(x: NDArray) -> NDArray:
            with torch.no_grad():
                tensor = torch.tensor(x, dtype=torch.float32)
                return model(tensor).numpy()

        self._models.append({
            "name": name,
            "predict_fn": predict_fn,
            "input_shape": input_shape,
            "model_size_mb": model_size,
        })
        logger.info(f"  Added PyTorch model '{name}' ({model_size:.2f} MB)")

    def add_onnx_model(
        self,
        name: str,
        predict_fn: Callable[[NDArray], NDArray],
        input_shape: tuple[int, ...],
        model_size_mb: float = 0.0,
    ) -> None:
        """Register an ONNX model for benchmarking.

        Args:
            name: Model variant name.
            predict_fn: Function that takes numpy input and returns numpy output.
            input_shape: Shape of input array (including batch dim).
            model_size_mb: Model file size in MB.
        """
        self._models.append({
            "name": name,
            "predict_fn": predict_fn,
            "input_shape": input_shape,
            "model_size_mb": model_size_mb,
        })
        logger.info(f"  Added ONNX model '{name}' ({model_size_mb:.2f} MB)")

    def run_all(self, batch_sizes: Optional[list[int]] = None) -> BenchmarkComparison:
        """Run benchmarks for all registered models.

        Args:
            batch_sizes: List of batch sizes to test. Defaults to [1].

        Returns:
            BenchmarkComparison with results for all models.
        """
        if batch_sizes is None:
            batch_sizes = [1]

        comparison = BenchmarkComparison()

        for model_info in self._models:
            for batch_size in batch_sizes:
                name = (
                    f"{model_info['name']}_bs{batch_size}"
                    if len(batch_sizes) > 1
                    else model_info["name"]
                )

                result = self._benchmark_single(
                    name=name,
                    predict_fn=model_info["predict_fn"],
                    input_shape=model_info["input_shape"],
                    model_size_mb=model_info["model_size_mb"],
                    batch_size=batch_size,
                )
                comparison.results.append(result)

        # Print summary
        logger.info(comparison.summary_table())
        return comparison

    def _benchmark_single(
        self,
        name: str,
        predict_fn: Callable,
        input_shape: tuple[int, ...],
        model_size_mb: float,
        batch_size: int,
    ) -> BenchmarkResult:
        """Benchmark a single model variant.

        Args:
            name: Model name.
            predict_fn: Prediction function.
            input_shape: Input shape (batch dim will be replaced).
            model_size_mb: Model size.
            batch_size: Batch size for this run.

        Returns:
            BenchmarkResult with latency and throughput stats.
        """
        # Create input with correct batch size
        shape = (batch_size,) + input_shape[1:]
        test_input = np.random.randn(*shape).astype(np.float32)

        # Warmup
        for _ in range(self.num_warmup):
            predict_fn(test_input)

        # Timed runs
        latencies: list[float] = []
        for _ in range(self.num_iterations):
            start = time.perf_counter()
            predict_fn(test_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        latencies_arr = np.array(latencies)

        return BenchmarkResult(
            name=name,
            mean_latency_ms=float(latencies_arr.mean()),
            std_latency_ms=float(latencies_arr.std()),
            p50_ms=float(np.percentile(latencies_arr, 50)),
            p95_ms=float(np.percentile(latencies_arr, 95)),
            p99_ms=float(np.percentile(latencies_arr, 99)),
            throughput_rps=float(1000 / latencies_arr.mean() * batch_size),
            model_size_mb=model_size_mb,
            batch_size=batch_size,
        )

    @staticmethod
    def _get_model_size_mb(model: nn.Module) -> float:
        """Calculate PyTorch model size in megabytes."""
        size_bytes = sum(
            p.nelement() * p.element_size() for p in model.parameters()
        ) + sum(
            b.nelement() * b.element_size() for b in model.buffers()
        )
        return size_bytes / (1024 * 1024)
