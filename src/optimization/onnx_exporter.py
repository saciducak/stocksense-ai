"""ONNX model export and optimization for cross-platform deployment.

Converts PyTorch models to ONNX format and applies graph-level
optimizations for production inference.

This addresses the job posting requirement:
    - "Modellerin gerçek zamanlı ve kısıtlı donanım ortamlarında kullanılabilirliği"

Why ONNX?
    1. Framework-agnostic: Deploy across different runtimes
    2. Graph optimization: Operator fusion, constant folding
    3. Hardware portability: CPU (OpenVINO), GPU (TensorRT), Apple (CoreML)
    4. Industry standard: Used by Azure ML, AWS SageMaker for serving
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ONNXExporter:
    """Export PyTorch models to ONNX and run optimized inference.

    Pipeline: PyTorch Model → ONNX Export → ONNX Runtime Inference

    Graph optimizations applied by ONNX Runtime:
        - Constant folding: Pre-compute constant expressions
        - Operator fusion: Combine sequential operations
        - Memory optimization: Reduce intermediate tensor allocations

    Example:
        >>> exporter = ONNXExporter()
        >>> onnx_path = exporter.export(model, sample_input)
        >>> predictions = exporter.predict(test_data)
        >>> latency = exporter.measure_latency(test_data)
    """

    def __init__(
        self,
        output_dir: str = "models/onnx",
        opset_version: int = 17,
    ) -> None:
        """Initialize ONNX exporter.

        Args:
            output_dir: Directory for exported ONNX models.
            opset_version: ONNX opset version (17 recommended for PyTorch 2.x).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.opset_version = opset_version
        self._session = None

    def export(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        filename: str = "model.onnx",
        dynamic_axes: bool = True,
    ) -> Path:
        """Export a PyTorch model to ONNX format.

        Args:
            model: PyTorch model to export.
            sample_input: Example input tensor for tracing.
            filename: Output ONNX model filename.
            dynamic_axes: Enable dynamic batch size support.

        Returns:
            Path to the exported ONNX model.
        """
        logger.info("Exporting model to ONNX format...")

        model.eval()
        model.cpu()
        sample_input = sample_input.cpu()

        export_path = self.output_dir / filename

        # Configure dynamic axes
        dynamic = None
        if dynamic_axes:
            dynamic = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }

        # Export
        torch.onnx.export(
            model,
            sample_input,
            str(export_path),
            export_params=True,
            opset_version=self.opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic,
        )

        # Validate exported model
        self._validate_onnx(export_path)

        # Report size
        onnx_size = export_path.stat().st_size / 1024 / 1024
        logger.info(f"  ✅ ONNX model exported: {export_path} ({onnx_size:.2f} MB)")

        # Initialize ONNX Runtime session
        self._init_session(export_path)

        return export_path

    def _validate_onnx(self, model_path: Path) -> None:
        """Validate the exported ONNX model.

        Checks graph structure and consistency.
        Always validate after export — catches silent export errors.

        Args:
            model_path: Path to ONNX model file.
        """
        try:
            import onnx
            model = onnx.load(str(model_path))
            onnx.checker.check_model(model)
            logger.info("  ✓ ONNX model validation passed")
        except ImportError:
            logger.warning("  onnx package not installed, skipping validation")
        except Exception as e:
            logger.error(f"  ✗ ONNX validation failed: {e}")
            raise

    def _init_session(self, model_path: Path) -> None:
        """Initialize ONNX Runtime inference session.

        Configures execution providers in priority order:
        1. CUDA (NVIDIA GPU)
        2. CoreML (Apple Silicon)
        3. CPU (fallback)

        Args:
            model_path: Path to ONNX model.
        """
        try:
            import onnxruntime as ort

            # Select best available execution provider
            available_providers = ort.get_available_providers()
            providers = []
            if "CUDAExecutionProvider" in available_providers:
                providers.append("CUDAExecutionProvider")
            if "CoreMLExecutionProvider" in available_providers:
                providers.append("CoreMLExecutionProvider")
            providers.append("CPUExecutionProvider")

            # Session options for optimization
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4

            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=providers,
            )

            active_provider = self._session.get_providers()[0]
            logger.info(f"  ONNX Runtime session ready (provider: {active_provider})")

        except ImportError:
            logger.warning("  onnxruntime not installed, ONNX inference unavailable")
            self._session = None

    def predict(self, input_data: NDArray) -> NDArray:
        """Run inference using ONNX Runtime.

        Args:
            input_data: Input numpy array.

        Returns:
            Model output as numpy array.

        Raises:
            RuntimeError: If ONNX session is not initialized.
        """
        if self._session is None:
            raise RuntimeError("ONNX session not initialized. Call export() first.")

        input_name = self._session.get_inputs()[0].name
        input_data = input_data.astype(np.float32)

        outputs = self._session.run(None, {input_name: input_data})
        return outputs[0]

    def measure_latency(
        self,
        input_data: NDArray,
        num_warmup: int = 10,
        num_iterations: int = 100,
    ) -> dict:
        """Measure inference latency of the ONNX model.

        Performs warmup runs followed by timed iterations
        to get accurate latency measurements.

        Args:
            input_data: Input numpy array (single sample or batch).
            num_warmup: Number of warmup iterations.
            num_iterations: Number of timed iterations.

        Returns:
            Dictionary with latency statistics (mean, std, p50, p95, p99).
        """
        import time

        if self._session is None:
            raise RuntimeError("ONNX session not initialized.")

        input_name = self._session.get_inputs()[0].name
        input_data = input_data.astype(np.float32)
        feed = {input_name: input_data}

        # Warmup
        for _ in range(num_warmup):
            self._session.run(None, feed)

        # Timed runs
        latencies: list[float] = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self._session.run(None, feed)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

        latencies_arr = np.array(latencies)

        stats = {
            "mean_ms": float(latencies_arr.mean()),
            "std_ms": float(latencies_arr.std()),
            "p50_ms": float(np.percentile(latencies_arr, 50)),
            "p95_ms": float(np.percentile(latencies_arr, 95)),
            "p99_ms": float(np.percentile(latencies_arr, 99)),
            "min_ms": float(latencies_arr.min()),
            "max_ms": float(latencies_arr.max()),
            "throughput_rps": float(1000 / latencies_arr.mean()),
        }

        logger.info(
            f"  ONNX Latency: mean={stats['mean_ms']:.2f}ms, "
            f"p95={stats['p95_ms']:.2f}ms, "
            f"throughput={stats['throughput_rps']:.1f} req/s"
        )

        return stats

    def load_session(self, model_path: str) -> None:
        """Load an existing ONNX model for inference.

        Args:
            model_path: Path to existing ONNX model file.
        """
        self._init_session(Path(model_path))
