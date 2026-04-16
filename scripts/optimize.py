"""Inference optimization pipeline: Quantize → ONNX Export → Benchmark.

Usage:
    python scripts/optimize.py --config configs/config.yaml
    python scripts/optimize.py --config configs/config.yaml --benchmark-only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.transformer_model import TransformerPredictor
from src.optimization.quantizer import ModelQuantizer
from src.optimization.onnx_exporter import ONNXExporter
from src.optimization.benchmark import InferenceBenchmark
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run the full inference optimization pipeline."""
    parser = argparse.ArgumentParser(description="Optimize models for inference")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--model", type=str, default="transformer")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load model
    model_cfg = config["models"][args.model]
    opt_cfg = config["optimization"]

    forecast_horizon = config["data"]["forecast_horizon"]

    # Try loading saved weights first to infer num_features
    weights_path = Path("models") / args.model / "model.pt"
    num_features = 24  # Default fallback feature count
    
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        if args.model == "transformer" and "input_projection.0.weight" in state_dict:
            num_features = state_dict["input_projection.0.weight"].size(1)
            logger.info(f"Inferred num_features={num_features} from checkpoint")
        elif args.model == "lstm" and "lstm.weight_ih_l0" in state_dict:
            num_features = state_dict["lstm.weight_ih_l0"].size(1)
            logger.info(f"Inferred num_features={num_features} from checkpoint")
    

    model = TransformerPredictor(
        input_size=num_features,
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_encoder_layers=model_cfg["num_encoder_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
        forecast_horizon=forecast_horizon,
    )

    # Try loading saved weights
    weights_path = Path("models") / args.model / "model.pt"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        logger.info(f"Loaded model weights from {weights_path}")
    else:
        logger.warning("No saved weights found, using random initialization for demo")

    model.eval()

    # Sample input for export/benchmark
    seq_length = config["data"]["sequence_length"]
    sample_input = torch.randn(1, seq_length, num_features)

    if not args.benchmark_only:
        # ─── Step 1: Quantization ──────────────────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: Dynamic Quantization (FP32 → INT8)")
        logger.info("=" * 60)

        quantizer = ModelQuantizer(output_dir="models/quantized")
        quantized_model = quantizer.quantize_dynamic(model)

        # Compare accuracy
        test_data = torch.randn(50, seq_length, num_features)
        with torch.no_grad():
            test_labels = model(test_data)

        comparison = quantizer.compare_accuracy(
            original=model,
            quantized=quantized_model,
            test_data=test_data,
            test_labels=test_labels,
        )

        # Save quantized model
        quantizer.save_quantized(quantized_model)

        # ─── Step 2: ONNX Export ───────────────────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: ONNX Export & Optimization")
        logger.info("=" * 60)

        exporter = ONNXExporter(
            output_dir="models/onnx",
            opset_version=opt_cfg["onnx"]["opset_version"],
        )
        onnx_path = exporter.export(
            model=model,
            sample_input=sample_input,
            dynamic_axes=opt_cfg["onnx"]["dynamic_axes"],
        )

    # ─── Step 3: Benchmark ─────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Inference Benchmark")
    logger.info("=" * 60)

    bench = InferenceBenchmark(
        num_warmup=opt_cfg["benchmark"]["num_warmup"],
        num_iterations=opt_cfg["benchmark"]["num_iterations"],
    )

    # Add PyTorch FP32 model
    bench.add_pytorch_model(
        "pytorch_fp32", model, input_shape=(1, seq_length, num_features),
    )

    # Add quantized model if available
    if not args.benchmark_only:
        bench.add_pytorch_model(
            "pytorch_int8", quantized_model, input_shape=(1, seq_length, num_features),
        )

        # Add ONNX model if available
        try:
            onnx_predict = exporter.predict
            onnx_size = onnx_path.stat().st_size / 1024 / 1024
            bench.add_onnx_model(
                "onnx_optimized", onnx_predict,
                input_shape=(1, seq_length, num_features),
                model_size_mb=onnx_size,
            )
        except Exception as e:
            logger.warning(f"ONNX benchmark skipped: {e}")

        # PyTorch 2.1 ONNX exporter does not properly track dynamic axes across 
        # MultiheadAttention internal reshapes for batch_first=True.
        batch_sizes = [1] if model.__class__.__name__ == "TransformerPredictor" else opt_cfg["benchmark"]["batch_sizes"]
        
        comparison = bench.run_all(batch_sizes=batch_sizes)
    else:
        # Run benchmarks
        comparison = bench.run_all(batch_sizes=opt_cfg["benchmark"]["batch_sizes"])

    # Save results
    results_path = Path("models") / "benchmark_results.yaml"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        yaml.dump(comparison.to_dict(), f, default_flow_style=False)

    logger.info(f"\n💾 Results saved to {results_path}")
    logger.info("🎉 Optimization pipeline complete!")


if __name__ == "__main__":
    main()
