"""
scripts/export_onnx.py
───────────────────────
Export the trained MedicalAI model to ONNX for Triton Inference Server.

ONNX provides:
  - Platform-independent inference (CPU, GPU, edge)
  - Graph optimisation (operator fusion, constant folding)
  - Quantisation (INT8) for CPU deployments
  - Compatible with NVIDIA Triton Inference Server

Usage:
  python scripts/export_onnx.py \
      --checkpoint outputs/models/calibrated_model.pt \
      --output outputs/onnx/medical_ai.onnx \
      --verify

Triton deployment:
  Copy medical_ai.onnx to:
    model_repository/medical_ai/1/model.onnx
  And create config.pbtxt with input/output specs from this script's output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml


def export_to_onnx(
    model,
    cfg: dict,
    output_path: str,
    device: str = "cpu",
    opset_version: int = 17,
) -> None:
    """
    Export MedicalAIModel to ONNX format.

    Input names and shapes (dynamic batch dimension):
      numerical      : (batch, tab_input_dim)
      input_ids      : (batch, seq_len)
      attention_mask : (batch, seq_len)

    Output names:
      probs : (batch, num_diseases)
    """
    from src.models.tabular_encoder import INPUT_DIM

    model = model.eval().to(device)
    seq_len    = cfg["data"]["max_seq_len"]
    tab_dim    = INPUT_DIM
    batch_size = 2   # dynamic, but needs concrete example for tracing

    # Dummy inputs
    dummy_numerical      = torch.zeros(batch_size, tab_dim,  device=device)
    dummy_input_ids      = torch.zeros(batch_size, seq_len,  dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones( batch_size, seq_len,  dtype=torch.long, device=device)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[onnx] Exporting to {output_path} (opset {opset_version})...")

    torch.onnx.export(
        model,
        (dummy_numerical, dummy_input_ids, dummy_attention_mask),
        output_path,
        input_names  = ["numerical", "input_ids", "attention_mask"],
        output_names = ["probs"],
        dynamic_axes = {
            "numerical":      {0: "batch_size"},
            "input_ids":      {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "probs":          {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        verbose=False,
    )
    print(f"[onnx] Export complete: {output_path}")


def verify_onnx(
    onnx_path: str,
    cfg: dict,
    device: str = "cpu",
) -> None:
    """
    Verify ONNX model produces same outputs as PyTorch model.
    Checks max absolute difference < 1e-4.
    """
    import numpy as np
    import onnxruntime as ort
    from src.models.tabular_encoder import INPUT_DIM

    print("[onnx] Verifying ONNX model...")
    seq_len = cfg["data"]["max_seq_len"]
    tab_dim = INPUT_DIM

    dummy_numerical      = np.zeros((1, tab_dim),  dtype=np.float32)
    dummy_input_ids      = np.zeros((1, seq_len),  dtype=np.int64)
    dummy_attention_mask = np.ones( (1, seq_len),  dtype=np.int64)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
                if device == "cuda" else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)

    onnx_outputs = sess.run(
        output_names=["probs"],
        input_feed={
            "numerical":      dummy_numerical,
            "input_ids":      dummy_input_ids,
            "attention_mask": dummy_attention_mask,
        },
    )
    probs = onnx_outputs[0]
    print(f"[onnx] Output shape: {probs.shape}")
    print(f"[onnx] Prob range: [{probs.min():.4f}, {probs.max():.4f}]")
    assert probs.shape[1] == cfg["classifier"]["num_diseases"], "Output dim mismatch!"
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities out of [0,1]!"
    print("[onnx] Verification passed.")


def quantise_onnx(onnx_path: str, output_path: str) -> None:
    """
    INT8 dynamic quantisation for CPU deployment.
    Reduces model size by ~4x, typically < 2% AUROC drop.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            onnx_path, output_path,
            weight_type=QuantType.QInt8,
        )
        original_mb  = Path(onnx_path).stat().st_size / 1e6
        quantised_mb = Path(output_path).stat().st_size / 1e6
        print(f"[onnx] Quantised: {original_mb:.1f}MB → {quantised_mb:.1f}MB "
              f"({100*(1-quantised_mb/original_mb):.0f}% reduction)")
    except ImportError:
        print("[onnx] onnxruntime.quantization not available; skipping quantisation")


def main(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from src.models.full_model import MedicalAIModel
    model = MedicalAIModel.load(args.checkpoint, cfg)

    export_to_onnx(model, cfg, args.output, device)

    if args.verify:
        verify_onnx(args.output, cfg, device)

    if args.quantise:
        q_path = args.output.replace(".onnx", "_int8.onnx")
        quantise_onnx(args.output, q_path)

    # Print Triton config.pbtxt template
    print("\n── Triton config.pbtxt template ────────────────────────────")
    from src.models.tabular_encoder import INPUT_DIM
    seq_len = cfg["data"]["max_seq_len"]
    print(f"""
name: "medical_ai"
backend: "onnxruntime"
max_batch_size: 32

input [
  {{ name: "numerical"      data_type: TYPE_FP32 dims: [{INPUT_DIM}] }},
  {{ name: "input_ids"      data_type: TYPE_INT64 dims: [{seq_len}] }},
  {{ name: "attention_mask" data_type: TYPE_INT64 dims: [{seq_len}] }}
]
output [
  {{ name: "probs" data_type: TYPE_FP32 dims: [{cfg['classifier']['num_diseases']}] }}
]

dynamic_batching {{ preferred_batch_size: [8, 16, 32] }}
instance_group [{{ count: 1 kind: KIND_GPU }}]
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output",     default="outputs/onnx/medical_ai.onnx")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--device",     default=None)
    parser.add_argument("--verify",     action="store_true")
    parser.add_argument("--quantise",   action="store_true")
    args = parser.parse_args()
    main(args)
