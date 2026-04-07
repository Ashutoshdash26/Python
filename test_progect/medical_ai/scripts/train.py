"""
scripts/train.py
─────────────────
Main training entry point.

Usage:
  python scripts/train.py --config configs/config.yaml
  python scripts/train.py --config configs/config.yaml --device cuda --wandb

Steps:
  1. Load config
  2. Build ICD vocabulary from training data
  3. Load tokenizer + build dataloaders
  4. Build MedicalAIModel
  5. Train with MedicalAITrainer
  6. Calibrate probabilities on validation set
  7. Save calibrated model + export ONNX
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build_icd_vocab(train_parquet_path: str, top_k: int = 1000) -> dict[str, int]:
    """
    Build ICD-10 vocabulary from training data.
    Selects the top_k most frequent diagnoses.
    Returns dict {icd10_code: index}.

    In production: run this once and save to data/icd10_vocab.json.
    """
    try:
        import pandas as pd, json as _json, collections

        df = pd.read_parquet(train_parquet_path)
        counter: dict[str, int] = collections.Counter()
        for labels in df["icd10_labels"]:
            if isinstance(labels, str):
                labels = _json.loads(labels)
            for code in labels:
                counter[code] += 1
        top_codes = [code for code, _ in counter.most_common(top_k)]
        return {code: i for i, code in enumerate(sorted(top_codes))}
    except Exception as e:
        print(f"[warn] Could not build vocab from data: {e}")
        print("[warn] Using built-in ICD registry for demo")
        from src.utils.icd_codes import _BUILTIN_CODES
        return {c["icd10"]: c["index"] for c in _BUILTIN_CODES}


def main(args: argparse.Namespace) -> None:
    # ── Load config ──────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])
    output_dir = Path(cfg["paths"]["model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # ── ICD Vocabulary ───────────────────────────────────────────────────
    vocab_path = Path("data/icd10_vocab.json")
    if vocab_path.exists():
        with open(vocab_path) as f:
            icd_vocab = json.load(f)
        print(f"[train] Loaded ICD vocab: {len(icd_vocab)} codes")
    else:
        print("[train] Building ICD vocab from training data...")
        train_path = str(Path(cfg["paths"]["processed_dir"]) / "train.parquet")
        icd_vocab  = build_icd_vocab(train_path, top_k=cfg["classifier"]["num_diseases"])
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_path, "w") as f:
            json.dump(icd_vocab, f, indent=2)
        print(f"[train] ICD vocab saved: {len(icd_vocab)} codes → {vocab_path}")

    # Override num_diseases from vocab size
    cfg["classifier"]["num_diseases"] = len(icd_vocab)

    # ── Tokenizer + Dataloaders ─────────────────────────────────────────
    from transformers import AutoTokenizer
    from src.data.dataset import get_dataloaders

    tokenizer = AutoTokenizer.from_pretrained(cfg["nlp"]["model_name"])

    processed = Path(cfg["paths"]["processed_dir"])
    train_loader, val_loader, test_loader = get_dataloaders(
        train_path=str(processed / "train.parquet"),
        val_path  =str(processed / "val.parquet"),
        test_path =str(processed / "test.parquet"),
        icd_vocab =icd_vocab,
        tokenizer =tokenizer,
        batch_size =cfg["training"]["batch_size"],
        max_seq_len=cfg["data"]["max_seq_len"],
        num_workers=cfg["training"]["num_workers"],
    )
    print(f"[train] Train batches: {len(train_loader)}, Val: {len(val_loader)}")

    # ── Build Model ──────────────────────────────────────────────────────
    from src.models.full_model import MedicalAIModel

    model = MedicalAIModel.from_config(cfg)
    params = model.parameter_count()
    print(f"[train] Model parameters: {params}")

    # ── Train ────────────────────────────────────────────────────────────
    from src.training.trainer import MedicalAITrainer

    trainer = MedicalAITrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        output_dir=output_dir,
        icd_vocab=icd_vocab,
        device=device,
        use_wandb=args.wandb,
    )
    best_model = trainer.train()
    print("[train] Training complete. Running calibration...")

    # ── Calibrate ────────────────────────────────────────────────────────
    from src.training.calibration import TemperatureCalibrator

    calibrator = TemperatureCalibrator(best_model, device=device)
    calibrated_model = calibrator.calibrate_model(val_loader)

    # Save calibrated model
    cal_path = output_dir / "calibrated_model.pt"
    calibrated_model.save(cal_path)
    print(f"[train] Calibrated model saved to {cal_path}")

    # ── Export ONNX (optional) ────────────────────────────────────────────
    if args.export_onnx:
        from scripts.export_onnx import export_to_onnx
        onnx_path = Path(cfg["paths"]["onnx_dir"]) / "medical_ai.onnx"
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_onnx(calibrated_model, cfg, str(onnx_path), device)
        print(f"[train] ONNX model exported to {onnx_path}")

    print("[train] All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MedicalAI model")
    parser.add_argument("--config",       default="configs/config.yaml")
    parser.add_argument("--device",       default=None,  help="cuda or cpu")
    parser.add_argument("--wandb",        action="store_true", help="Enable W&B logging")
    parser.add_argument("--export-onnx",  action="store_true", help="Export ONNX after training")
    args = parser.parse_args()
    main(args)
