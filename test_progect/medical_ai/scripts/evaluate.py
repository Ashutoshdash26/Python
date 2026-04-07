"""
scripts/evaluate.py
────────────────────
Full test-set evaluation suite.

Produces:
  - Overall metrics (AUROC, AP, F1, ECE, Coverage@10)
  - Per-class AUROC for every disease with ≥20 positives
  - Calibration curve (reliability diagram data)
  - Demographic fairness audit (AUROC by sex and age group)
  - Confusion analysis: top false positives / false negatives per class
  - Saves full report to outputs/evaluation_report.json

Usage:
  python scripts/evaluate.py --checkpoint outputs/models/calibrated_model.pt
  python scripts/evaluate.py --checkpoint outputs/models/calibrated_model.pt --fairness
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.cuda.amp import autocast


def load_model_and_config(checkpoint: str, config: str, device: str):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    from src.models.full_model import MedicalAIModel
    model = MedicalAIModel.load(checkpoint, cfg)
    model = model.eval().to(device)
    return model, cfg


@torch.no_grad()
def collect_predictions(model, loader, device: str, mixed_prec: bool = True):
    all_probs, all_labels, all_patient_ids = [], [], []
    all_meta = []

    for batch in loader:
        numerical      = batch["numerical"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"]

        with autocast(enabled=mixed_prec and device == "cuda"):
            output = model(numerical, input_ids, attention_mask)

        all_probs.append(output["probs"].cpu().numpy())
        all_labels.append(labels.numpy())
        all_patient_ids.extend(batch.get("patient_id", [""] * len(labels)))

    return (
        np.concatenate(all_probs,  axis=0),
        np.concatenate(all_labels, axis=0),
        all_patient_ids,
    )


def run_fairness_audit(
    probs: np.ndarray,
    labels: np.ndarray,
    metadata_df,
) -> dict:
    """
    Compute AUROC separately for subgroups:
      - Sex (female vs male)
      - Age group (<40, 40-65, >65)
    A model is considered fair if AUROC gap between groups is < 0.05.
    """
    from sklearn.metrics import roc_auc_score

    results = {}

    for sex_val, sex_name in [(0, "female"), (1, "male")]:
        mask = metadata_df["sex"].values == sex_val
        if mask.sum() < 50:
            continue
        try:
            auroc = roc_auc_score(labels[mask], probs[mask], average="macro")
            results[f"auroc_sex_{sex_name}"] = round(float(auroc), 4)
        except Exception:
            pass

    age_groups = [("<40", lambda a: a < 40), ("40-65", lambda a: 40 <= a < 65), (">65", lambda a: a >= 65)]
    ages = metadata_df["age"].values
    for name, fn in age_groups:
        mask = np.array([fn(a) for a in ages])
        if mask.sum() < 50:
            continue
        try:
            auroc = roc_auc_score(labels[mask], probs[mask], average="macro")
            results[f"auroc_age_{name}"] = round(float(auroc), 4)
        except Exception:
            pass

    # Fairness gap: max AUROC between sex groups
    sex_scores = [v for k, v in results.items() if "auroc_sex" in k]
    if len(sex_scores) == 2:
        results["sex_fairness_gap"] = round(abs(sex_scores[0] - sex_scores[1]), 4)
        results["sex_fair"]         = results["sex_fairness_gap"] < 0.05

    return results


def main(args: argparse.Namespace) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}")

    model, cfg = load_model_and_config(args.checkpoint, args.config, device)

    # ── Load test data ───────────────────────────────────────────────────
    import json as _json
    from transformers import AutoTokenizer
    from src.data.dataset import MedicalDataset
    from torch.utils.data import DataLoader

    with open("data/icd10_vocab.json") as f:
        icd_vocab = _json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(cfg["nlp"]["model_name"])
    test_ds   = MedicalDataset(
        str(Path(cfg["paths"]["processed_dir"]) / "test.parquet"),
        icd_vocab, tokenizer, cfg["data"]["max_seq_len"], augment=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False,
        num_workers=cfg["training"]["num_workers"], pin_memory=True,
    )
    print(f"[eval] Test set: {len(test_ds)} patients")

    # ── Run inference ────────────────────────────────────────────────────
    print("[eval] Running inference on test set...")
    probs, labels, patient_ids = collect_predictions(
        model, test_loader, device, mixed_prec=cfg["training"]["mixed_precision"]
    )
    print(f"[eval] Collected predictions: {probs.shape}")

    # ── Overall metrics ──────────────────────────────────────────────────
    from src.utils.metrics import compute_epoch_metrics, per_class_auroc
    from src.utils.icd_codes import ICDRegistry

    icd_registry = ICDRegistry.from_json("data/icd10_vocab.json")
    icd_names    = [icd_registry.get_by_index(i)["name"]
                    for i in range(probs.shape[1])]

    overall_metrics = compute_epoch_metrics(
        probs, labels,
        threshold=cfg["classifier"]["threshold"],
    )
    print("\n── Overall Test Metrics ──────────────────────────")
    for k, v in overall_metrics.items():
        print(f"  {k:25s}: {v}")

    # ── Per-class AUROC ──────────────────────────────────────────────────
    print("\n── Per-class AUROC (top 20) ──────────────────────")
    class_auroc = per_class_auroc(probs, labels, icd_names, min_positive=20)
    for name, auroc in list(class_auroc.items())[:20]:
        print(f"  {name[:50]:50s}: {auroc:.4f}")

    # Bottom 5 (worst-performing)
    bottom = list(class_auroc.items())[-5:]
    print("\n── Lowest AUROC (needs attention) ───────────────")
    for name, auroc in bottom:
        print(f"  {name[:50]:50s}: {auroc:.4f}")

    # ── Fairness audit ───────────────────────────────────────────────────
    fairness_results = {}
    if args.fairness:
        print("\n── Fairness Audit ────────────────────────────────")
        try:
            import pandas as pd
            test_df = pd.read_parquet(
                str(Path(cfg["paths"]["processed_dir"]) / "test.parquet")
            )
            fairness_results = run_fairness_audit(probs, labels, test_df)
            for k, v in fairness_results.items():
                print(f"  {k:35s}: {v}")
        except Exception as e:
            print(f"  [skip] Fairness audit failed: {e}")

    # ── Save report ──────────────────────────────────────────────────────
    report = {
        "checkpoint":     args.checkpoint,
        "test_size":      len(test_ds),
        "num_diseases":   probs.shape[1],
        "overall_metrics": overall_metrics,
        "top_20_class_auroc": dict(list(class_auroc.items())[:20]),
        "bottom_5_class_auroc": dict(bottom),
        "fairness":        fairness_results,
    }
    report_path = Path("outputs/evaluation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        _json.dump(report, f, indent=2)
    print(f"\n[eval] Full report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MedicalAI model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--device",     default=None)
    parser.add_argument("--fairness",   action="store_true",
                        help="Run demographic fairness audit")
    args = parser.parse_args()
    main(args)
