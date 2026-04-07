"""
src/utils/metrics.py
─────────────────────
Evaluation metrics for multi-label disease prediction.

Reported metrics:
  - AUROC (macro, micro) — primary metric
  - Average Precision (AP, macro) — ranking metric
  - F1 (macro, micro, per-class) — threshold-dependent
  - Precision, Recall
  - Expected Calibration Error (ECE)
  - Coverage (% of true labels appearing in top-10 predictions)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_epoch_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.35,
    min_positive_samples: int = 10,
) -> dict[str, float]:
    """
    Compute all validation metrics for one epoch.

    Args:
        probs    : (N, num_diseases) float — model probabilities
        labels   : (N, num_diseases) float — binary ground truth
        threshold: decision boundary for binary predictions
        min_positive_samples: skip per-class AUROC for rarer classes

    Returns:
        dict of metric name → float value
    """
    preds = (probs >= threshold).astype(float)

    # ── AUROC ─────────────────────────────────────────────────────────────
    # Only compute per class where we have sufficient positive examples
    class_sums = labels.sum(axis=0)
    valid_cols  = class_sums >= min_positive_samples

    auroc_macro = auroc_micro = 0.0
    if valid_cols.sum() >= 2:
        try:
            auroc_macro = roc_auc_score(
                labels[:, valid_cols], probs[:, valid_cols],
                average="macro",
            )
            auroc_micro = roc_auc_score(
                labels[:, valid_cols], probs[:, valid_cols],
                average="micro",
            )
        except Exception:
            pass

    # ── Average Precision ─────────────────────────────────────────────────
    ap_macro = ap_micro = 0.0
    if valid_cols.sum() >= 2:
        try:
            ap_macro = average_precision_score(
                labels[:, valid_cols], probs[:, valid_cols], average="macro"
            )
            ap_micro = average_precision_score(
                labels[:, valid_cols], probs[:, valid_cols], average="micro"
            )
        except Exception:
            pass

    # ── F1 / Precision / Recall ────────────────────────────────────────────
    f1_macro = f1_score(labels, preds, average="macro",  zero_division=0)
    f1_micro = f1_score(labels, preds, average="micro",  zero_division=0)
    prec     = precision_score(labels, preds, average="macro", zero_division=0)
    rec      = recall_score(labels, preds, average="macro", zero_division=0)

    # ── Label coverage in top-10 ───────────────────────────────────────────
    top10_coverage = _top_k_coverage(probs, labels, k=10)

    # ── Calibration (ECE) ─────────────────────────────────────────────────
    ece = _expected_calibration_error(probs.flatten(), labels.flatten())

    # ── Prevalence stats ──────────────────────────────────────────────────
    mean_labels_per_patient = float(labels.sum(axis=1).mean())

    return {
        "auroc_macro":   round(auroc_macro,  4),
        "auroc_micro":   round(auroc_micro,  4),
        "ap_macro":      round(ap_macro,     4),
        "ap_micro":      round(ap_micro,     4),
        "f1_macro":      round(f1_macro,     4),
        "f1_micro":      round(f1_micro,     4),
        "precision":     round(prec,         4),
        "recall":        round(rec,          4),
        "top10_coverage":round(top10_coverage, 4),
        "ece":           round(ece,          4),
        "mean_labels":   round(mean_labels_per_patient, 2),
    }


def _top_k_coverage(
    probs: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
) -> float:
    """
    Coverage@K: fraction of true labels that appear in the top-K predictions.
    Measures whether the correct diseases are being surfaced.
    """
    n_patients   = probs.shape[0]
    covered      = 0
    total_labels = 0

    for i in range(n_patients):
        true_idxs = np.where(labels[i] > 0)[0]
        if len(true_idxs) == 0:
            continue
        top_k_idxs = np.argsort(probs[i])[::-1][:k]
        covered      += len(set(true_idxs) & set(top_k_idxs))
        total_labels += len(true_idxs)

    return covered / total_labels if total_labels > 0 else 0.0


def _expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute ECE (lower is better; 0 = perfect calibration)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(probs)

    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        conf = probs[mask].mean()
        acc  = labels[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)

    return float(ece)


def per_class_auroc(
    probs: np.ndarray,
    labels: np.ndarray,
    icd_names: list[str],
    min_positive: int = 20,
) -> dict[str, float]:
    """
    Compute AUROC per ICD-10 class for detailed evaluation report.
    Used in evaluate.py for the full test-set evaluation.
    """
    results = {}
    for i, name in enumerate(icd_names):
        y_true = labels[:, i]
        y_prob = probs[:, i]
        if y_true.sum() < min_positive:
            continue
        try:
            results[name] = round(float(roc_auc_score(y_true, y_prob)), 4)
        except Exception:
            pass
    return dict(sorted(results.items(), key=lambda kv: kv[1], reverse=True))
