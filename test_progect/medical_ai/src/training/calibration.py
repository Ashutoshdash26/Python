"""
src/training/calibration.py
─────────────────────────────
Post-hoc probability calibration via temperature scaling.

After training, the model's raw sigmoid outputs are often overconfident
(probabilities cluster near 0 and 1). Temperature scaling finds a single
scalar T that minimises the Expected Calibration Error (ECE) on the
validation set.

New probability = sigmoid(logit / T)

T > 1 → softer (more conservative) probabilities
T < 1 → sharper probabilities (rarely needed)

Reference: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS
from torch.utils.data import DataLoader

from src.models.full_model import MedicalAIModel
from src.utils.logger import get_logger

log = get_logger(__name__)


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute ECE: average calibration error across confidence bins.
    ECE = Σ_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    Lower is better; 0.0 = perfectly calibrated.

    Args:
        probs  : (N, C) predicted probabilities
        labels : (N, C) binary labels
        n_bins : number of confidence bins
    """
    # Flatten multi-label
    probs_flat  = probs.flatten()
    labels_flat = labels.flatten()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n   = len(probs_flat)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask   = (probs_flat >= lo) & (probs_flat < hi)
        if mask.sum() == 0:
            continue
        bin_conf = probs_flat[mask].mean()
        bin_acc  = labels_flat[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


class TemperatureCalibrator:
    """
    Finds the optimal temperature T for a trained MedicalAIModel.

    Usage:
        cal = TemperatureCalibrator(model, device)
        T   = cal.fit(val_loader)
        model.classifier.set_temperature(T)
    """

    def __init__(
        self,
        model: MedicalAIModel,
        device: str = "cuda",
        max_iter: int = 50,
        lr: float = 0.01,
    ):
        self.model    = model.eval()
        self.device   = device
        self.max_iter = max_iter
        self.lr       = lr

    @torch.no_grad()
    def _collect_logits(self, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference on val set, collect logits and labels."""
        all_logits = []
        all_labels = []

        for batch in loader:
            numerical      = batch["numerical"].to(self.device)
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"]

            output = self.model(numerical, input_ids, attention_mask)
            all_logits.append(output["logits"].cpu())
            all_labels.append(labels.cpu())

        return torch.cat(all_logits), torch.cat(all_labels)

    def fit(self, val_loader: DataLoader) -> float:
        """
        Optimise temperature T using NLL loss on calibration set.

        Returns:
            T : optimal temperature (float)
        """
        log.info("Collecting logits for calibration...")
        logits, labels = self._collect_logits(val_loader)

        temperature = nn.Parameter(torch.ones(1) * 1.5)
        optimizer   = LBFGS([temperature], lr=self.lr, max_iter=self.max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_probs = torch.sigmoid(logits / temperature)
            loss = nn.functional.binary_cross_entropy(
                scaled_probs.clamp(1e-7, 1 - 1e-7), labels
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        T = float(temperature.item())
        T = max(0.1, min(T, 10.0))  # safety clamp

        # Evaluate ECE before/after
        probs_uncal = torch.sigmoid(logits).numpy()
        probs_cal   = torch.sigmoid(logits / T).numpy()
        labels_np   = labels.numpy()

        ece_before = expected_calibration_error(probs_uncal, labels_np)
        ece_after  = expected_calibration_error(probs_cal,   labels_np)

        log.info(
            "Calibration complete",
            temperature=f"{T:.4f}",
            ece_before=f"{ece_before:.4f}",
            ece_after=f"{ece_after:.4f}",
            improvement=f"{(ece_before - ece_after) / ece_before * 100:.1f}%",
        )
        return T

    def calibrate_model(self, val_loader: DataLoader) -> MedicalAIModel:
        """Fit temperature and apply it to the model in-place."""
        T = self.fit(val_loader)
        self.model.classifier.set_temperature(T)
        log.info(f"Model calibrated. Temperature set to {T:.4f}")
        return self.model
