"""
src/training/losses.py
───────────────────────
Loss functions for multi-label disease prediction.

Multi-label class imbalance is severe in medical datasets:
  - Common conditions (hypertension, DM2) appear in 30-40% of records
  - Rare diseases (Wilson's disease, Fabry disease) appear in <0.01%

Two complementary strategies:
  1. WeightedBCELoss   — upweights rare classes via pos_weight
  2. AsymmetricFocalLoss — additionally hard-mines difficult examples
                            and asymmetrically handles false negatives
                            (missing a rare disease) vs false positives
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy for multi-label classification.
    pos_weight[i] = (N - pos_i) / pos_i  (computed on training set)

    Args:
        pos_weight : (num_diseases,) tensor of per-class positive weights
        reduction  : 'mean' | 'sum' | 'none'
    """

    def __init__(self, pos_weight: torch.Tensor, reduction: str = "mean"):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,  # (B, num_diseases)
        targets: torch.Tensor, # (B, num_diseases) float
    ) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )
        return loss


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss (ASL) — Ridnik et al., 2021.
    https://arxiv.org/abs/2009.14119

    Designed for multi-label imbalanced classification:
      - γ_neg (positive margin shifting): reduces loss contribution
        from easy negatives (overwhelming majority)
      - γ_pos: standard focal loss on positives
      - clip: probability clipping to prevent near-zero contributions
              from trivially easy negatives

    This is the state-of-the-art loss for medical multi-label
    classification and substantially outperforms vanilla BCE.

    Args:
        gamma_neg  : focusing parameter for negatives (typically 4)
        gamma_pos  : focusing parameter for positives (typically 0 or 1)
        clip       : probability margin shift for negatives
        reduction  : 'mean' | 'sum'
        eps        : numerical stability
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float      = 0.05,
        reduction: str   = "mean",
        eps: float       = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip
        self.reduction = reduction
        self.eps       = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits  : (B, C) — raw logits (before sigmoid)
            targets : (B, C) — float binary labels {0.0, 1.0}
        """
        probs = torch.sigmoid(logits)

        # Probability margin shift for negatives
        probs_neg = probs
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs + self.clip).clamp(max=1.0)

        # Xs_pos: log prob for positives
        # Xs_neg: log(1 - shifted_prob) for negatives
        xs_pos = probs
        xs_neg = 1.0 - probs_neg

        # Asymmetric binary cross entropy
        los_pos = targets      * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets)* torch.log(xs_neg.clamp(min=self.eps))

        loss = los_pos + los_neg

        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0     = xs_pos * targets
            pt1     = xs_neg * (1 - targets)
            pt      = pt0 + pt1
            one_side_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            weight  = (1 - pt) ** one_side_gamma
            loss    = loss * weight

        loss = -loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """
    Production loss: ASL + TabNet sparsity regularisation.

    total_loss = alpha * asl_loss + beta * entropy_loss

    Args:
        gamma_neg      : ASL negative focusing
        gamma_pos      : ASL positive focusing
        clip           : ASL probability clip
        entropy_weight : weight for TabNet sparsity loss (default 1e-3)
    """

    def __init__(
        self,
        gamma_neg: float      = 4.0,
        gamma_pos: float      = 1.0,
        clip: float           = 0.05,
        entropy_weight: float = 1e-3,
    ):
        super().__init__()
        self.asl            = AsymmetricFocalLoss(gamma_neg, gamma_pos, clip)
        self.entropy_weight = entropy_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        entropy_loss: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        asl_loss   = self.asl(logits, targets)
        total_loss = asl_loss + self.entropy_weight * entropy_loss
        return {
            "total": total_loss,
            "asl":   asl_loss,
            "entropy": entropy_loss,
        }


class LabelSmoothingBCE(nn.Module):
    """
    Label smoothing for multi-label: reduces overconfident predictions.
    Positive label 1.0 → 1 - eps
    Negative label 0.0 → eps
    """

    def __init__(self, eps: float = 0.05, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smooth_targets = targets * (1 - self.eps) + (1 - targets) * self.eps
        return F.binary_cross_entropy_with_logits(
            logits, smooth_targets, reduction=self.reduction
        )
