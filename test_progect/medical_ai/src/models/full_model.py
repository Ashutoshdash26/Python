"""
src/models/full_model.py
─────────────────────────
End-to-end MedicalAI model:

  PatientInput
      │
      ├── numerical tensor ──► TabularEncoderNet ──► (B, 256)  ─┐
      │                                                           ├─► CrossModalFusion ──► (B, 512)
      └── input_ids + mask ──► ClinicalNLPEncoder ──► (B, 256) ─┘
                                                                             │
                                                            DiseaseClassifierHead
                                                                             │
                                                      calibrated P(disease_i) for i in 1..N
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tabular_encoder import TabularEncoderNet, INPUT_DIM
from src.models.nlp_encoder import ClinicalNLPEncoder
from src.models.fusion_model import CrossModalFusion


# ─────────────────────────────────────────────────────────
#  Multi-label Disease Classifier Head
# ─────────────────────────────────────────────────────────

class DiseaseClassifierHead(nn.Module):
    """
    Multi-label classification head.
    Each output node produces an independent logit for one ICD-10 disease.
    Sigmoid (not softmax) is applied because diseases are not mutually exclusive.

    Architecture:
      (B, 512) → LayerNorm → Linear(256) → GELU → Dropout →
      Linear(num_diseases) → [calibrated sigmoid]

    The temperature parameter T is learned during calibration (post-training).
    A higher T flattens the sigmoid curve → more conservative probabilities.
    """

    def __init__(
        self,
        input_dim: int  = 512,
        hidden_dim: int = 256,
        num_diseases: int = 1000,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_diseases),
        )
        # Temperature for calibration — initialised to 1.0 (no-op)
        # Optimised separately after main training
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, fused: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Returns:
            logits : (B, num_diseases) — raw scores
            probs  : (B, num_diseases) — calibrated probabilities
        """
        logits = self.net(fused)
        probs  = torch.sigmoid(logits / self.temperature)
        return {"logits": logits, "probs": probs}

    def set_temperature(self, T: float) -> None:
        """Call after calibration to fix the temperature."""
        self.temperature.data = torch.tensor([T])
        self.temperature.requires_grad = False


# ─────────────────────────────────────────────────────────
#  Assembled Model
# ─────────────────────────────────────────────────────────

class MedicalAIModel(nn.Module):
    """
    Full disease prediction model.

    Usage:
        model = MedicalAIModel.from_config(cfg)
        output = model(numerical, input_ids, attention_mask)
        probs  = output["probs"]   # (B, num_diseases)

    Training:
        loss = criterion(output["logits"], labels) + output["entropy_loss"]
        loss.backward()

    Inference (after calibration):
        probs  = output["probs"]
        top_k  = probs.topk(10, dim=-1)
    """

    def __init__(
        self,
        # Tabular encoder
        tab_input_dim: int    = INPUT_DIM,
        tab_output_dim: int   = 256,
        tab_n_d: int          = 64,
        tab_n_a: int          = 64,
        tab_n_steps: int      = 5,
        tab_gamma: float      = 1.3,
        tab_dropout: float    = 0.1,
        # NLP encoder
        nlp_model_name: str   = "emilyalsentzer/Bio_ClinicalBERT",
        nlp_output_dim: int   = 256,
        nlp_freeze_layers: int = 6,
        nlp_dropout: float    = 0.1,
        nlp_pooling: str      = "cls",
        # Fusion
        fusion_hidden_dim: int = 512,
        fusion_num_heads: int  = 8,
        fusion_num_layers: int = 4,
        fusion_ff_dim: int     = 2048,
        fusion_dropout: float  = 0.1,
        fusion_output_dim: int = 512,
        # Classifier
        clf_hidden_dim: int   = 256,
        num_diseases: int     = 1000,
        clf_dropout: float    = 0.2,
    ):
        super().__init__()

        self.tabular_encoder = TabularEncoderNet(
            input_dim=tab_input_dim,
            output_dim=tab_output_dim,
            n_d=tab_n_d,
            n_a=tab_n_a,
            n_steps=tab_n_steps,
            gamma=tab_gamma,
            dropout=tab_dropout,
        )

        self.nlp_encoder = ClinicalNLPEncoder(
            model_name=nlp_model_name,
            output_dim=nlp_output_dim,
            freeze_layers=nlp_freeze_layers,
            dropout=nlp_dropout,
            pooling=nlp_pooling,
        )

        self.fusion = CrossModalFusion(
            tab_dim=tab_output_dim,
            nlp_dim=nlp_output_dim,
            hidden_dim=fusion_hidden_dim,
            num_heads=fusion_num_heads,
            num_layers=fusion_num_layers,
            ff_dim=fusion_ff_dim,
            dropout=fusion_dropout,
            output_dim=fusion_output_dim,
        )

        self.classifier = DiseaseClassifierHead(
            input_dim=fusion_output_dim,
            hidden_dim=clf_hidden_dim,
            num_diseases=num_diseases,
            dropout=clf_dropout,
        )

        self.num_diseases = num_diseases

    def forward(
        self,
        numerical: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attentions: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            numerical      : (B, tab_input_dim) — normalised tabular features
            input_ids      : (B, L) — tokenised clinical text
            attention_mask : (B, L)
            return_attentions: include BERT attention matrices in output

        Returns dict:
            logits        : (B, num_diseases)
            probs         : (B, num_diseases) calibrated
            tab_embedding : (B, 256)
            nlp_embedding : (B, 256)
            fused         : (B, 512)
            entropy_loss  : scalar (TabNet sparsity regularisation)
            attentions    : BERT attention tuple (if requested)
        """
        # ── Tabular branch ───────────────────────────────────────────────
        tab_emb, entropy_loss = self.tabular_encoder(numerical)   # (B,256), scalar

        # ── NLP branch ───────────────────────────────────────────────────
        nlp_out = self.nlp_encoder(
            input_ids, attention_mask,
            return_attentions=return_attentions,
        )
        nlp_emb = nlp_out["embedding"]  # (B, 256)

        # ── Fusion ───────────────────────────────────────────────────────
        fused = self.fusion(tab_emb, nlp_emb)   # (B, 512)

        # ── Classification ───────────────────────────────────────────────
        clf_out = self.classifier(fused)

        result = {
            "logits": clf_out["logits"],
            "probs": clf_out["probs"],
            "tab_embedding": tab_emb,
            "nlp_embedding": nlp_emb,
            "fused": fused,
            "entropy_loss": entropy_loss,
        }
        if return_attentions:
            result["attentions"] = nlp_out.get("attentions")

        return result

    # ── Model serialisation ───────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict) -> "MedicalAIModel":
        """Build model from a config dict (loaded from config.yaml)."""
        return cls(
            tab_input_dim    = cfg.get("tab_input_dim", INPUT_DIM),
            tab_output_dim   = cfg["tabular"]["tabnet"]["output_dim"],
            tab_n_d          = cfg["tabular"]["tabnet"]["n_d"],
            tab_n_a          = cfg["tabular"]["tabnet"]["n_a"],
            tab_n_steps      = cfg["tabular"]["tabnet"]["n_steps"],
            tab_gamma        = cfg["tabular"]["tabnet"]["gamma"],
            nlp_model_name   = cfg["nlp"]["model_name"],
            nlp_output_dim   = cfg["nlp"]["output_dim"],
            nlp_freeze_layers= cfg["nlp"]["freeze_layers"],
            nlp_dropout      = cfg["nlp"]["dropout"],
            fusion_hidden_dim= cfg["fusion"]["hidden_dim"],
            fusion_num_heads = cfg["fusion"]["num_heads"],
            fusion_num_layers= cfg["fusion"]["num_layers"],
            fusion_ff_dim    = cfg["fusion"]["ff_dim"],
            fusion_dropout   = cfg["fusion"]["dropout"],
            fusion_output_dim= cfg["fusion"]["hidden_dim"],
            num_diseases     = cfg["classifier"]["num_diseases"],
            clf_hidden_dim   = cfg["classifier"]["hidden_dim"],
            clf_dropout      = cfg["classifier"]["dropout"],
        )

    def save(self, path: str | Path) -> None:
        torch.save({"state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, cfg: dict) -> "MedicalAIModel":
        model = cls.from_config(cfg)
        ckpt  = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    def parameter_count(self) -> dict[str, int]:
        """Return parameter counts per sub-module."""
        def count(m): return sum(p.numel() for p in m.parameters())
        return {
            "tabular_encoder": count(self.tabular_encoder),
            "nlp_encoder":     count(self.nlp_encoder),
            "fusion":          count(self.fusion),
            "classifier":      count(self.classifier),
            "total":           count(self),
        }
