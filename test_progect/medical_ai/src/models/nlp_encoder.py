"""
src/models/nlp_encoder.py
──────────────────────────
ClinicalBERT-based encoder for free-text clinical notes.

Pre-trained checkpoint: emilyalsentzer/Bio_ClinicalBERT
  - Trained on MIMIC-III discharge summaries and nursing notes
  - 110M parameters, BERT-base architecture

We add a lightweight projection head that maps the [CLS] token
representation → 256-d, matching the tabular encoder output.

Optionally freezes the first N transformer layers so only
the top layers + projection head are fine-tuned. This reduces
memory usage and prevents overfitting on small datasets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class ClinicalNLPEncoder(nn.Module):
    """
    ClinicalBERT encoder with a trainable projection head.

    Args:
        model_name   : HuggingFace model identifier
        output_dim   : projection dimension (must match tabular encoder)
        freeze_layers: freeze this many BERT encoder layers from the bottom.
                       Typical: 6 (freeze first half) for fine-tuning on
                       small medical datasets. Set 0 to train all layers.
        dropout      : dropout on the projection head
        pooling      : 'cls'  — use [CLS] token representation
                       'mean' — mean-pool all token representations
                       'max'  — max-pool all token representations
    """

    SUPPORTED_POOLINGS = ("cls", "mean", "max")

    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        output_dim: int = 256,
        freeze_layers: int = 6,
        dropout: float = 0.1,
        pooling: str = "cls",
    ):
        super().__init__()
        assert pooling in self.SUPPORTED_POOLINGS, \
            f"pooling must be one of {self.SUPPORTED_POOLINGS}"

        self.pooling = pooling

        # Load pre-trained model (weights will be downloaded on first run)
        config    = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        bert_hidden = config.hidden_size  # 768 for BERT-base

        # Freeze bottom layers to preserve general medical language knowledge
        self._freeze_layers(freeze_layers)

        # Projection: BERT hidden → output_dim
        self.projection = nn.Sequential(
            nn.Linear(bert_hidden, bert_hidden),
            nn.LayerNorm(bert_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bert_hidden, output_dim),
            nn.LayerNorm(output_dim),
        )

    def _freeze_layers(self, n: int) -> None:
        """Freeze embedding layer and first n transformer blocks."""
        # Always freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        # Freeze encoder layers
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < n:
                for param in layer.parameters():
                    param.requires_grad = False
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[NLPEncoder] {trainable:,} / {total:,} parameters trainable "
              f"(frozen layers 0–{n-1})")

    def _pool(
        self,
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool token representations → single sentence vector."""
        if self.pooling == "cls":
            return last_hidden[:, 0, :]

        # Expand mask: (B, L) → (B, L, H)
        mask_exp = attention_mask.unsqueeze(-1).float()

        if self.pooling == "mean":
            summed = (last_hidden * mask_exp).sum(dim=1)
            count  = mask_exp.sum(dim=1).clamp(min=1e-9)
            return summed / count

        # max pooling
        last_hidden_masked = last_hidden.masked_fill(
            attention_mask.unsqueeze(-1) == 0, float("-inf")
        )
        return last_hidden_masked.max(dim=1).values

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attentions: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids      : (B, L) int64
            attention_mask : (B, L) int64
            return_attentions: if True, include all attention matrices
                              (needed for attention rollout explainability)

        Returns dict with:
            embedding   : (B, output_dim)
            attentions  : tuple of (B, H, L, L) tensors (if requested)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attentions,
            output_hidden_states=False,
        )

        last_hidden = outputs.last_hidden_state   # (B, L, hidden)
        pooled      = self._pool(last_hidden, attention_mask)
        embedding   = self.projection(pooled)     # (B, output_dim)

        result = {"embedding": embedding}
        if return_attentions:
            result["attentions"] = outputs.attentions   # tuple of (B,H,L,L)
        return result

    def get_trainable_params(self) -> list[dict]:
        """
        Return parameter groups with different learning rates:
          - BERT layers: lower lr (fine-tuning)
          - Projection head: higher lr (task-specific)
        Useful for AdamW with layer-wise LR decay.
        """
        bert_params = [p for p in self.bert.parameters() if p.requires_grad]
        proj_params = list(self.projection.parameters())
        return [
            {"params": bert_params, "lr_multiplier": 0.1},
            {"params": proj_params, "lr_multiplier": 1.0},
        ]
