"""
src/models/fusion_model.py
───────────────────────────
Cross-attention fusion transformer that combines the two encoder embeddings.

Design:
  - Tabular embedding  (B, 256) and NLP embedding (B, 256) are treated
    as two "tokens" in a tiny 2-token sequence.
  - A 4-layer transformer encoder with 8 heads learns how much to weight
    each modality given the specific patient presentation.
  - The fused representation is then pooled and projected to 512-d.

This design is superior to simple concatenation because:
  1. The attention mechanism learns to trust the NLP branch more when
     clinical notes are rich, and the tabular branch more when labs
     are abnormal — dynamically per patient.
  2. It preserves both modalities' information rather than compressing
     immediately after concatenation.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalFusion(nn.Module):
    """
    Cross-attention fusion of tabular and NLP encoder outputs.

    Architecture:
      [tab_emb, nlp_emb] → positional type embedding →
      TransformerEncoder (4 layers, 8 heads) →
      concatenate CLS tokens → LayerNorm → Linear(512)

    Args:
        tab_dim    : tabular encoder output dimension (256)
        nlp_dim    : NLP encoder output dimension (256)
        hidden_dim : internal transformer dimension (512)
        num_heads  : attention heads (must divide hidden_dim)
        num_layers : transformer encoder layers
        ff_dim     : feed-forward inner dimension
        dropout    : attention + FF dropout
        output_dim : final fused representation dimension
    """

    def __init__(
        self,
        tab_dim: int    = 256,
        nlp_dim: int    = 256,
        hidden_dim: int = 512,
        num_heads: int  = 8,
        num_layers: int = 4,
        ff_dim: int     = 2048,
        dropout: float  = 0.1,
        output_dim: int = 512,
    ):
        super().__init__()

        # Project both encoders to the same hidden_dim
        self.tab_proj = nn.Sequential(
            nn.Linear(tab_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.nlp_proj = nn.Sequential(
            nn.Linear(nlp_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Learned modality-type embeddings
        # Position 0 = tabular, Position 1 = NLP
        self.modality_embedding = nn.Embedding(2, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,    # Pre-norm (more stable training)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        # Final projection: 2 * hidden_dim → output_dim
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform initialisation for projection layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        tab_emb: torch.Tensor,    # (B, tab_dim)
        nlp_emb: torch.Tensor,    # (B, nlp_dim)
    ) -> torch.Tensor:
        """
        Returns:
            fused: (B, output_dim) — patient-level fused representation
        """
        B = tab_emb.size(0)
        device = tab_emb.device

        # Project to hidden_dim
        tab_h = self.tab_proj(tab_emb)   # (B, hidden_dim)
        nlp_h = self.nlp_proj(nlp_emb)   # (B, hidden_dim)

        # Add modality type embeddings
        mod_ids = torch.arange(2, device=device)   # [0, 1]
        mod_embs = self.modality_embedding(mod_ids) # (2, hidden_dim)
        tab_h = tab_h + mod_embs[0].unsqueeze(0)
        nlp_h = nlp_h + mod_embs[1].unsqueeze(0)

        # Stack into sequence: (B, 2, hidden_dim)
        seq = torch.stack([tab_h, nlp_h], dim=1)

        # Self-attention over 2-token sequence
        fused_seq = self.transformer(seq)   # (B, 2, hidden_dim)

        # Concatenate both positions and project
        fused_cat = fused_seq.reshape(B, -1)   # (B, 2*hidden_dim)
        return self.output_proj(fused_cat)      # (B, output_dim)

    def get_modality_weights(
        self,
        tab_emb: torch.Tensor,
        nlp_emb: torch.Tensor,
    ) -> dict[str, float]:
        """
        Compute approximate modality importance from the last attention layer.
        Returns dict: {'tabular': 0.62, 'nlp': 0.38} — interpretable output.
        """
        self.eval()
        with torch.no_grad():
            B = tab_emb.size(0)
            tab_h = self.tab_proj(tab_emb) + self.modality_embedding.weight[0]
            nlp_h = self.nlp_proj(nlp_emb) + self.modality_embedding.weight[1]
            seq = torch.stack([tab_h, nlp_h], dim=1)

            # Extract attention from last encoder layer
            # We manually run just the last layer to get attention weights
            last_layer = self.transformer.layers[-1]
            attn_output, attn_weights = last_layer.self_attn(
                seq, seq, seq,
                need_weights=True,
                average_attn_weights=True,
            )
            # attn_weights: (B, 2, 2) — rows are queries, cols are keys
            # Average over batch and queries
            avg_weights = attn_weights.mean(dim=(0, 1))  # (2,)
            tab_w = float(avg_weights[0].item())
            nlp_w = float(avg_weights[1].item())
            total = tab_w + nlp_w + 1e-9
        return {"tabular": round(tab_w / total, 3), "nlp": round(nlp_w / total, 3)}
