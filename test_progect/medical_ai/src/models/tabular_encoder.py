"""
src/models/tabular_encoder.py
──────────────────────────────
Tabular branch of the dual-encoder architecture.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.preprocessing import (
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, HISTORY_FLAGS
)

# ─────────────────────────────────────────────────────────
# Input Dimension
# ─────────────────────────────────────────────────────────

_BASE_FEATURES = (
    len(NUMERICAL_FEATURES) +
    len(CATEGORICAL_FEATURES) +
    len(HISTORY_FLAGS)
)

INPUT_DIM = _BASE_FEATURES * 2  # features + missing mask


# ─────────────────────────────────────────────────────────
# GLU Block
# ─────────────────────────────────────────────────────────

class GLUBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        shared_fc: Optional[nn.Linear] = None,
    ):
        super().__init__()

        self.shared_fc = shared_fc

        self.step_fc = nn.Linear(
            in_dim if shared_fc is None else out_dim,
            out_dim * 2,
            bias=False
        )

        self.bn = nn.BatchNorm1d(out_dim * 2, momentum=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shared_fc is not None:
            x = self.shared_fc(x)

        x = self.bn(self.step_fc(x))
        x1, x2 = x.chunk(2, dim=-1)

        return x1 * torch.sigmoid(x2)


# ─────────────────────────────────────────────────────────
# Attentive Transformer
# ─────────────────────────────────────────────────────────

class AttentiveTransformer(nn.Module):
    def __init__(self, in_dim: int, num_features: int):
        super().__init__()

        self.fc = nn.Linear(in_dim, num_features, bias=False)
        self.bn = nn.BatchNorm1d(num_features, momentum=0.02)

    def forward(self, processed, prior_scales):
        h = self.bn(self.fc(processed))
        h = h * prior_scales

        return F.softmax(h, dim=-1)


# ─────────────────────────────────────────────────────────
# TabNet Encoder
# ─────────────────────────────────────────────────────────

class TabularEncoderNet(nn.Module):

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        output_dim: int = 256,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 5,
        gamma: float = 1.3,
        n_shared: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_d = n_d
        self.n_a = n_a

        # ── Initial BN ─────────────────────────────
        self.initial_bn = nn.BatchNorm1d(input_dim, momentum=0.02)

        # ── Shared layers ──────────────────────────
        shared_layers = []
        in_d = input_dim

        for _ in range(n_shared):
            shared_layers.append(
                nn.Linear(in_d, n_d + n_a, bias=False)
            )
            in_d = n_d + n_a

        self.shared_fcs = nn.ModuleList(shared_layers)

        # ── Step-specific layers ───────────────────
        self.step_feature_transformers = nn.ModuleList([
            GLUBlock(n_d + n_a, n_d + n_a)
            for _ in range(n_steps)
        ])

        self.step_attention_transformers = nn.ModuleList([
            AttentiveTransformer(n_a, input_dim)
            for _ in range(n_steps)
        ])

        # ── Initial splitter ───────────────────────
        self.initial_splitter = nn.Linear(
            input_dim,
            n_d + n_a,
            bias=False
        )

        self.initial_bn2 = nn.BatchNorm1d(n_d + n_a, momentum=0.02)

        # ── Output projection ──────────────────────
        self.output_proj = nn.Sequential(
            nn.Linear(n_d, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        B = x.size(0)

        # ── Initial processing ─────────────────────
        x = self.initial_bn(x)

        h = F.gelu(self.initial_bn2(self.initial_splitter(x)))

        h1 = h[:, :self.n_d]
        h_attn = h[:, self.n_d:]

        prior_scales = torch.ones(B, self.input_dim, device=x.device)
        aggregated = torch.zeros(B, self.n_d, device=x.device)
        entropy_loss = torch.zeros(1, device=x.device)

        # ── Steps ──────────────────────────────────
        for step in range(self.n_steps):

            # Attention
            mask = self.step_attention_transformers[step](
                h_attn, prior_scales
            )

            prior_scales = prior_scales * (self.gamma - mask)

            entropy_loss += (
                -mask * torch.log(mask + 1e-15)
            ).sum(dim=-1).mean()

            # Apply mask
            masked_x = mask * x

            # Shared layers
            h_shared = masked_x
            for fc in self.shared_fcs:
                h_shared = F.gelu(fc(h_shared))

            # Step GLU
            h_step = self.step_feature_transformers[step](h_shared)

            h1_step = h_step[:, :self.n_d]
            h_attn = h_step[:, self.n_d:]

            aggregated += F.relu(h1_step)

        embedding = self.output_proj(aggregated)

        return embedding, entropy_loss / self.n_steps


# ─────────────────────────────────────────────────────────
# Fallback MLP
# ─────────────────────────────────────────────────────────

class TabularEncoderMLP(nn.Module):

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        output_dim: int = 256,
        hidden_dims: tuple = (512, 512, 256),
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        prev = input_dim

        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h

        layers += [
            nn.Linear(prev, output_dim),
            nn.LayerNorm(output_dim),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x), torch.zeros(1, device=x.device)