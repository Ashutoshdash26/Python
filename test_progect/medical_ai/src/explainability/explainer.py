"""
src/explainability/explainer.py
────────────────────────────────
Two-track explainability:

  1. SHAP (SHapley Additive exPlanations) for tabular features.
     Uses KernelExplainer with a background dataset of median values.
     Returns per-feature SHAP values for the top predicted diseases.

  2. Attention Rollout for clinical text.
     Rolls up attention weights across all BERT layers to compute
     which input tokens were most influential. Returns highlighted
     tokens to show the doctor which phrases triggered the prediction.

Both are run post-inference and are optional — the prediction is
returned even if explainability computation fails.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.data.preprocessing import (
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, HISTORY_FLAGS,
    POPULATION_MEDIANS,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + HISTORY_FLAGS


class MedicalExplainer:
    """
    Unified explainability engine for tabular (SHAP) and text (attention rollout).

    Args:
        model       : trained MedicalAIModel
        preprocessor: PatientPreprocessor (for tokenizer access)
        device      : inference device
        background_samples: number of background samples for KernelSHAP
    """

    def __init__(
        self,
        model,
        preprocessor,
        device: str = "cuda",
        background_samples: int = 100,
    ):
        self.model       = model.eval()
        self.preprocessor = preprocessor
        self.device      = device
        self.background  = self._build_background(background_samples)
        self._shap_explainer = None   # lazily initialised

    def _build_background(self, n: int) -> np.ndarray:
        """
        Background dataset for SHAP = population medians repeated n times.
        In production: use a sample from the actual training set.
        """
        medians = np.array([
            POPULATION_MEDIANS.get(f, 0.0) for f in ALL_FEATURES
        ], dtype=np.float32)
        # Append missingness indicators = 0 (no missing)
        zeros = np.zeros(len(ALL_FEATURES), dtype=np.float32)
        row   = np.concatenate([medians, zeros])
        return np.tile(row, (n, 1))

    def _tabular_predict_fn(self, x: np.ndarray) -> np.ndarray:
        """
        Wrapper for SHAP KernelExplainer: runs only the tabular branch.
        Returns (N, num_diseases) sigmoid probabilities.
        """
        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32).to(self.device)
            emb, _ = self.model.tabular_encoder(t)
            # Use zero NLP embedding for tabular-only SHAP
            nlp_zero = torch.zeros(t.size(0), 256, device=self.device)
            fused    = self.model.fusion(emb, nlp_zero)
            clf      = self.model.classifier(fused)
            return clf["probs"].cpu().numpy()

    def compute_shap(
        self,
        numerical_tensor: torch.Tensor,  # (1, input_dim)
        disease_indices: np.ndarray,
        n_shap_samples: int = 50,
    ) -> dict[str, float]:
        """
        Compute SHAP values for the top predicted disease classes.

        Returns:
            dict {feature_name: mean_abs_shap_value} sorted by importance.
            Positive values increase predicted risk; negative decrease it.
        """
        try:
            import shap
        except ImportError:
            log.warning("SHAP not installed; skipping feature importance")
            return {}

        if self._shap_explainer is None:
            log.debug("Initialising SHAP KernelExplainer")
            self._shap_explainer = shap.KernelExplainer(
                self._tabular_predict_fn,
                self.background,
                silent=True,
            )

        x = numerical_tensor.cpu().numpy()  # (1, input_dim)

        try:
            # Compute SHAP for selected disease columns only
            shap_values = self._shap_explainer.shap_values(
                x, nsamples=n_shap_samples, silent=True
            )
            # shap_values is list[num_diseases] of (1, input_dim) arrays
            # Average across the top-k disease indices
            relevant = np.array([shap_values[i][0] for i in disease_indices
                                  if i < len(shap_values)])
            if len(relevant) == 0:
                return {}
            mean_shap = relevant.mean(axis=0)  # (input_dim,)

            # Map back to feature names (first len(ALL_FEATURES) values,
            # the rest are missingness indicators)
            result: dict[str, float] = {}
            for i, name in enumerate(ALL_FEATURES):
                if i < len(mean_shap):
                    result[name] = float(mean_shap[i])

            # Sort by absolute value, return top 20
            result = dict(sorted(result.items(), key=lambda kv: abs(kv[1]), reverse=True)[:20])
            return result

        except Exception as e:
            log.warning("SHAP computation error", error=str(e))
            return {}

    def attention_rollout(
        self,
        input_ids: torch.Tensor,        # (1, L)
        attention_mask: torch.Tensor,   # (1, L)
        discard_ratio: float = 0.9,
    ) -> dict[str, float]:
        """
        Attention rollout for ClinicalBERT.
        Produces a per-token importance score by rolling attention weights
        from all layers, yielding a single attribution map.

        Reference: Abnar & Zuidema, 2020 (Quantifying Attention Flow)

        Args:
            input_ids      : tokenised input
            attention_mask : padding mask
            discard_ratio  : fraction of attention heads to discard (keep top 1-ratio)

        Returns:
            dict {token: importance_score} for the top-15 tokens.
        """
        self.model.eval()
        with torch.no_grad():
            output = self.model.nlp_encoder(
                input_ids.to(self.device),
                attention_mask.to(self.device),
                return_attentions=True,
            )

        attentions = output.get("attentions")
        if attentions is None:
            return {}

        # Stack: (num_layers, batch, heads, L, L)
        att_mat = torch.stack(attentions).squeeze(1)   # (layers, heads, L, L)

        # Average over heads
        att_mat = att_mat.mean(dim=1)   # (layers, L, L)

        # Add residual connection
        residual = torch.eye(att_mat.size(-1), device=att_mat.device).unsqueeze(0)
        att_mat  = att_mat + residual
        att_mat  = att_mat / att_mat.sum(dim=-1, keepdim=True)

        # Rollout: multiply layer by layer
        joint_att = att_mat[0]
        for i in range(1, att_mat.size(0)):
            joint_att = att_mat[i] @ joint_att

        # CLS token row = attention from CLS to all tokens
        cls_att = joint_att[0].cpu().numpy()   # (L,)
        cls_att = cls_att / (cls_att.max() + 1e-9)

        # Decode tokens
        tokens = self.preprocessor.tokenizer.convert_ids_to_tokens(
            input_ids.squeeze(0).cpu().tolist()
        )

        result: dict[str, float] = {}
        for tok, score in zip(tokens, cls_att):
            if tok in ("[CLS]", "[SEP]", "[PAD]"):
                continue
            clean_tok = tok.replace("##", "")
            result[clean_tok] = max(result.get(clean_tok, 0.0), float(score))

        # Return top 15 tokens by importance
        return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True)[:15])

    def explain(
        self,
        features: dict,
        disease_indices: np.ndarray,
    ) -> dict:
        """
        Run full explanation (SHAP + attention rollout).
        Returns combined dict suitable for API response.
        """
        shap_vals = self.compute_shap(
            features["numerical"].unsqueeze(0),
            disease_indices,
        )
        attn_vals = self.attention_rollout(
            features["input_ids"].unsqueeze(0),
            features["attention_mask"].unsqueeze(0),
        )
        return {
            "feature_shap": shap_vals,
            "text_attention": attn_vals,
        }
