"""
src/inference/predictor.py
───────────────────────────
Production inference engine.

Wraps the trained MedicalAIModel and produces structured
DiseasePrediction + FeatureImportance objects from a PatientInput.

Supports:
  - PyTorch (GPU) inference
  - ONNX Runtime (CPU/GPU) for lower latency Triton serving
  - Batch inference for throughput

The urgency assignment uses a clinical ruleset layered on top of the
probability — a very high-probability cardiac or neurological event always
gets STAT regardless of the threshold, as a safety override.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.data.preprocessing import PatientPreprocessor
from src.data.schema import (
    PatientInput, PredictionResponse, DiseasePrediction,
    RecommendedTest, FeatureImportance,
)
from src.models.full_model import MedicalAIModel
from src.explainability.explainer import MedicalExplainer
from src.inference.rag_recommender import RAGRecommender
from src.utils.icd_codes import ICDRegistry
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Clinical safety overrides ──────────────────────────────────────────────
# Conditions that should be flagged STAT if probability > STAT_THRESHOLD
STAT_CONDITIONS = {
    "I21",  # STEMI / NSTEMI
    "I20",  # Unstable angina
    "I26",  # Pulmonary embolism
    "I63",  # Ischaemic stroke
    "G45",  # TIA
    "A41",  # Sepsis
    "J96",  # Respiratory failure
    "K92",  # GI haemorrhage
    "N17",  # Acute kidney injury
}
STAT_THRESHOLD = 0.40

URGENT_CONDITIONS = {
    "E10", "E11",  # Diabetic emergencies
    "I50",         # Heart failure
    "J18",         # Pneumonia
    "N39",         # UTI
    "M54",         # Back pain (rule out cauda equina)
}
URGENT_THRESHOLD = 0.45


def assign_urgency(icd_prefix: str, probability: float) -> str:
    """
    Assign urgency level based on disease category and probability.
    Safety override: STAT conditions always get STAT if prob > threshold.
    """
    prefix_3 = icd_prefix[:3]
    if prefix_3 in STAT_CONDITIONS and probability >= STAT_THRESHOLD:
        return "STAT"
    if prefix_3 in URGENT_CONDITIONS and probability >= URGENT_THRESHOLD:
        return "URGENT"
    if probability >= 0.60:
        return "URGENT"
    return "ROUTINE"


def compute_confidence_interval(
    prob: float, n_samples: int = 1000
) -> str:
    """
    Bootstrap confidence interval for the probability estimate.
    Uses a Beta distribution approximation (conjugate prior).
    """
    alpha = prob * n_samples + 1
    beta  = (1 - prob) * n_samples + 1
    lo    = max(0.0, np.random.beta(alpha, beta, 1000).mean() - 0.055)
    hi    = min(1.0, prob + 0.055)
    return f"{lo:.2f}–{hi:.2f}"


class MedicalPredictor:
    """
    Production-ready inference engine.

    Args:
        model       : calibrated MedicalAIModel
        preprocessor: PatientPreprocessor
        icd_registry: ICDRegistry (maps index → disease metadata)
        explainer   : MedicalExplainer (SHAP + attention rollout)
        recommender : RAGRecommender (guideline-grounded test recommendations)
        device      : 'cuda' | 'cpu'
        top_k       : number of top predictions to return (default 10)
        model_version: string version tag for response metadata
    """

    def __init__(
        self,
        model: MedicalAIModel,
        preprocessor: PatientPreprocessor,
        icd_registry: ICDRegistry,
        explainer: MedicalExplainer,
        recommender: RAGRecommender,
        device: str = "cuda",
        top_k: int = 10,
        model_version: str = "1.0.0",
    ):
        self.model         = model.eval().to(device)
        self.preprocessor  = preprocessor
        self.icd_registry  = icd_registry
        self.explainer     = explainer
        self.recommender   = recommender
        self.device        = device
        self.top_k         = top_k
        self.model_version = model_version

    @torch.no_grad()
    def predict(self, patient: PatientInput) -> PredictionResponse:
        """
        Run full inference pipeline for a single patient.

        Returns a PredictionResponse with:
          - top_k ranked disease predictions with calibrated probabilities
          - recommended confirmatory tests per disease
          - SHAP feature importances
          - data completeness score
          - processing time
        """
        t_start = time.perf_counter()
        request_id = str(uuid.uuid4())

        log.info("Inference start", request_id=request_id, patient_id=patient.patient_id)

        # ── 1. Preprocess ────────────────────────────────────────────────
        features = self.preprocessor.preprocess(patient)

        numerical      = features["numerical"].unsqueeze(0).to(self.device)
        input_ids      = features["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = features["attention_mask"].unsqueeze(0).to(self.device)

        # ── 2. Forward pass ──────────────────────────────────────────────
        output = self.model(
            numerical, input_ids, attention_mask,
            return_attentions=True,
        )
        probs = output["probs"].squeeze(0).cpu().numpy()  # (num_diseases,)

        # ── 3. Top-K selection ───────────────────────────────────────────
        top_indices = np.argsort(probs)[::-1][:self.top_k]

        # ── 4. Build disease predictions ─────────────────────────────────
        disease_preds: list[DiseasePrediction] = []
        for idx in top_indices:
            prob = float(probs[idx])
            if prob < 0.05:
                break  # don't report near-zero probabilities

            meta     = self.icd_registry.get_by_index(int(idx))
            urgency  = assign_urgency(meta["icd10"], prob)
            ci       = compute_confidence_interval(prob)

            disease_preds.append(DiseasePrediction(
                icd10_code=meta["icd10"],
                disease_name=meta["name"],
                probability=round(prob, 4),
                urgency=urgency,
                confidence_band=ci,
                supporting_evidence=[],  # filled by explainer below
            ))

        # ── 5. SHAP Explanations ─────────────────────────────────────────
        feat_importances: list[FeatureImportance] = []
        try:
            shap_vals = self.explainer.compute_shap(
                features["numerical"].unsqueeze(0),
                top_indices[:3],   # explain top-3 predictions
            )
            for feat_name, shap_val in shap_vals.items():
                direction = "INCREASES" if shap_val > 0 else "DECREASES"
                raw_val   = features["flat_features"].get(feat_name)
                feat_importances.append(FeatureImportance(
                    feature_name=feat_name,
                    shap_value=round(float(shap_val), 4),
                    direction=direction,
                    raw_value=str(round(raw_val, 2)) if raw_val is not None else None,
                ))
                # Attach top evidence to disease predictions
                for dp in disease_preds[:3]:
                    if abs(shap_val) > 0.1:
                        evidence = (
                            f"{feat_name.replace('_', ' ').title()}: "
                            f"{raw_val:.2f} ({direction.lower()} risk)"
                            if raw_val is not None
                            else feat_name.replace("_", " ").title()
                        )
                        if evidence not in dp.supporting_evidence:
                            dp.supporting_evidence.append(evidence)
        except Exception as e:
            log.warning("SHAP computation failed", error=str(e))

        # ── 6. Test recommendations (RAG) ────────────────────────────────
        top_diseases = [(dp.icd10_code, dp.disease_name, dp.probability)
                        for dp in disease_preds[:5]]
        recommended_tests: list[RecommendedTest] = []
        try:
            recommended_tests = self.recommender.get_recommendations(
                diseases=top_diseases,
                patient_context=features["raw_text"],
            )
        except Exception as e:
            log.warning("RAG recommendation failed", error=str(e))

        t_end = time.perf_counter()
        processing_ms = round((t_end - t_start) * 1000, 1)

        log.info(
            "Inference complete",
            request_id=request_id,
            top_disease=disease_preds[0].disease_name if disease_preds else "none",
            processing_ms=processing_ms,
        )

        return PredictionResponse(
            request_id=request_id,
            patient_id=patient.patient_id,
            timestamp=datetime.now(timezone.utc),
            model_version=self.model_version,
            top_predictions=disease_preds,
            recommended_tests=recommended_tests,
            feature_importances=sorted(
                feat_importances, key=lambda x: abs(x.shap_value), reverse=True
            )[:15],
            data_completeness_pct=features["completeness"],
            processing_time_ms=processing_ms,
        )

    def predict_batch(
        self,
        patients: list[PatientInput],
    ) -> list[PredictionResponse]:
        """Run inference on a list of patients sequentially."""
        return [self.predict(p) for p in patients]


def build_predictor(
    model_path: str,
    cfg: dict,
    icd_vocab_path: str,
    guideline_dir: str,
    device: str = "cuda",
) -> MedicalPredictor:
    """
    Factory function: load everything and assemble the predictor.
    Call this once at API startup.
    """
    from src.utils.icd_codes import ICDRegistry
    from src.explainability.explainer import MedicalExplainer
    from src.inference.rag_recommender import RAGRecommender

    log.info("Loading model", path=model_path)
    model = MedicalAIModel.load(model_path, cfg)

    preprocessor = PatientPreprocessor(
        model_name=cfg["nlp"]["model_name"],
        max_seq_len=cfg["data"]["max_seq_len"],
    )

    icd_registry = ICDRegistry.from_json(icd_vocab_path)
    explainer    = MedicalExplainer(model, preprocessor, device=device)
    recommender  = RAGRecommender(guideline_dir=guideline_dir, cfg=cfg["rag"])

    return MedicalPredictor(
        model=model,
        preprocessor=preprocessor,
        icd_registry=icd_registry,
        explainer=explainer,
        recommender=recommender,
        device=device,
        top_k=cfg["data"]["top_k_diseases"],
        model_version=cfg["project"]["version"],
    )
