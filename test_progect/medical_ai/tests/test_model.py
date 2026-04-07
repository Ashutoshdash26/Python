"""
tests/test_model.py
────────────────────
Unit and integration tests for MedicalAI components.

Run with:
  pytest tests/ -v
  pytest tests/test_model.py -v -k "test_fusion"
"""

from __future__ import annotations

import json
import pytest
import torch
import numpy as np


# ─────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────

BATCH = 4
SEQ_LEN = 128
TAB_DIM_FEATURES = 30 + 4 + 16   # numerical + categorical + history
TAB_INPUT_DIM = TAB_DIM_FEATURES * 2  # + missingness
NUM_DISEASES = 40   # small for tests


@pytest.fixture(scope="module")
def dummy_numerical():
    return torch.randn(BATCH, TAB_INPUT_DIM)


@pytest.fixture(scope="module")
def dummy_text_inputs():
    return {
        "input_ids":      torch.randint(0, 1000, (BATCH, SEQ_LEN)),
        "attention_mask": torch.ones(BATCH, SEQ_LEN, dtype=torch.long),
    }


@pytest.fixture(scope="module")
def tabular_encoder():
    from src.models.tabular_encoder import TabularEncoderNet
    return TabularEncoderNet(input_dim=TAB_INPUT_DIM, output_dim=256)


@pytest.fixture(scope="module")
def fusion_model():
    from src.models.fusion_model import CrossModalFusion
    return CrossModalFusion(tab_dim=256, nlp_dim=256, output_dim=512)


# ─────────────────────────────────────────────────────────────────────────
#  Tabular Encoder Tests
# ─────────────────────────────────────────────────────────────────────────

class TestTabularEncoder:

    def test_output_shape(self, tabular_encoder, dummy_numerical):
        emb, ent = tabular_encoder(dummy_numerical)
        assert emb.shape == (BATCH, 256), f"Expected ({BATCH}, 256), got {emb.shape}"

    def test_entropy_loss_scalar(self, tabular_encoder, dummy_numerical):
        _, entropy = tabular_encoder(dummy_numerical)
        assert entropy.shape == torch.Size([1]) or entropy.numel() == 1

    def test_entropy_loss_positive(self, tabular_encoder, dummy_numerical):
        _, entropy = tabular_encoder(dummy_numerical)
        assert float(entropy) >= 0.0

    def test_gradients_flow(self, dummy_numerical):
        from src.models.tabular_encoder import TabularEncoderNet
        model = TabularEncoderNet(input_dim=TAB_INPUT_DIM, output_dim=256)
        x = dummy_numerical.clone().requires_grad_(True)
        emb, ent = model(x)
        loss = emb.sum() + ent
        loss.backward()
        assert x.grad is not None

    def test_batch_size_one(self, tabular_encoder):
        x = torch.randn(1, TAB_INPUT_DIM)
        emb, _ = tabular_encoder(x)
        assert emb.shape == (1, 256)

    def test_missing_values_handled(self, tabular_encoder):
        """NaN/zero inputs (representing missing) should not cause NaN outputs."""
        x = torch.zeros(BATCH, TAB_INPUT_DIM)
        emb, _ = tabular_encoder(x)
        assert not torch.isnan(emb).any(), "NaN in tabular encoder output"


# ─────────────────────────────────────────────────────────────────────────
#  Fusion Model Tests
# ─────────────────────────────────────────────────────────────────────────

class TestFusionModel:

    def test_output_shape(self, fusion_model):
        tab = torch.randn(BATCH, 256)
        nlp = torch.randn(BATCH, 256)
        out = fusion_model(tab, nlp)
        assert out.shape == (BATCH, 512)

    def test_modality_weights_sum_to_one(self, fusion_model):
        tab = torch.randn(1, 256)
        nlp = torch.randn(1, 256)
        weights = fusion_model.get_modality_weights(tab, nlp)
        total = weights["tabular"] + weights["nlp"]
        assert abs(total - 1.0) < 0.01, f"Weights don't sum to 1: {weights}"

    def test_different_inputs_different_outputs(self, fusion_model):
        tab1 = torch.randn(1, 256)
        nlp1 = torch.randn(1, 256)
        tab2 = torch.randn(1, 256)
        nlp2 = torch.randn(1, 256)
        out1 = fusion_model(tab1, nlp1)
        out2 = fusion_model(tab2, nlp2)
        assert not torch.allclose(out1, out2), "Different inputs produced identical outputs"


# ─────────────────────────────────────────────────────────────────────────
#  Full Model Tests (lightweight — no BERT download)
# ─────────────────────────────────────────────────────────────────────────

class TestDiseaseClassifierHead:

    def test_output_shape(self):
        from src.models.full_model import DiseaseClassifierHead
        head = DiseaseClassifierHead(input_dim=512, num_diseases=NUM_DISEASES)
        fused = torch.randn(BATCH, 512)
        out   = head(fused)
        assert out["probs"].shape == (BATCH, NUM_DISEASES)
        assert out["logits"].shape == (BATCH, NUM_DISEASES)

    def test_probs_in_range(self):
        from src.models.full_model import DiseaseClassifierHead
        head = DiseaseClassifierHead(input_dim=512, num_diseases=NUM_DISEASES)
        fused = torch.randn(BATCH, 512)
        probs = head(fused)["probs"]
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_temperature_scaling(self):
        from src.models.full_model import DiseaseClassifierHead
        head = DiseaseClassifierHead(input_dim=512, num_diseases=NUM_DISEASES)
        fused = torch.randn(1, 512)

        probs_T1 = head(fused)["probs"]
        head.set_temperature(2.0)
        probs_T2 = head(fused)["probs"]

        # Higher temperature → probs closer to 0.5 (less extreme)
        dist_T1 = (probs_T1 - 0.5).abs().mean()
        dist_T2 = (probs_T2 - 0.5).abs().mean()
        assert dist_T2 < dist_T1, "Higher temperature should produce softer probabilities"


# ─────────────────────────────────────────────────────────────────────────
#  Preprocessing Tests
# ─────────────────────────────────────────────────────────────────────────

class TestPreprocessing:

    def _make_patient(self, **overrides):
        from src.data.schema import PatientInput
        defaults = dict(
            patient_id="test_001",
            age=55.0,
            sex=1,  # male
            chief_complaint="Chest pain radiating to left arm, diaphoresis",
            smoking_status=1,
        )
        defaults.update(overrides)
        return PatientInput(**defaults)

    def test_build_numerical_tensor_shape(self):
        from src.data.preprocessing import build_numerical_tensor, POPULATION_MEDIANS, POPULATION_STDS
        from src.data.preprocessing import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, HISTORY_FLAGS
        flat = {f: None for f in NUMERICAL_FEATURES + CATEGORICAL_FEATURES + HISTORY_FLAGS}
        flat["age"] = 45.0
        t = build_numerical_tensor(flat)
        expected_len = (len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES) + len(HISTORY_FLAGS)) * 2
        assert t.shape == (expected_len,)

    def test_no_nan_in_tensor(self):
        from src.data.preprocessing import build_numerical_tensor
        from src.data.preprocessing import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, HISTORY_FLAGS
        flat = {f: None for f in NUMERICAL_FEATURES + CATEGORICAL_FEATURES + HISTORY_FLAGS}
        t = build_numerical_tensor(flat)
        assert not torch.isnan(t).any()

    def test_completeness_zero_missing(self):
        from src.data.preprocessing import (
            _flatten_patient, compute_completeness, NUMERICAL_FEATURES, HISTORY_FLAGS
        )
        from src.data.schema import PatientInput, Vitals, LabResults, PatientHistory
        patient = PatientInput(
            patient_id="p1", age=40.0, sex=0,
            chief_complaint="Cough and fever",
        )
        flat = _flatten_patient(patient)
        completeness = compute_completeness(flat)
        assert 0.0 <= completeness <= 100.0

    def test_clinical_text_contains_complaint(self):
        from src.data.preprocessing import build_clinical_text
        patient = self._make_patient()
        text = build_clinical_text(patient)
        assert "Chest pain" in text

    def test_patientinput_validation_missing_complaint(self):
        import pydantic
        from src.data.schema import PatientInput
        with pytest.raises((pydantic.ValidationError, ValueError)):
            PatientInput(patient_id="x", age=40, sex=0, chief_complaint="")


# ─────────────────────────────────────────────────────────────────────────
#  Loss Function Tests
# ─────────────────────────────────────────────────────────────────────────

class TestLosses:

    def test_asl_loss_positive(self):
        from src.training.losses import AsymmetricFocalLoss
        loss_fn = AsymmetricFocalLoss()
        logits  = torch.randn(8, 40)
        targets = (torch.rand(8, 40) > 0.8).float()
        loss    = loss_fn(logits, targets)
        assert float(loss) > 0.0
        assert not torch.isnan(loss)

    def test_combined_loss_keys(self):
        from src.training.losses import CombinedLoss
        loss_fn    = CombinedLoss()
        logits     = torch.randn(4, 40)
        targets    = (torch.rand(4, 40) > 0.8).float()
        entropy    = torch.tensor(0.1)
        loss_dict  = loss_fn(logits, targets, entropy)
        assert "total" in loss_dict
        assert "asl"   in loss_dict
        assert "entropy" in loss_dict

    def test_higher_gamma_lower_easy_negative_loss(self):
        """ASL with higher gamma_neg should down-weight easy negatives."""
        from src.training.losses import AsymmetricFocalLoss
        logits  = -3.0 * torch.ones(4, 10)  # confident negatives
        targets = torch.zeros(4, 10)
        loss_low  = AsymmetricFocalLoss(gamma_neg=1.0)(logits, targets)
        loss_high = AsymmetricFocalLoss(gamma_neg=5.0)(logits, targets)
        assert float(loss_high) < float(loss_low)


# ─────────────────────────────────────────────────────────────────────────
#  Metrics Tests
# ─────────────────────────────────────────────────────────────────────────

class TestMetrics:

    def test_auroc_perfect_predictor(self):
        from src.utils.metrics import compute_epoch_metrics
        n, c = 200, 10
        labels = (np.random.rand(n, c) > 0.8).astype(float)
        # Perfect predictor: probs = labels
        probs  = labels.copy()
        metrics = compute_epoch_metrics(probs, labels)
        assert metrics["auroc_macro"] > 0.99

    def test_auroc_random_predictor(self):
        from src.utils.metrics import compute_epoch_metrics
        np.random.seed(42)
        n, c   = 500, 20
        labels = (np.random.rand(n, c) > 0.8).astype(float)
        probs  = np.random.rand(n, c)
        metrics = compute_epoch_metrics(probs, labels, min_positive_samples=5)
        assert 0.4 < metrics["auroc_macro"] < 0.6

    def test_ece_calibrated(self):
        """A calibrated predictor should have low ECE."""
        from src.utils.metrics import _expected_calibration_error
        n = 1000
        probs  = np.random.uniform(0, 1, n)
        labels = np.random.binomial(1, probs)   # perfectly calibrated
        ece    = _expected_calibration_error(probs, labels.astype(float))
        assert ece < 0.05, f"ECE too high for calibrated predictor: {ece:.4f}"


# ─────────────────────────────────────────────────────────────────────────
#  ICD Registry Tests
# ─────────────────────────────────────────────────────────────────────────

class TestICDRegistry:

    def test_builtin_registry_loads(self):
        from src.utils.icd_codes import ICDRegistry
        reg = ICDRegistry.builtin()
        assert len(reg) > 0

    def test_get_by_index(self):
        from src.utils.icd_codes import ICDRegistry
        reg  = ICDRegistry.builtin()
        meta = reg.get_by_index(0)
        assert "icd10" in meta
        assert "name"  in meta

    def test_get_by_code(self):
        from src.utils.icd_codes import ICDRegistry
        reg  = ICDRegistry.builtin()
        meta = reg.get_by_code("I21.0")
        assert meta is not None
        assert "STEMI" in meta["name"] or "myocardial" in meta["name"].lower()

    def test_unknown_index_returns_placeholder(self):
        from src.utils.icd_codes import ICDRegistry
        reg  = ICDRegistry.builtin()
        meta = reg.get_by_index(99999)
        assert "UNKNOWN" in meta["icd10"]


# ─────────────────────────────────────────────────────────────────────────
#  RAG Recommender Tests
# ─────────────────────────────────────────────────────────────────────────

class TestRAGRecommender:

    @pytest.fixture
    def recommender(self, tmp_path):
        from src.inference.rag_recommender import RAGRecommender
        cfg = {"embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
               "chunk_size": 512, "chunk_overlap": 64, "top_k_chunks": 3}
        return RAGRecommender(guideline_dir=str(tmp_path), cfg=cfg)

    def test_known_disease_returns_tests(self, recommender):
        diseases = [("I21.0", "STEMI", 0.85)]
        tests = recommender.get_recommendations(diseases, "Chest pain, diaphoresis")
        assert len(tests) > 0
        test_names = [t.test_name for t in tests]
        assert any("ECG" in n or "Troponin" in n for n in test_names)

    def test_unknown_disease_returns_defaults(self, recommender):
        diseases = [("Z99.9", "Unknown rare syndrome", 0.40)]
        tests = recommender.get_recommendations(diseases, "Unknown symptoms")
        assert len(tests) > 0

    def test_all_outputs_have_required_fields(self, recommender):
        diseases = [("I10", "Hypertension", 0.70), ("E11.9", "Type 2 DM", 0.65)]
        tests = recommender.get_recommendations(diseases, "Headache, polyuria")
        for t in tests:
            assert t.test_name
            assert t.rationale
            assert t.guideline_source
            assert t.priority in ("FIRST_LINE", "SECOND_LINE", "OPTIONAL")


# ─────────────────────────────────────────────────────────────────────────
#  Schema Validation Tests
# ─────────────────────────────────────────────────────────────────────────

class TestSchema:

    def test_valid_patient_input(self):
        from src.data.schema import PatientInput, Vitals, LabResults
        patient = PatientInput(
            patient_id="pt_001",
            age=62.0, sex=1,
            chief_complaint="Acute onset shortness of breath and pleuritic chest pain",
            vitals=Vitals(systolic_bp=145, heart_rate=110, spo2=91),
            labs=LabResults(d_dimer=None, troponin=0.02),
        )
        assert patient.patient_id == "pt_001"

    def test_invalid_age_raises(self):
        import pydantic
        from src.data.schema import PatientInput
        with pytest.raises(pydantic.ValidationError):
            PatientInput(patient_id="x", age=-5, sex=0,
                         chief_complaint="Test complaint")

    def test_invalid_spo2_raises(self):
        import pydantic
        from src.data.schema import PatientInput, Vitals
        with pytest.raises(pydantic.ValidationError):
            PatientInput(patient_id="x", age=40, sex=0,
                         chief_complaint="Test",
                         vitals=Vitals(spo2=150))  # > 100%

    def test_urgency_stat_for_cardiac(self):
        from src.inference.predictor import assign_urgency
        assert assign_urgency("I21", 0.85) == "STAT"

    def test_urgency_routine_low_prob(self):
        from src.inference.predictor import assign_urgency
        assert assign_urgency("I21", 0.10) == "ROUTINE"
