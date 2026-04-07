"""
src/data/preprocessing.py
──────────────────────────
Converts a PatientInput into the two tensors the model needs:
  1. numerical_tensor  — shape (num_features,)
  2. input_ids, attention_mask — tokenized clinical text

Also computes data_completeness_pct for the response.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from src.data.schema import PatientInput
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Feature order must match training ────────────────────────────────────────
NUMERICAL_FEATURES = [
    "age", "bmi",
    "systolic_bp", "diastolic_bp", "heart_rate",
    "respiratory_rate", "temperature_celsius", "spo2",
    "hemoglobin", "wbc", "platelets",
    "creatinine", "bun", "sodium", "potassium",
    "glucose", "hba1c",
    "ldl", "hdl", "triglycerides",
    "alt", "ast", "bilirubin",
    "troponin", "bnp",
    "crp", "esr", "ferritin",
    "tsh", "ft4",
]

CATEGORICAL_FEATURES = [
    "sex", "smoking_status", "alcohol_use", "diabetes_type",
]

HISTORY_FLAGS = [
    "hypertension", "hyperlipidemia", "coronary_artery_disease",
    "heart_failure", "atrial_fibrillation", "stroke_tia",
    "copd", "asthma", "ckd", "liver_disease", "cancer_history",
    "autoimmune_disease", "hiv", "tuberculosis_history",
    "family_hx_cancer", "family_hx_cvd",
]

# Population-level medians used for missing-value imputation.
# In production these come from fit_scaler() run on training data.
POPULATION_MEDIANS: dict[str, float] = {
    "age": 45.0, "bmi": 26.0,
    "systolic_bp": 120.0, "diastolic_bp": 80.0,
    "heart_rate": 75.0, "respiratory_rate": 16.0,
    "temperature_celsius": 37.0, "spo2": 98.0,
    "hemoglobin": 13.5, "wbc": 7.0, "platelets": 250.0,
    "creatinine": 0.9, "bun": 14.0, "sodium": 140.0, "potassium": 4.0,
    "glucose": 95.0, "hba1c": 5.5,
    "ldl": 100.0, "hdl": 50.0, "triglycerides": 130.0,
    "alt": 25.0, "ast": 25.0, "bilirubin": 0.8,
    "troponin": 0.01, "bnp": 50.0,
    "crp": 3.0, "esr": 15.0, "ferritin": 80.0,
    "tsh": 2.5, "ft4": 1.2,
}

POPULATION_STDS: dict[str, float] = {
    "age": 20.0, "bmi": 5.0,
    "systolic_bp": 20.0, "diastolic_bp": 12.0,
    "heart_rate": 15.0, "respiratory_rate": 4.0,
    "temperature_celsius": 0.5, "spo2": 2.0,
    "hemoglobin": 2.0, "wbc": 3.0, "platelets": 80.0,
    "creatinine": 0.5, "bun": 8.0, "sodium": 4.0, "potassium": 0.5,
    "glucose": 25.0, "hba1c": 1.0,
    "ldl": 35.0, "hdl": 15.0, "triglycerides": 80.0,
    "alt": 20.0, "ast": 20.0, "bilirubin": 0.5,
    "troponin": 0.5, "bnp": 200.0,
    "crp": 10.0, "esr": 15.0, "ferritin": 100.0,
    "tsh": 1.5, "ft4": 0.3,
}


def _flatten_patient(patient: PatientInput) -> dict[str, Optional[float]]:
    """Flatten nested Pydantic model into a flat dict."""
    flat: dict[str, Optional[float]] = {
        "age": patient.age,
        "bmi": patient.bmi,
        "sex": float(patient.sex),
        "smoking_status": float(patient.smoking_status),
        "alcohol_use": float(patient.alcohol_use),
        "diabetes_type": float(patient.diabetes_type),
        "symptom_duration_days": patient.symptom_duration_days,
    }

    if patient.vitals:
        v = patient.vitals
        flat.update({
            "systolic_bp": v.systolic_bp,
            "diastolic_bp": v.diastolic_bp,
            "heart_rate": v.heart_rate,
            "respiratory_rate": v.respiratory_rate,
            "temperature_celsius": v.temperature_celsius,
            "spo2": v.spo2,
        })

    if patient.labs:
        lb = patient.labs
        flat.update({
            "hemoglobin": lb.hemoglobin, "wbc": lb.wbc, "platelets": lb.platelets,
            "creatinine": lb.creatinine, "bun": lb.bun,
            "sodium": lb.sodium, "potassium": lb.potassium,
            "glucose": lb.glucose, "hba1c": lb.hba1c,
            "ldl": lb.ldl, "hdl": lb.hdl, "triglycerides": lb.triglycerides,
            "alt": lb.alt, "ast": lb.ast, "bilirubin": lb.bilirubin,
            "troponin": lb.troponin, "bnp": lb.bnp,
            "crp": lb.crp, "esr": lb.esr, "ferritin": lb.ferritin,
            "tsh": lb.tsh, "ft4": lb.ft4,
        })

    if patient.history:
        h = patient.history
        for flag in HISTORY_FLAGS:
            flat[flag] = float(getattr(h, flag, False))

    return flat


def compute_completeness(flat: dict) -> float:
    """Fraction of expected numerical features that are non-null."""
    expected = set(NUMERICAL_FEATURES) | set(HISTORY_FLAGS)
    present = sum(1 for k in expected if flat.get(k) is not None)
    return round(100.0 * present / len(expected), 1)


def build_numerical_tensor(
    flat: dict[str, Optional[float]],
    medians: dict[str, float] = POPULATION_MEDIANS,
    stds: dict[str, float] = POPULATION_STDS,
) -> torch.Tensor:
    """
    Produce a z-score normalised float32 tensor for all numerical features.
    Missing values → imputed with population median, then normalised.
    Missingness indicators appended as extra binary columns.
    """
    values: list[float] = []
    missingness: list[float] = []

    all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + HISTORY_FLAGS

    for feat in all_features:
        raw = flat.get(feat)
        is_missing = raw is None
        missingness.append(1.0 if is_missing else 0.0)

        if is_missing:
            val = medians.get(feat, 0.0)
        else:
            val = float(raw)

        # z-score normalise only continuous features
        if feat in NUMERICAL_FEATURES:
            std = stds.get(feat, 1.0)
            val = (val - medians.get(feat, 0.0)) / (std if std > 0 else 1.0)

        values.append(val)

    combined = values + missingness
    return torch.tensor(combined, dtype=torch.float32)


def build_clinical_text(patient: PatientInput) -> str:
    """
    Concatenate all free-text fields into a single ClinicalBERT input string.
    Uses a structured template so the model sees consistent formatting.
    """
    parts: list[str] = []

    parts.append(f"Patient: {int(patient.age)}-year-old {'female' if patient.sex == 0 else 'male'}.")

    if patient.bmi:
        parts.append(f"BMI {patient.bmi:.1f} kg/m².")

    if patient.symptom_duration_days is not None:
        parts.append(f"Symptom duration: {patient.symptom_duration_days:.0f} days.")

    parts.append(f"Chief complaint: {patient.chief_complaint.strip()}")

    if patient.history_of_present_illness:
        parts.append(f"HPI: {patient.history_of_present_illness.strip()}")

    if patient.review_of_systems:
        parts.append(f"ROS: {patient.review_of_systems.strip()}")

    if patient.current_medications:
        meds = re.sub(r"\s+", " ", patient.current_medications).strip()
        parts.append(f"Medications: {meds}")

    if patient.allergies:
        parts.append(f"Allergies: {patient.allergies.strip()}")

    if patient.history:
        h = patient.history
        active = [
            flag.replace("_", " ")
            for flag in HISTORY_FLAGS
            if getattr(h, flag, False)
        ]
        if active:
            parts.append("Past medical history: " + ", ".join(active) + ".")

    return " ".join(parts)


class PatientPreprocessor:
    """
    Stateful preprocessor — holds the tokenizer.
    Call preprocess() to get everything the model needs in one shot.
    """

    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                 max_seq_len: int = 512):
        log.info("Loading tokenizer", model=model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_len = max_seq_len

    def preprocess(self, patient: PatientInput) -> dict[str, torch.Tensor]:
        """
        Returns a dict with keys:
          numerical     : (N,)  float32
          input_ids     : (L,)  int64
          attention_mask: (L,)  int64
          completeness  : scalar float
        """
        flat = _flatten_patient(patient)
        completeness = compute_completeness(flat)
        numerical = build_numerical_tensor(flat)

        text = build_clinical_text(patient)
        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "numerical": numerical,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "completeness": completeness,
            "raw_text": text,
            "flat_features": flat,
        }
