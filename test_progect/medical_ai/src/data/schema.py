"""
src/data/schema.py
──────────────────
Pydantic v2 schemas for all API input / output.
Every field has a description so the OpenAPI docs are self-documenting.
"""

from __future__ import annotations

from datetime import datetime
from enum import IntEnum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────
#  Enumerations
# ─────────────────────────────────────────────────────────

class Sex(IntEnum):
    FEMALE = 0
    MALE = 1
    OTHER = 2

class SmokingStatus(IntEnum):
    NEVER = 0
    FORMER = 1
    CURRENT = 2

class AlcoholUse(IntEnum):
    NONE = 0
    MODERATE = 1
    HEAVY = 2

class DiabetesType(IntEnum):
    NONE = 0
    TYPE_1 = 1
    TYPE_2 = 2

class UrgencyLevel(str):
    STAT = "STAT"           # immediate — life-threatening
    URGENT = "URGENT"       # within 4 hours
    ROUTINE = "ROUTINE"     # within 24–48 hours


# ─────────────────────────────────────────────────────────
#  Patient Input
# ─────────────────────────────────────────────────────────

class Vitals(BaseModel):
    systolic_bp: Optional[float]       = Field(None, ge=40,  le=300,  description="Systolic blood pressure mmHg")
    diastolic_bp: Optional[float]      = Field(None, ge=20,  le=200,  description="Diastolic blood pressure mmHg")
    heart_rate: Optional[float]        = Field(None, ge=20,  le=300,  description="Heart rate bpm")
    respiratory_rate: Optional[float]  = Field(None, ge=4,   le=60,   description="Breaths per minute")
    temperature_celsius: Optional[float] = Field(None, ge=30, le=45,  description="Body temperature °C")
    spo2: Optional[float]              = Field(None, ge=50,  le=100,  description="Oxygen saturation %")


class LabResults(BaseModel):
    # Haematology
    hemoglobin: Optional[float]    = Field(None, ge=2,   le=25,   description="g/dL")
    wbc: Optional[float]           = Field(None, ge=0.1, le=100,  description="White blood cells ×10³/µL")
    platelets: Optional[float]     = Field(None, ge=1,   le=2000, description="×10³/µL")

    # Renal
    creatinine: Optional[float]    = Field(None, ge=0.1, le=30,   description="mg/dL")
    bun: Optional[float]           = Field(None, ge=1,   le=200,  description="Blood urea nitrogen mg/dL")

    # Electrolytes
    sodium: Optional[float]        = Field(None, ge=100, le=180,  description="mEq/L")
    potassium: Optional[float]     = Field(None, ge=1.5, le=9,    description="mEq/L")

    # Metabolic
    glucose: Optional[float]       = Field(None, ge=20,  le=800,  description="Fasting glucose mg/dL")
    hba1c: Optional[float]         = Field(None, ge=3,   le=20,   description="HbA1c %")

    # Lipids
    ldl: Optional[float]           = Field(None, ge=10,  le=500,  description="mg/dL")
    hdl: Optional[float]           = Field(None, ge=5,   le=200,  description="mg/dL")
    triglycerides: Optional[float] = Field(None, ge=20,  le=3000, description="mg/dL")

    # Liver
    alt: Optional[float]           = Field(None, ge=1,   le=5000, description="U/L")
    ast: Optional[float]           = Field(None, ge=1,   le=5000, description="U/L")
    bilirubin: Optional[float]     = Field(None, ge=0.1, le=50,   description="Total bilirubin mg/dL")

    # Cardiac
    troponin: Optional[float]      = Field(None, ge=0,   le=1000, description="Troponin I ng/mL")
    bnp: Optional[float]           = Field(None, ge=0,   le=5000, description="BNP pg/mL")

    # Inflammatory
    crp: Optional[float]           = Field(None, ge=0,   le=500,  description="C-reactive protein mg/L")
    esr: Optional[float]           = Field(None, ge=0,   le=150,  description="mm/hr")
    ferritin: Optional[float]      = Field(None, ge=1,   le=10000, description="ng/mL")

    # Thyroid
    tsh: Optional[float]           = Field(None, ge=0,   le=100,  description="mIU/L")
    ft4: Optional[float]           = Field(None, ge=0,   le=10,   description="Free T4 ng/dL")


class PatientHistory(BaseModel):
    """Structured past medical history flags."""
    hypertension: bool         = False
    hyperlipidemia: bool       = False
    coronary_artery_disease: bool = False
    heart_failure: bool        = False
    atrial_fibrillation: bool  = False
    stroke_tia: bool           = False
    copd: bool                 = False
    asthma: bool               = False
    ckd: bool                  = False
    liver_disease: bool        = False
    cancer_history: bool       = False
    autoimmune_disease: bool   = False
    hiv: bool                  = False
    tuberculosis_history: bool = False
    family_hx_cancer: bool     = False
    family_hx_cvd: bool        = False


class PatientInput(BaseModel):
    """
    Complete patient case history submitted for disease prediction.
    At minimum, chief_complaint + age + sex are required.
    """
    # ── Demographics ─────────────────────────────────────────
    patient_id: str              = Field(...,  description="Anonymised patient identifier")
    age: float                   = Field(...,  ge=0, le=120, description="Age in years")
    sex: Sex                     = Field(...,  description="0=female 1=male 2=other")
    bmi: Optional[float]         = Field(None, ge=10, le=80, description="kg/m²")

    # ── Lifestyle ────────────────────────────────────────────
    smoking_status: SmokingStatus   = Field(SmokingStatus.NEVER)
    alcohol_use: AlcoholUse         = Field(AlcoholUse.NONE)
    diabetes_type: DiabetesType     = Field(DiabetesType.NONE)

    # ── Clinical text ────────────────────────────────────────
    chief_complaint: str         = Field(..., min_length=5, max_length=2000,
                                         description="Patient's presenting complaint in free text")
    history_of_present_illness: Optional[str] = Field(None, max_length=5000,
                                         description="Detailed HPI narrative")
    review_of_systems: Optional[str]  = Field(None, max_length=3000)
    current_medications: Optional[str] = Field(None, max_length=2000,
                                         description="Comma-separated or free-text medication list")
    allergies: Optional[str]          = Field(None, max_length=500)

    # ── Structured clinical data ──────────────────────────────
    vitals: Optional[Vitals]          = None
    labs: Optional[LabResults]        = None
    history: Optional[PatientHistory] = None

    # ── Duration ─────────────────────────────────────────────
    symptom_duration_days: Optional[float] = Field(None, ge=0, le=3650,
                                         description="Duration of current symptoms in days")

    @field_validator("age")
    @classmethod
    def age_must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Age cannot be negative")
        return v

    @model_validator(mode="after")
    def at_least_one_clinical_input(self) -> "PatientInput":
        has_text = bool(self.chief_complaint)
        has_labs = self.labs is not None
        has_vitals = self.vitals is not None
        if not (has_text or has_labs or has_vitals):
            raise ValueError("At least one of: chief_complaint, labs, or vitals must be provided")
        return self


# ─────────────────────────────────────────────────────────
#  Prediction Output
# ─────────────────────────────────────────────────────────

class DiseasePrediction(BaseModel):
    icd10_code: str          = Field(..., description="ICD-10-CM code e.g. I21.0")
    disease_name: str        = Field(..., description="Human-readable disease name")
    probability: float       = Field(..., ge=0, le=1, description="Calibrated probability 0–1")
    urgency: str             = Field(..., description="STAT | URGENT | ROUTINE")
    confidence_band: str     = Field(..., description="e.g. 0.68–0.79 (95% CI)")
    supporting_evidence: list[str] = Field(default_factory=list,
                                           description="Key features driving this prediction")


class RecommendedTest(BaseModel):
    test_name: str           = Field(..., description="Name of investigation")
    rationale: str           = Field(..., description="Why this test is recommended")
    guideline_source: str    = Field(..., description="e.g. ACC/AHA 2023, NICE NG185")
    priority: str            = Field(..., description="FIRST_LINE | SECOND_LINE | OPTIONAL")
    expected_turnaround: str = Field(..., description="e.g. 1 hour, 24 hours")


class FeatureImportance(BaseModel):
    feature_name: str
    shap_value: float
    direction: str           = Field(..., description="INCREASES or DECREASES risk")
    raw_value: Optional[str] = None


class PredictionResponse(BaseModel):
    request_id: str
    patient_id: str
    timestamp: datetime
    model_version: str

    # Core predictions
    top_predictions: list[DiseasePrediction]
    recommended_tests: list[RecommendedTest]
    feature_importances: list[FeatureImportance]

    # Meta
    data_completeness_pct: float = Field(..., description="% of expected features present")
    disclaimer: str = (
        "This output is an AI-generated clinical decision support opinion. "
        "It does not constitute a diagnosis. All clinical decisions must be "
        "made by a qualified medical professional."
    )
    processing_time_ms: float
