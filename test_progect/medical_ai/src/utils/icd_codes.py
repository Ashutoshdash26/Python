"""
src/utils/icd_codes.py
───────────────────────
ICD-10-CM code registry.
Maps integer index ↔ ICD-10 code ↔ human-readable disease name.

In production, loaded from a JSON file generated from the official
CMS ICD-10-CM tabular file (public domain).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


# ── Minimal built-in registry for development / testing ──────────────────
# Production version has all ~70,000 ICD-10 codes.
# We only train on the ~1,000 most prevalent codes in clinical datasets.
_BUILTIN_CODES: list[dict] = [
    {"index": 0,  "icd10": "I21.0", "name": "ST elevation myocardial infarction (STEMI)",     "system": "Cardiovascular"},
    {"index": 1,  "icd10": "I21.4", "name": "Non-ST elevation myocardial infarction (NSTEMI)", "system": "Cardiovascular"},
    {"index": 2,  "icd10": "I20.0", "name": "Unstable angina",                                 "system": "Cardiovascular"},
    {"index": 3,  "icd10": "I50.0", "name": "Congestive heart failure",                        "system": "Cardiovascular"},
    {"index": 4,  "icd10": "I48.0", "name": "Atrial fibrillation",                             "system": "Cardiovascular"},
    {"index": 5,  "icd10": "I10",   "name": "Essential hypertension",                          "system": "Cardiovascular"},
    {"index": 6,  "icd10": "I26.9", "name": "Pulmonary embolism",                              "system": "Respiratory"},
    {"index": 7,  "icd10": "I63.9", "name": "Ischaemic stroke",                                "system": "Neurological"},
    {"index": 8,  "icd10": "G45.9", "name": "Transient ischaemic attack (TIA)",                "system": "Neurological"},
    {"index": 9,  "icd10": "A41.9", "name": "Sepsis, unspecified",                             "system": "Infectious"},
    {"index": 10, "icd10": "J18.9", "name": "Pneumonia, unspecified",                          "system": "Respiratory"},
    {"index": 11, "icd10": "J96.0", "name": "Acute respiratory failure",                       "system": "Respiratory"},
    {"index": 12, "icd10": "J44.1", "name": "Chronic obstructive pulmonary disease (COPD)",   "system": "Respiratory"},
    {"index": 13, "icd10": "J45.9", "name": "Asthma, unspecified",                             "system": "Respiratory"},
    {"index": 14, "icd10": "E11.9", "name": "Type 2 diabetes mellitus",                       "system": "Endocrine"},
    {"index": 15, "icd10": "E10.9", "name": "Type 1 diabetes mellitus",                       "system": "Endocrine"},
    {"index": 16, "icd10": "E05.9", "name": "Hyperthyroidism, unspecified",                    "system": "Endocrine"},
    {"index": 17, "icd10": "E03.9", "name": "Hypothyroidism, unspecified",                    "system": "Endocrine"},
    {"index": 18, "icd10": "N17.9", "name": "Acute kidney injury",                             "system": "Renal"},
    {"index": 19, "icd10": "N18.3", "name": "Chronic kidney disease, stage 3",                "system": "Renal"},
    {"index": 20, "icd10": "N39.0", "name": "Urinary tract infection",                         "system": "Renal"},
    {"index": 21, "icd10": "K92.1", "name": "Melaena (GI haemorrhage)",                       "system": "Gastrointestinal"},
    {"index": 22, "icd10": "K80.2", "name": "Cholecystitis",                                   "system": "Gastrointestinal"},
    {"index": 23, "icd10": "K85.9", "name": "Acute pancreatitis",                              "system": "Gastrointestinal"},
    {"index": 24, "icd10": "K57.3", "name": "Diverticulitis",                                  "system": "Gastrointestinal"},
    {"index": 25, "icd10": "M79.3", "name": "Fibromyalgia",                                    "system": "Musculoskeletal"},
    {"index": 26, "icd10": "M54.5", "name": "Low back pain",                                   "system": "Musculoskeletal"},
    {"index": 27, "icd10": "G40.9", "name": "Epilepsy, unspecified",                           "system": "Neurological"},
    {"index": 28, "icd10": "G43.9", "name": "Migraine",                                        "system": "Neurological"},
    {"index": 29, "icd10": "F32.9", "name": "Major depressive disorder",                       "system": "Psychiatric"},
    {"index": 30, "icd10": "F41.1", "name": "Generalised anxiety disorder",                    "system": "Psychiatric"},
    {"index": 31, "icd10": "C34.9", "name": "Lung carcinoma, unspecified",                    "system": "Oncology"},
    {"index": 32, "icd10": "C50.9", "name": "Breast carcinoma, unspecified",                  "system": "Oncology"},
    {"index": 33, "icd10": "C18.9", "name": "Colorectal carcinoma",                           "system": "Oncology"},
    {"index": 34, "icd10": "B20",   "name": "HIV disease",                                     "system": "Infectious"},
    {"index": 35, "icd10": "A15.0", "name": "Pulmonary tuberculosis",                          "system": "Infectious"},
    {"index": 36, "icd10": "M05.9", "name": "Rheumatoid arthritis",                            "system": "Autoimmune"},
    {"index": 37, "icd10": "M32.9", "name": "Systemic lupus erythematosus",                   "system": "Autoimmune"},
    {"index": 38, "icd10": "D50.9", "name": "Iron-deficiency anaemia",                        "system": "Haematology"},
    {"index": 39, "icd10": "D64.9", "name": "Anaemia, unspecified",                           "system": "Haematology"},
]


class ICDRegistry:
    """
    ICD-10 code registry for mapping model output indices to disease metadata.

    Attributes:
        codes        : list of code dicts sorted by index
        index_to_code: dict {index: code_dict}
        code_to_index: dict {icd10_str: index}
        vocab        : dict {icd10_str: index} (alias for code_to_index)
    """

    def __init__(self, codes: list[dict]):
        self.codes         = sorted(codes, key=lambda c: c["index"])
        self.index_to_code = {c["index"]: c for c in self.codes}
        self.code_to_index = {c["icd10"]: c["index"] for c in self.codes}
        self.vocab         = self.code_to_index

    def get_by_index(self, idx: int) -> dict:
        """Retrieve code metadata by model output index."""
        return self.index_to_code.get(idx, {
            "index": idx,
            "icd10": f"UNKNOWN_{idx}",
            "name": "Unknown disease",
            "system": "Unknown",
        })

    def get_by_code(self, code: str) -> Optional[dict]:
        """Retrieve code metadata by ICD-10 code string."""
        idx = self.code_to_index.get(code)
        return self.index_to_code.get(idx) if idx is not None else None

    def __len__(self) -> int:
        return len(self.codes)

    @classmethod
    def builtin(cls) -> "ICDRegistry":
        """Return registry with the built-in minimal code list."""
        return cls(_BUILTIN_CODES)

    @classmethod
    def from_json(cls, path: str) -> "ICDRegistry":
        """Load from a JSON file (list of code dicts)."""
        p = Path(path)
        if p.exists():
            with open(p) as f:
                codes = json.load(f)
            return cls(codes)
        # Fall back to built-in if file not found
        return cls.builtin()

    def to_json(self, path: str) -> None:
        """Serialise registry to JSON."""
        with open(path, "w") as f:
            json.dump(self.codes, f, indent=2)
