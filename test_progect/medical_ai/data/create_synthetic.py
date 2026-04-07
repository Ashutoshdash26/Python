"""
data/create_synthetic.py
─────────────────────────
Creates 500 fake (synthetic) patient records for testing the pipeline.
No real patient data is used. Run this before training.

Usage:
    python data/create_synthetic.py
"""

import json
import random
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
random.seed(42)

n = 500

CLINICAL_TEXTS = [
    "Patient presents with crushing chest pain radiating to the left arm, associated with diaphoresis and nausea. Troponin markedly elevated.",
    "62-year-old male with sudden onset severe chest pain, 9/10 severity, pressure-like. Diaphoretic. ECG changes noted. Troponin rising.",
    "Shortness of breath on exertion, bilateral ankle oedema, raised JVP. BNP significantly elevated. Query decompensated heart failure.",
    "Acute onset dyspnoea, orthopnoea, paroxysmal nocturnal dyspnoea. Bilateral crepitations on auscultation. Heart failure likely.",
    "Fever 38.9°C, productive cough with green sputum, right basal dullness. CXR shows consolidation. Community-acquired pneumonia.",
    "High-grade fever, rigors, hypotension, tachycardia. Blood cultures taken. Raised lactate. Sepsis protocol initiated.",
    "HbA1c 9.8%. Polyuria, polydipsia, blurred vision. Poor medication compliance. Poorly controlled type 2 diabetes mellitus.",
    "Sudden onset severe headache, worst of life. Neck stiffness. Photophobia. CT head performed. Neurological review requested.",
    "Acute onset left-sided weakness and facial droop. FAST positive. Last seen well 2 hours ago. Urgent CT and thrombolysis assessment.",
    "Pleuritic chest pain, haemoptysis, tachycardia. D-dimer elevated. Wells score intermediate. CTPA arranged. Query pulmonary embolism.",
    "Oliguria, rising creatinine, fluid overload. Recent NSAID use. Acute kidney injury on CKD background. Nephrology review.",
    "Epigastric pain radiating to the back, vomiting, elevated amylase and lipase. Nil by mouth. IV fluids commenced. Acute pancreatitis.",
    "Right upper quadrant pain, fever, jaundice. Murphy's sign positive. Ultrasound shows gallstones with wall thickening. Cholecystitis.",
    "Known COPD exacerbation. Increased breathlessness, wheeze, yellow sputum. SpO2 88% on air. Nebulisers and steroids started.",
    "Palpitations, irregularly irregular pulse. ECG confirms atrial fibrillation. Rate 138 bpm. Rate control commenced.",
]

ICD10_POOL = [
    "I21.0", "I21.4", "I20.0", "I50.0", "I50.9",
    "I48.0", "I10",   "I26.9", "I63.9", "G45.9",
    "A41.9", "J18.9", "J96.0", "J44.1", "E11.9",
    "E10.9", "N17.9", "K85.9", "K80.2", "I25.1",
]

data = {
    "patient_id":      [f"P{i:04d}" for i in range(n)],
    "subject_id":      list(range(n)),
    "hadm_id":         list(range(10000, 10000 + n)),
    "age":             np.random.uniform(20, 85, n).round(1),
    "sex":             np.random.randint(0, 2, n),
    "bmi":             np.random.uniform(18, 42, n).round(1),
    "smoking_status":  np.random.randint(0, 3, n),
    "alcohol_use":     np.random.randint(0, 3, n),
    "diabetes_type":   np.random.randint(0, 3, n),
    # Vitals
    "systolic_bp":     np.random.uniform(85, 190, n).round(0),
    "diastolic_bp":    np.random.uniform(55, 115, n).round(0),
    "heart_rate":      np.random.uniform(45, 140, n).round(0),
    "respiratory_rate":np.random.uniform(12, 30, n).round(0),
    "temperature_celsius": np.random.uniform(35.5, 40.2, n).round(1),
    "spo2":            np.random.uniform(85, 100, n).round(0),
    # Labs
    "hemoglobin":      np.random.uniform(7, 17, n).round(1),
    "wbc":             np.random.uniform(2, 20, n).round(1),
    "platelets":       np.random.uniform(80, 450, n).round(0),
    "creatinine":      np.random.uniform(0.5, 5.0, n).round(2),
    "bun":             np.random.uniform(8, 80, n).round(0),
    "sodium":          np.random.uniform(128, 150, n).round(0),
    "potassium":       np.random.uniform(2.8, 6.0, n).round(1),
    "glucose":         np.random.uniform(60, 400, n).round(0),
    "hba1c":           np.random.uniform(4.5, 13.0, n).round(1),
    "ldl":             np.random.uniform(50, 220, n).round(0),
    "hdl":             np.random.uniform(25, 90, n).round(0),
    "triglycerides":   np.random.uniform(50, 500, n).round(0),
    "alt":             np.random.uniform(10, 300, n).round(0),
    "ast":             np.random.uniform(10, 300, n).round(0),
    "bilirubin":       np.random.uniform(0.2, 8.0, n).round(1),
    "troponin":        np.abs(np.random.exponential(0.8, n)).round(3),
    "bnp":             np.abs(np.random.exponential(200, n)).round(0),
    "crp":             np.abs(np.random.exponential(15, n)).round(1),
    "esr":             np.random.uniform(5, 100, n).round(0),
    "ferritin":        np.random.uniform(10, 500, n).round(0),
    "tsh":             np.random.uniform(0.1, 8.0, n).round(2),
    "ft4":             np.random.uniform(0.5, 2.5, n).round(2),
    # History flags
    "hypertension":           np.random.randint(0, 2, n),
    "hyperlipidemia":         np.random.randint(0, 2, n),
    "coronary_artery_disease":np.random.randint(0, 2, n),
    "heart_failure":          np.random.randint(0, 2, n),
    "atrial_fibrillation":    np.random.randint(0, 2, n),
    "stroke_tia":             np.random.randint(0, 2, n),
    "copd":                   np.random.randint(0, 2, n),
    "asthma":                 np.random.randint(0, 2, n),
    "ckd":                    np.random.randint(0, 2, n),
    "liver_disease":          np.random.randint(0, 2, n),
    "cancer_history":         np.random.randint(0, 2, n),
    "autoimmune_disease":     np.random.randint(0, 2, n),
    "hiv":                    np.random.randint(0, 2, n),
    "tuberculosis_history":   np.random.randint(0, 2, n),
    "family_hx_cancer":       np.random.randint(0, 2, n),
    "family_hx_cvd":          np.random.randint(0, 2, n),
    # Text
    "clinical_text": [random.choice(CLINICAL_TEXTS) for _ in range(n)],
    # Labels
    "icd10_labels_json": [
        json.dumps(random.sample(ICD10_POOL, k=random.randint(1, 3)))
        for _ in range(n)
    ],
}

df = pd.DataFrame(data)
df["icd10_labels"] = df["icd10_labels_json"].apply(json.loads)

# Introduce realistic missing values (20% chance per lab)
lab_cols = ["troponin","bnp","hba1c","ldl","hdl","triglycerides",
            "alt","ast","bilirubin","ferritin","tsh","ft4","esr"]
for col in lab_cols:
    mask = np.random.random(n) < 0.20
    df.loc[mask, col] = np.nan

# 70 / 15 / 15 patient-level split
train_df = df.iloc[:350].copy()
val_df   = df.iloc[350:425].copy()
test_df  = df.iloc[425:].copy()

out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)

train_df.to_parquet(out / "train.parquet", index=False)
val_df.to_parquet(  out / "val.parquet",   index=False)
test_df.to_parquet( out / "test.parquet",  index=False)

# Save ICD vocab
from collections import Counter
all_codes = [c for labels in df["icd10_labels"] for c in labels]
top_codes = [code for code, _ in Counter(all_codes).most_common()]
vocab = {code: i for i, code in enumerate(sorted(top_codes))}
with open("data/icd10_vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)

print("=" * 50)
print("  Synthetic data created successfully!")
print("=" * 50)
print(f"  Train : {len(train_df):>5} patients → data/processed/train.parquet")
print(f"  Val   : {len(val_df):>5} patients → data/processed/val.parquet")
print(f"  Test  : {len(test_df):>5} patients → data/processed/test.parquet")
print(f"  Vocab : {len(vocab)} ICD-10 codes → data/icd10_vocab.json")
print("=" * 50)
print("\nNext step: python scripts/train.py --config configs/config.yaml")
