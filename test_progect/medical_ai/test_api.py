"""
test_api.py
────────────
Quick test script — sends a sample cardiac patient to the running API
and prints the predictions in a readable format.

Usage:
    1. Start the server:  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
    2. In a second terminal: python test_api.py
"""

import os
import sys
import json
import requests

# ── Configuration ────────────────────────────────────────────────────────
BASE_URL = "http://localhost:8000"
TOKEN    = os.environ.get("DEV_TOKEN", "my-test-token-123")
HEADERS  = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type":  "application/json",
}

# ── Test 1: Health check ─────────────────────────────────────────────────
def test_health():
    print("\n── Health check ────────────────────────────────────")
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/health", timeout=5)
        data = resp.json()
        status = "OK" if data.get("ready") else "NOT READY"
        print(f"  Status  : {status}")
        print(f"  Version : {data.get('version','?')}")
        return data.get("ready", False)
    except requests.exceptions.ConnectionError:
        print("  ERROR: Cannot connect to server.")
        print("  Make sure the server is running:")
        print("  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")
        return False


# ── Test 2: Cardiac patient prediction ───────────────────────────────────
def test_cardiac_patient():
    print("\n── Cardiac patient prediction ──────────────────────")
    patient = {
        "patient_id": "TEST_CARDIAC_001",
        "age": 62,
        "sex": 1,
        "bmi": 28.4,
        "smoking_status": 1,
        "diabetes_type": 2,
        "chief_complaint": (
            "Sudden onset severe crushing chest pain radiating to the left arm "
            "and jaw. Associated with profuse diaphoresis and nausea. "
            "Started 2 hours ago at rest."
        ),
        "history_of_present_illness": (
            "62-year-old male with known hypertension and type 2 diabetes. "
            "Troponin markedly elevated at 2.8 ng/mL. ECG shows ST elevation "
            "in leads II, III and aVF."
        ),
        "vitals": {
            "systolic_bp": 158,
            "diastolic_bp": 96,
            "heart_rate": 102,
            "respiratory_rate": 20,
            "temperature_celsius": 37.1,
            "spo2": 95
        },
        "labs": {
            "troponin": 2.8,
            "bnp": 420,
            "crp": 18.0,
            "wbc": 11.2,
            "hemoglobin": 13.8,
            "creatinine": 1.1,
            "glucose": 162,
            "hba1c": 7.4,
        },
        "history": {
            "hypertension": True,
            "hyperlipidemia": True,
            "family_hx_cvd": True,
        }
    }

    try:
        resp = requests.post(
            f"{BASE_URL}/api/v1/predict",
            json=patient,
            headers=HEADERS,
            timeout=60,
        )

        if resp.status_code != 200:
            print(f"  ERROR {resp.status_code}: {resp.text[:300]}")
            return

        result = resp.json()

        print(f"\n  Patient  : {result['patient_id']}")
        print(f"  Request  : {result['request_id'][:16]}...")
        print(f"  Data completeness: {result['data_completeness_pct']}%")
        print(f"  Processing time  : {result['processing_time_ms']} ms")

        print(f"\n  TOP DISEASE PREDICTIONS:")
        print(f"  {'Disease':<45} {'Prob':>6}  {'Urgency':<10}")
        print(f"  {'-'*45} {'-'*6}  {'-'*10}")
        for pred in result.get("top_predictions", []):
            bar = "█" * int(pred["probability"] * 20)
            print(f"  {pred['disease_name'][:45]:<45} "
                  f"{pred['probability']:>5.0%}  "
                  f"{pred['urgency']:<10}")

        print(f"\n  RECOMMENDED TESTS:")
        for i, test in enumerate(result.get("recommended_tests", [])[:5], 1):
            print(f"  {i}. [{test['priority']}] {test['test_name']}")
            print(f"     Source: {test['guideline_source']}")

        print(f"\n  TOP FEATURE IMPORTANCES (what drove these predictions):")
        for fi in result.get("feature_importances", [])[:5]:
            direction = "↑" if fi["direction"] == "INCREASES" else "↓"
            val = f"= {fi['raw_value']}" if fi.get("raw_value") else ""
            print(f"  {direction} {fi['feature_name']:<25} {val}  (SHAP: {fi['shap_value']:+.3f})")

        print(f"\n  DISCLAIMER: {result['disclaimer'][:80]}...")

    except requests.exceptions.ConnectionError:
        print("  ERROR: Cannot connect. Is the server running?")
    except Exception as e:
        print(f"  ERROR: {e}")


# ── Test 3: Model info ────────────────────────────────────────────────────
def test_model_info():
    print("\n── Model information ───────────────────────────────")
    try:
        resp = requests.get(
            f"{BASE_URL}/api/v1/model/info",
            headers=HEADERS,
            timeout=10
        )
        if resp.status_code == 200:
            info = resp.json()
            print(f"  Version       : {info.get('model_version')}")
            print(f"  NLP backbone  : {info.get('nlp_backbone')}")
            print(f"  Num diseases  : {info.get('num_diseases')}")
            print(f"  Device        : {info.get('device')}")
            params = info.get("parameter_counts", {})
            total = params.get("total", 0)
            print(f"  Total params  : {total:,}")
    except Exception as e:
        print(f"  ERROR: {e}")


# ── Run all tests ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  MedicalAI API Test")
    print("=" * 55)

    ready = test_health()
    if not ready:
        print("\nServer not ready. Exiting.")
        sys.exit(1)

    test_model_info()
    test_cardiac_patient()

    print("\n" + "=" * 55)
    print("  Test complete.")
    print(f"  API docs: {BASE_URL}/api/docs")
    print("=" * 55)
