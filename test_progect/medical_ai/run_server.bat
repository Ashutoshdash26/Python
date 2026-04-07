@echo off
echo ================================================
echo   MedicalAI - Starting API Server
echo ================================================
echo.

call conda activate medicalai

set DEV_TOKEN=my-test-token-123
set MODEL_PATH=outputs\models\calibrated_model.pt
set ICD_VOCAB_PATH=data\icd10_vocab.json
set GUIDELINE_DIR=data\guidelines

echo Environment variables set.
echo.
echo Starting server on http://localhost:8000
echo API docs: http://localhost:8000/api/docs
echo.
echo Press CTRL+C to stop the server.
echo.

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
