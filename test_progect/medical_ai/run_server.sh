#!/bin/bash
# Run with: bash run_server.sh

echo "================================================"
echo "  MedicalAI - Starting API Server"
echo "================================================"
echo ""

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate medicalai

export DEV_TOKEN=my-test-token-123
export MODEL_PATH=outputs/models/calibrated_model.pt
export ICD_VOCAB_PATH=data/icd10_vocab.json
export GUIDELINE_DIR=data/guidelines

echo "Environment variables set."
echo ""
echo "Starting server on http://localhost:8000"
echo "API docs: http://localhost:8000/api/docs"
echo ""
echo "Press CTRL+C to stop the server."
echo ""

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
