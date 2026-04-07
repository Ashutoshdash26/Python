#!/bin/bash
# ================================================
#   MedicalAI - Mac / Linux Setup Script
#   Run with: bash setup.sh
# ================================================

set -e  # Stop on any error

echo "================================================"
echo "  MedicalAI - Setup Script"
echo "================================================"
echo ""

# Check conda exists
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found."
    echo "Please install Anaconda from: https://www.anaconda.com/download"
    exit 1
fi

echo "Step 1: Creating conda environment..."
conda create -n medicalai python=3.11 -y

echo ""
echo "Step 2: Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate medicalai

echo ""
echo "Step 3: Installing all libraries (10-20 minutes)..."
pip install -r requirements.txt

echo ""
echo "Step 4: Creating folder structure..."
mkdir -p data/raw data/processed data/guidelines
mkdir -p outputs/models outputs/logs outputs/onnx

echo ""
echo "Step 5: Creating synthetic test data..."
python data/create_synthetic.py

echo ""
echo "================================================"
echo "  Setup complete!"
echo "================================================"
echo ""
echo "To edit config for a quick test run:"
echo "  Open configs/config.yaml and change:"
echo "  - epochs: 50  →  epochs: 3"
echo "  - batch_size: 32  →  batch_size: 8"
echo "  - num_diseases: 1000  →  num_diseases: 8"
echo ""
echo "To train the model:"
echo "  conda activate medicalai"
echo "  python scripts/train.py --config configs/config.yaml"
echo ""
echo "To start the API server:"
echo "  conda activate medicalai"
echo "  export DEV_TOKEN=my-test-token-123"
echo "  export MODEL_PATH=outputs/models/calibrated_model.pt"
echo "  export ICD_VOCAB_PATH=data/icd10_vocab.json"
echo "  export GUIDELINE_DIR=data/guidelines"
echo "  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
