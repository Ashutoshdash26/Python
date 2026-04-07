@echo off
echo ================================================
echo   MedicalAI - Windows Setup Script
echo ================================================
echo.

echo Step 1: Creating conda environment...
call conda create -n medicalai python=3.11 -y
if errorlevel 1 (
    echo ERROR: conda not found. Please install Anaconda first.
    echo Download from: https://www.anaconda.com/download
    pause
    exit /b 1
)

echo.
echo Step 2: Activating environment...
call conda activate medicalai

echo.
echo Step 3: Installing all libraries (this takes 10-20 minutes)...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed. Check your internet connection.
    pause
    exit /b 1
)

echo.
echo Step 4: Creating data folders...
if not exist "data\raw"        mkdir data\raw
if not exist "data\processed"  mkdir data\processed
if not exist "data\guidelines" mkdir data\guidelines
if not exist "outputs\models"  mkdir outputs\models
if not exist "outputs\logs"    mkdir outputs\logs
if not exist "outputs\onnx"    mkdir outputs\onnx

echo.
echo Step 5: Creating synthetic test data...
python data\create_synthetic.py

echo.
echo ================================================
echo   Setup complete!
echo ================================================
echo.
echo To train the model, run:
echo   conda activate medicalai
echo   python scripts\train.py --config configs\config.yaml
echo.
echo To start the API server, run:
echo   conda activate medicalai
echo   set DEV_TOKEN=my-test-token-123
echo   set MODEL_PATH=outputs\models\calibrated_model.pt
echo   set ICD_VOCAB_PATH=data\icd10_vocab.json
echo   set GUIDELINE_DIR=data\guidelines
echo   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
echo.
pause
