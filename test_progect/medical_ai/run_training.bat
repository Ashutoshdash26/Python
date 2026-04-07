@echo off
echo ================================================
echo   MedicalAI - Training the Model
echo ================================================
echo.

call conda activate medicalai

echo Running training with config: configs\config.yaml
echo This will take several minutes. Do not close this window.
echo.

python scripts\train.py --config configs\config.yaml

echo.
echo Training complete!
echo Model saved to: outputs\models\calibrated_model.pt
echo.
pause
