# MedicalAI — Disease Prediction System

An AI that predicts diseases from patient history and recommends tests.

## Quick Start

### Step 1 — Windows setup
Double-click `setup.bat`

### Step 1 — Mac / Linux setup
```bash
bash setup.sh
```

### Step 2 — Edit config for quick test
Open configs/config.yaml, change:
- epochs: 3
- batch_size: 8
- num_diseases: 8
- freeze_layers: 11

### Step 3 — Train
```
Windows: double-click run_training.bat
Mac/Linux: bash run_training.sh
```

### Step 4 — Start API server
```
Windows: double-click run_server.bat
Mac/Linux: bash run_server.sh
```

### Step 5 — Test
Open browser: http://localhost:8000/api/docs

Or run: python test_api.py

## Common commands
```bash
conda activate medicalai
python data/create_synthetic.py     # create test data
python scripts/train.py --config configs/config.yaml
python test_api.py
```

## Disclaimer
AI clinical decision support only. Not a diagnosis. Always consult a qualified doctor.
