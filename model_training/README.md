# CrowdGuard AI — Model Training Pipeline
## GRU & LSTM Temporal Models for Crowd Risk Estimation

---

## Overview

This module implements two independent temporal deep learning models:

| Model | Architecture | Task |
|---|---|---|
| **CrowdGuardGRU** | 2-layer GRU + FC head | Binary anomaly classification |
| **CrowdGuardLSTM** | 2-layer LSTM + FC head | Binary anomaly classification |

Both models are trained on the **UCSD Anomaly Detection Dataset** to detect anomalous crowd behaviour as an experimental baseline for crowd-risk estimation in controlled event venues.

> ⚠️ **IMPORTANT:** These are **experimental anomaly detection models**, not validated crowd-risk classifiers. The UCSD dataset provides binary anomaly labels (normal vs. anomalous pedestrian activity), not expert-annotated crowd-risk severity levels. Do not use outputs for safety-critical decisions without independent venue validation.

---

## Quick Start

### 1. Install Dependencies
```powershell
cd model_training
pip install -r requirements.txt
```

### 2. Run Smoke Tests (No Dataset Required)
```powershell
python scripts/run_smoke_test.py
```
Expected output: `Results: 10 passed, 0 failed`

### 3. Download the UCSD Dataset

**Manual download:**
1. Visit https://www.kaggle.com/datasets/orvile/ucsd-anomaly-dataset
2. Download the ZIP file (requires free Kaggle account)
3. Extract to `data/raw/ucsd/`

**Via Kaggle CLI:**
```powershell
# Requires kaggle.json API key in C:\Users\<user>\.kaggle\
kaggle datasets download orvile/ucsd-anomaly-dataset -p data/raw/
Expand-Archive data/raw/ucsd-anomaly-dataset.zip -DestinationPath data/raw/ucsd/
```

### 4. Extract Features & Build Dataset
```powershell
python scripts/inspect_dataset.py
python scripts/validate_dataset.py
```

### 5. Train Both Models + Evaluate + Compare
```powershell
python scripts/train_both.py
```

Or step by step:
```powershell
python src/train_gru.py
python src/train_lstm.py
python src/evaluate.py --model-type gru
python src/evaluate.py --model-type lstm
python src/compare_models.py
```

---

## Project Structure

```
model_training/
├── configs/
│   ├── dataset.yaml        # Dataset & feature configuration
│   ├── gru.yaml            # GRU architecture & training config
│   └── lstm.yaml           # LSTM architecture & training config
├── src/
│   ├── preprocessing.py    # Feature extraction & normalisation
│   ├── sequence_builder.py # Temporal windows & clip-level splits
│   ├── dataset.py          # PyTorch Dataset & DataLoader
│   ├── gru_model.py        # CrowdGuardGRU model class
│   ├── lstm_model.py       # CrowdGuardLSTM model class
│   ├── train_engine.py     # Shared training loop (AdamW, EarlyStopping, etc.)
│   ├── train_gru.py        # GRU training entry point
│   ├── train_lstm.py       # LSTM training entry point
│   ├── evaluate.py         # Evaluation + plots
│   ├── compare_models.py   # GRU vs LSTM comparison
│   ├── inference.py        # Shared inference interface
│   ├── metrics.py          # Evaluation metrics
│   └── reproducibility.py  # Seed management
├── scripts/
│   ├── inspect_dataset.py  # Feature extraction pipeline
│   ├── validate_dataset.py # Data validation
│   ├── run_smoke_test.py   # Smoke tests (no dataset needed)
│   ├── train_both.py       # Master pipeline
│   └── evaluate_both.py    # Evaluate both models
├── tests/
│   └── test_all.py         # Pytest test suite
├── outputs/
│   ├── checkpoints/        # Saved model checkpoints
│   ├── metrics/            # JSON metrics files
│   ├── plots/              # Training curves, confusion matrices
│   └── logs/               # Training logs
├── data/
│   ├── raw/ucsd/           # Raw UCSD dataset (download here)
│   └── processed/          # Extracted features & split arrays
├── MODEL_TRAINING_AUDIT.md
├── DATASET_CARD.md
├── MODEL_REPORT.md
└── requirements.txt
```

---

## Model Architecture

### Input Features (10 dimensions)

```
X_t = [D, V, θ_sin, θ_cos, F_x, F_y, O, ΔD, σ_θ, σ_V]
```

| Feature | Symbol | Description |
|---|---|---|
| Crowd density | D | Foreground pixel fraction |
| Motion magnitude | V | Mean optical flow magnitude |
| Direction (sin) | θ_sin | Circular-encoded flow direction |
| Direction (cos) | θ_cos | Circular-encoded flow direction |
| Flow X | F_x | Mean horizontal optical flow |
| Flow Y | F_y | Mean vertical optical flow |
| Zone occupancy | O | Foreground pixel ratio (zone-level) |
| Density change | ΔD | Frame-to-frame density difference |
| Direction consistency | σ_θ | Circular std of flow directions |
| Flow std | σ_V | Std dev of flow magnitudes |

### Sequence Input
```
X = [X(t-L+1), ..., X(t)]   where L = 20 time steps
```

### GRU Architecture
```
Input (batch, 20, 10) → GRU(64 units, 2 layers) → Dropout(0.3)
  → FC(32) → ReLU → Dropout(0.3) → Output(2 logits)
```

### LSTM Architecture
```
Input (batch, 20, 10) → LSTM(64 units, 2 layers) → Dropout(0.3)
  → FC(32) → ReLU → Dropout(0.3) → Output(2 logits)
```

---

## Training Configuration

| Setting | Value |
|---|---|
| Optimiser | AdamW (lr=1e-3, wd=1e-4) |
| Loss | CrossEntropyLoss + class weights |
| Gradient clip | 1.0 |
| Early stopping | patience=15, monitor=val_F1 |
| LR scheduler | ReduceLROnPlateau |
| Max epochs | 80 |
| Batch size | 32 |
| Seed | 42 |

---

## Inference Usage

```python
from src.inference import CrowdGuardPredictor
import numpy as np

# Load trained GRU model
predictor = CrowdGuardPredictor(model_type="gru")

# Prepare feature sequence: shape (seq_len, n_features)
sequence = np.array(...)  # your 20×10 feature matrix

# Run inference
result = predictor.predict(
    sequence,
    camera_id="cam01",
    zone_id="main_entrance"
)

print(result)
# {
#   "status": "experimental",
#   "prediction": {
#     "class": 1,
#     "label": "Anomalous",
#     "anomaly_probability": 0.87,
#     "confidence": 0.87
#   },
#   "data_quality": { "ok": true, "warnings": [] },
#   "camera_id": "cam01",
#   "zone_id": "main_entrance"
# }
```

---

## Outputs

After running `scripts/train_both.py`:

| File | Description |
|---|---|
| `outputs/checkpoints/gru/best_model.pt` | Best GRU checkpoint |
| `outputs/checkpoints/lstm/best_model.pt` | Best LSTM checkpoint |
| `outputs/metrics/gru_metrics.json` | GRU test evaluation metrics |
| `outputs/metrics/lstm_metrics.json` | LSTM test evaluation metrics |
| `outputs/metrics/model_comparison.csv` | Side-by-side comparison table |
| `outputs/plots/gru_training_curves.png` | GRU loss/F1 curves |
| `outputs/plots/lstm_training_curves.png` | LSTM loss/F1 curves |
| `outputs/plots/gru_confusion_matrix.png` | GRU confusion matrix |
| `outputs/plots/lstm_confusion_matrix.png` | LSTM confusion matrix |
| `outputs/plots/gru_risk_timeline.png` | Predicted vs actual timeline |
| `outputs/plots/lstm_risk_timeline.png` | Predicted vs actual timeline |
| `outputs/plots/gru_vs_lstm_comparison.png` | Comparison bar chart |

---

## Running Tests

```powershell
# Smoke tests (no dataset required)
python scripts/run_smoke_test.py

# Full pytest suite
pytest tests/test_all.py -v
```

---

## Environment

| Component | Version |
|---|---|
| Python | 3.14.2 |
| PyTorch | 2.14.1+cpu |
| NumPy | 2.5.1 |
| Pandas | 3.1.0 |
| scikit-learn | 1.9.0 |
| Matplotlib | 3.11.2 |

> **Note:** No CUDA GPU available. Training runs on CPU. For larger experiments, use a machine with a CUDA-capable GPU.

---

## Limitations

1. **Single dataset, single venue type:** UCSD shows outdoor pedestrian walkways, not event venues.
2. **Binary labels only:** Cannot train 4-class crowd risk model without expert annotation.
3. **No GPU:** Training is slower but functionally correct.
4. **Simplified optical flow:** Uses gradient-based approximation, not Farneback/LK.
5. **No person tracking:** No YOLO/ByteTrack integration — features are frame-level only.
6. **No real-world validation:** Model has not been tested at actual concert halls, auditoriums, etc.

See `MODEL_TRAINING_AUDIT.md` for the full audit and `MODEL_REPORT.md` for results.
