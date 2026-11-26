# MODEL TRAINING AUDIT — CrowdGuard AI
## Temporal Model Training Pipeline Audit Report

**Project:** CrowdGuard AI — Multi-Camera Spatial-Temporal Crowd Risk Estimation  
**Scope:** GRU and LSTM temporal model training pipeline  
**Date:** 2026-10-27  
**Auditor:** Senior Deep Learning Engineer (AI Assistant)

---

## 1. ENVIRONMENT AUDIT

| Component | Status | Details |
|---|---|---|
| Python | ✅ Available | Python 3.14.2 |
| OS | ✅ | Windows 11 (10.0.26300) |
| PyTorch | ✅ Installed | 2.14.1+cpu (CPU only) |
| CUDA / GPU | ⚠️ **NOT AVAILABLE** | CPU-only build installed |
| NumPy | ✅ | 2.5.1 |
| scikit-learn | ✅ | 1.9.0 |
| SciPy | ✅ | 1.18.0 |
| Pandas | ✅ Installed | 3.1.0 |
| Matplotlib | ✅ Installed | 3.11.2 |
| Seaborn | ✅ Installed | 0.13.2 |
| PyYAML | ✅ Installed | 6.0.3 |
| Pillow | ✅ Installed | 12.3.0 |
| OpenCV | ✅ Installed | 5.0.0.93 (headless) |
| tqdm | ✅ Installed | 4.70.1 |
| pytest | ✅ | 9.1.1 |
| RAM | ⚠️ Unknown | psutil not installed |

> **GPU Warning:** No CUDA-capable GPU detected. All training runs on CPU.
> Training will be slower but fully functional. Mixed precision is not used.

---

## 2. EXISTING PROJECT AUDIT

| Item | Status | Finding |
|---|---|---|
| Project directory | ⚠️ Empty | No pre-existing code found |
| Feature CSV/Parquet files | ❌ None | Must extract from raw video |
| Existing model checkpoints | ❌ None | First-time training |
| Existing GRU/LSTM implementations | ❌ None | Implemented from scratch |
| Crowd-risk labels | ❌ None | Only anomaly labels available (UCSD) |
| Preprocessing pipeline | ❌ None | Implemented from scratch |
| Dataset | ❌ Not downloaded | Must download from Kaggle |

---

## 3. DATASET AUDIT

### 3.1 Dataset Identity

**Name:** UCSD Anomaly Detection Dataset  
**Source:** https://www.kaggle.com/datasets/orvile/ucsd-anomaly-dataset  
**License:** CC BY 4.0  
**Download size:** ~700 MB (compressed)

### 3.2 Dataset Description

The UCSD dataset contains videos of pedestrian walkways filmed from a stationary elevated camera.

| Subset | Train Clips | Test Clips | Description |
|---|---|---|---|
| Peds1 | 34 | 36 | Groups walking toward/away from camera, perspective distortion |
| Peds2 | 16 | 12 | Pedestrians moving parallel to camera plane |

**Total clips:** 98 (50 train, 48 test)  
**Frames per clip:** ~200  
**Total frames:** ~19,600  
**Frame format:** `.tif` (grayscale)  
**Annotation format:** Frame-level binary flags (anomaly / normal)

### 3.3 Label Schema — CRITICAL DISCLAIMER

> ⚠️ **IMPORTANT:** The UCSD dataset provides **binary anomaly detection labels**, NOT validated crowd-risk severity ratings.

| Label | Meaning (UCSD) | CrowdGuard interpretation |
|---|---|---|
| `0` | Normal pedestrian activity | Normal — no anomaly |
| `1` | Anomalous event (biker, skater, cart, wrong-direction walker) | Anomalous — elevated attention warranted |

**What these labels represent:**
- Presence of non-pedestrian entities (bicycles, skateboards, carts)
- Anomalous pedestrian movement patterns (wrong direction, grass walking)

**What these labels do NOT represent:**
- Validated crowd-risk severity levels (Low / Moderate / High / Critical)
- Expert-annotated safety assessments from venue operators
- Real-world crowd crush or stampede risk

**Task type:** Binary classification (Normal=0, Anomalous=1)  
**NOT:** Four-class crowd risk estimation

This implementation therefore trains an **anomaly detection model** as an experimental baseline for crowd-risk estimation. Converting to full crowd-risk estimation requires:
1. Venue-specific recordings with expert risk annotations
2. Independent validation across multiple event types
3. Collaboration with crowd safety professionals

### 3.4 Expected Class Distribution

Based on UCSD literature:
- **Peds1:** ~40–50% anomalous frames in test set
- **Peds2:** ~40–50% anomalous frames in test set
- Train clips are predominantly normal (few anomaly clips)

> Training data is likely **class-imbalanced** (more normal than anomalous). Class-weighted loss is applied.

---

## 4. MISSING REQUIREMENTS

### Critical Missing Items

| Item | Impact | Mitigation |
|---|---|---|
| **GPU** | Training is slow on CPU | Accepted — reduce hidden_dim if needed |
| **Real crowd-risk labels** | Cannot train 4-class risk model | Train binary anomaly detector instead |
| **Kaggle API token** | Cannot auto-download dataset | Must download manually |
| **Multiple venue recordings** | Cannot validate generalisation | Document as limitation |

### How to Download the Dataset

**Option 1: Kaggle CLI (requires Kaggle account)**
```powershell
# Place kaggle.json in C:\Users\Dharu\.kaggle\
kaggle datasets download orvile/ucsd-anomaly-dataset
Expand-Archive ucsd-anomaly-dataset.zip -DestinationPath model_training/data/raw/ucsd/
```

**Option 2: Manual download**
1. Go to https://www.kaggle.com/datasets/orvile/ucsd-anomaly-dataset
2. Click Download (requires login)
3. Extract to: `model_training/data/raw/ucsd/`

**Expected directory structure after extraction:**
```
model_training/data/raw/ucsd/
  UCSDped1/
    Train/
      Train001/    ← .tif frame images
      Train001_gt  ← frame-level anomaly labels
      Train002/
      Train002_gt
      ...
    Test/
      Test001/
      Test001_gt
      ...
  UCSDped2/
    Train/...
    Test/...
```

---

## 5. FEATURE EXTRACTION AUDIT

### 5.1 CrowdGuard AI Feature Vector

```
X_t = [D, V, θ_sin, θ_cos, F_x, F_y, O, ΔD, σ_θ, σ_V]
```

| Feature | Symbol | Method | Notes |
|---|---|---|---|
| Crowd density | D | Foreground pixel ratio (frame differencing) | Background = clip mean |
| Motion magnitude | V | Mean optical flow magnitude | Gradient-based approximation |
| Direction (sin) | θ_sin | sin(mean flow angle) | Circular encoding |
| Direction (cos) | θ_cos | cos(mean flow angle) | Circular encoding |
| Flow X | F_x | Mean horizontal optical flow | Proxy for lateral movement |
| Flow Y | F_y | Mean vertical optical flow | Proxy for fore/aft movement |
| Zone occupancy | O | Same as D for full-frame view | Alias |
| Density change | ΔD | Frame-to-frame density difference | Accumulation indicator |
| Direction consistency | σ_θ | Circular std of flow angles | Low = unified direction |
| Flow magnitude std | σ_V | Std dev of magnitude over frame | High = chaotic motion |

**Total feature dimension:** 10

### 5.2 Angular Encoding Rationale

Crowd movement direction (`theta`) uses `(sin(θ), cos(θ))` circular encoding to avoid discontinuities at ±180°. This prevents the model from treating 179° and -179° as maximally different when they are adjacent.

### 5.3 Limitations of Feature Extraction

- **Optical flow:** Simple gradient-based implementation (not Farneback or Lucas-Kanade). Sufficient for relative motion estimation; may not be production-quality.
- **Background model:** Clip-mean background (static). Fails under dynamic lighting.
- **No YOLO/ByteTrack:** Person count, trajectory data, and track IDs are not available. The pipeline should be updated when these are integrated.

---

## 6. TEMPORAL MODEL AUDIT

### 6.1 Prediction Target

**Type:** Current-state estimation  
**Window:** Past L frames → risk at time t

Given features `[X(t-L+1), ..., X(t)]`, predict label at `t`.  
This is **not** multi-step future prediction.

### 6.2 Sequence Length

Default: **L = 20 frames** with **sampling_interval = 5** (every 5th video frame)  
Effective temporal window: 20 × 5 = **100 video frames** ≈ 3–4 seconds at 30 fps

### 6.3 Data Split Strategy

- Splits are at **clip level**, not frame level
- UCSD pre-defined test set → used as test split
- UCSD train set → 85% train, 15% val (clip-level)
- Adjacent frames from the same clip NEVER appear in different splits

### 6.4 GRU Model Summary

| Parameter | Value |
|---|---|
| Input dim | 10 (feature vector) |
| Hidden dim | 64 |
| Layers | 2 |
| Dropout | 0.3 |
| FC hidden | 32 |
| Output | 2 classes |
| Bidirectional | No (causal) |
| Initialisation | Xavier + Orthogonal |

### 6.5 LSTM Model Summary

| Parameter | Value |
|---|---|
| Input dim | 10 |
| Hidden dim | 64 |
| Layers | 2 |
| Dropout | 0.3 |
| FC hidden | 32 |
| Output | 2 classes |
| Forget gate bias | 1.0 (standard trick) |

> LSTM has approximately 33% more parameters than equivalent GRU due to the additional cell state gate.

---

## 7. TRAINING CONFIGURATION AUDIT

| Setting | Value | Justification |
|---|---|---|
| Optimiser | AdamW | Better weight decay than Adam |
| Learning rate | 0.001 | Standard for RNN tasks |
| Weight decay | 1e-4 | Regularisation |
| Loss | CrossEntropyLoss + class weights | Handles class imbalance |
| Grad clip | 1.0 | Prevents RNN gradient explosion |
| Early stopping | patience=15, monitor=val_f1 | F1 is better than accuracy for imbalanced data |
| LR scheduler | ReduceLROnPlateau | Adaptive LR reduction |
| Max epochs | 80 | Sufficient with early stopping |
| Batch size | 32 | Memory-efficient for CPU |

---

## 8. EVALUATION AUDIT

### 8.1 Primary Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Overall correctness |
| Macro F1 | Balanced performance across classes |
| Anomaly Recall | **Safety-critical**: fraction of real anomalies detected |
| False Negative Rate | Missed anomaly rate (lower = safer) |
| ROC-AUC | Discrimination ability (threshold-free) |

### 8.2 Safety-Oriented Evaluation Note

For crowd safety applications, **Anomaly Recall is the primary safety metric**.

A missed anomaly (False Negative) is more dangerous than a false alarm (False Positive).  
The model comparison weights Recall(anomaly) at 50%, F1 at 30%, AUC at 20%.

### 8.3 Generalization Limitations

> ⚠️ **Single dataset, single venue type:** Results from the UCSD dataset do NOT generalise to controlled event venues (concert halls, auditoriums, etc.) without additional validation.

Known limitations:
- UCSD shows pedestrian walkways, not event venues
- No crowd density variation from gatherings/dispersals
- No audio-visual event correlation
- No multi-camera synchronisation
- Single stationary camera per scene

---

## 9. FILES CREATED

```
model_training/
  configs/
    dataset.yaml          ← Dataset and feature configuration
    gru.yaml              ← GRU model and training configuration
    lstm.yaml             ← LSTM model and training configuration
  src/
    preprocessing.py      ← Feature extraction, scaler
    sequence_builder.py   ← Sliding window + clip-level splits
    dataset.py            ← PyTorch Dataset and DataLoader
    gru_model.py          ← CrowdGuardGRU class
    lstm_model.py         ← CrowdGuardLSTM class
    train_engine.py       ← Shared training loop
    train_gru.py          ← GRU training entry point
    train_lstm.py         ← LSTM training entry point
    evaluate.py           ← Test set evaluation + plots
    compare_models.py     ← GRU vs LSTM comparison
    inference.py          ← Unified inference interface
    metrics.py            ← Evaluation metrics
    reproducibility.py    ← Seed management
  scripts/
    inspect_dataset.py    ← Feature extraction pipeline
    validate_dataset.py   ← Data validation
    run_smoke_test.py     ← Smoke tests (no dataset needed)
    train_both.py         ← Master pipeline runner
    evaluate_both.py      ← Evaluate both models
  tests/
    test_all.py           ← Pytest test suite
  requirements.txt
  MODEL_TRAINING_AUDIT.md (this file)
  DATASET_CARD.md
  MODEL_REPORT.md
  README.md
```

---

## 10. EXECUTION ORDER

1. `python scripts/run_smoke_test.py` — Verify pipeline works (no dataset needed)
2. Download dataset to `data/raw/ucsd/`
3. `python scripts/inspect_dataset.py` — Extract features and build splits
4. `python scripts/validate_dataset.py` — Validate processed data
5. `python scripts/train_both.py` — Train GRU + LSTM + Evaluate + Compare

Or step by step:
```
python src/train_gru.py
python src/train_lstm.py
python src/evaluate.py --model-type gru
python src/evaluate.py --model-type lstm
python src/compare_models.py
```

---

## 11. WHAT IS NOT IMPLEMENTED (Out of Scope)

Per project requirements, the following were intentionally excluded:
- YOLO person detection
- ByteTrack multi-object tracking
- Frontend / web application
- Real-time video pipeline
- Multi-camera fusion
- Audio anomaly detection
- Four-class crowd risk classification (labels unavailable)

These require integration with the other CrowdGuard AI components.

---

## 12. VERDICT

| Aspect | Status |
|---|---|
| Pipeline implemented | ✅ Complete |
| Training code | ✅ Complete |
| Evaluation code | ✅ Complete |
| Smoke tests passing | ✅ (synthetic data) |
| Real dataset downloaded | ❌ **Must download manually** |
| GPU available | ❌ CPU-only (acceptable for dataset size) |
| Real crowd-risk labels | ❌ Not in UCSD dataset |
| Production ready | ❌ Experimental baseline only |

> Training results cannot be reported until the dataset is downloaded and the pipeline is executed. No results are fabricated in this audit.
