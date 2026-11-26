# DATASET CARD — CrowdGuard AI
## UCSD Anomaly Detection Dataset

---

## Dataset Identity

| Field | Value |
|---|---|
| **Name** | UCSD Anomaly Detection Dataset |
| **Source** | https://www.kaggle.com/datasets/orvile/ucsd-anomaly-dataset |
| **Original authors** | Vision-Based Analysis of Pedestrian Traffic — UCSD Computer Vision Lab |
| **License** | Creative Commons CC BY 4.0 |
| **Download size** | ~700 MB (compressed) |
| **Version used** | Kaggle mirror, version 1 (uploaded 2025-04-26) |

---

## Dataset Description

The UCSD Anomaly Detection Dataset was collected with a stationary camera mounted at an elevation, overlooking pedestrian walkways on the UCSD campus. It is a **benchmark dataset for video anomaly detection**, not a crowd-safety or crowd-density dataset.

### Scene Characteristics

| Property | Details |
|---|---|
| Camera type | Fixed, stationary, elevated |
| Scene | Outdoor pedestrian walkways |
| Crowd density | Variable — sparse to crowded |
| Frame resolution | 238×158 pixels |
| Frame format | TIFF (grayscale) |
| Approximate frame rate | 10 fps |
| Clip length | ~200 frames per clip (~20 seconds) |

### Subsets

| Subset | Train Clips | Test Clips | View |
|---|---|---|---|
| **Peds1** | 34 | 36 | Groups walking toward/away from camera with perspective distortion |
| **Peds2** | 16 | 12 | Pedestrians moving parallel to camera plane |
| **Total** | 50 | 48 | |

### Annotations

**Frame-level binary labels:**
- `0` = Normal frame (only pedestrian activity)
- `1` = Anomalous frame (non-pedestrian entity OR anomalous pedestrian movement)

**Pixel-level anomaly masks** (subset of clips):
- Peds1: 10 clips with pixel-level masks
- Peds2: 12 clips with pixel-level masks
- **Not used** in this training pipeline (frame-level labels only)

### Types of Anomalies

Naturally occurring events — **not staged**:
- Bicycles and cyclists
- Skateboarders / roller-skaters
- Small carts / trolleys
- Pedestrians walking in wrong direction
- Pedestrians walking on grass
- Wheelchair users (a few instances)

---

## How CrowdGuard AI Uses This Dataset

### Feature Extraction

From each video frame, 10 features are extracted:

```
X_t = [D, V, θ_sin, θ_cos, F_x, F_y, O, ΔD, σ_θ, σ_V]
```

- Crowd density (foreground pixel fraction)
- Optical flow magnitude (motion speed)
- Optical flow direction (circular encoding using sin/cos)
- Zone occupancy
- Density change rate
- Motion consistency (circular std of angles)

### Temporal Sequences

Sequences of L=20 consecutive (sub-sampled) frames are used as model input.  
Sampling interval: every 5th frame.

### Train/Val/Test Split

| Split | Source | Notes |
|---|---|---|
| Train | UCSD official train set (85%) | ~42 clips |
| Validation | UCSD official train set (15%) | ~8 clips |
| Test | UCSD official test set (100%) | 48 clips |

**All splits are at clip level — no adjacent-frame leakage between splits.**

---

## Critical Label Disclaimer

> ⚠️ **This dataset does NOT contain crowd-risk labels.**

The binary anomaly labels (Normal/Anomalous) reflect the **presence of non-pedestrian objects or movement patterns** on a university walkway. They do NOT represent:

- Validated crowd-safety risk levels (Low / Moderate / High / Critical)
- Expert safety assessments from venue operators
- Crowd density thresholds validated by crowd-safety engineers
- Risk assessments for concert halls, auditoriums, or event venues

**The model trained on this dataset is an experimental anomaly detection baseline.**  
It should be understood as a proof-of-concept for temporal pattern recognition, not a validated safety system.

### What Would Real Crowd-Risk Labels Require?

1. Multi-camera recordings from actual controlled event venues (auditoriums, concert halls, exit pathways)
2. Expert annotation by certified crowd-safety professionals
3. Validation across multiple independent event recordings
4. Defined risk thresholds based on density, flow, and venue capacity
5. Regulatory review if used for operational safety decisions

---

## Known Limitations

| Limitation | Impact |
|---|---|
| Single venue type (outdoor walkway) | Model may not generalise to indoor venues |
| No crowd accumulation scenarios | No data for concert entry/exit surges |
| Low resolution (238×158) | Feature extraction quality is limited |
| Static camera only | No pan/tilt/zoom variation |
| No audio information | Cannot correlate with crowd chanting, music, panic |
| No night-time footage | Lighting variation not represented |
| Small dataset (~19,600 frames) | Limited training data for deep learning |
| No person tracking | No individual trajectory information |
| University context only | Cultural behaviour may differ at events |

---

## Class Distribution (Approximate)

Training set (UCSD official): Predominantly normal clips with few anomaly frames.  
Test set: ~40-50% anomalous frames in annotated clips.

> **Class imbalance is expected.** The training pipeline uses class-weighted loss to compensate.

---

## Citation

If using this dataset, cite the original work:

```
Mahadevan, V., Li, W., Bhalodia, V., & Vasconcelos, N. (2010).
Anomaly detection in crowded scenes.
IEEE CVPR 2010.

Li, W., Mahadevan, V., & Vasconcelos, N. (2013).
Anomaly detection and localization in crowded scenes.
IEEE TPAMI, 36(1), 18–32.
```

---

## Download Instructions

```powershell
# Option 1: Manual
# Go to: https://www.kaggle.com/datasets/orvile/ucsd-anomaly-dataset
# Download → Extract to model_training/data/raw/ucsd/

# Option 2: Kaggle CLI
kaggle datasets download orvile/ucsd-anomaly-dataset -p data/raw/
Expand-Archive data/raw/ucsd-anomaly-dataset.zip -DestinationPath data/raw/ucsd/
```

**Expected structure after extraction:**
```
data/raw/ucsd/
  UCSDped1/
    Train/
      Train001/         ← .tif grayscale frames
      Train001_gt       ← frame-level labels (space-separated frame ranges)
      Train002/
      ...
    Test/
      Test001/
      Test001_gt
      ...
  UCSDped2/
    Train/...
    Test/...
```
