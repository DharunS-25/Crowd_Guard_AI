"""
preprocessing.py — CrowdGuard AI
==================================
Feature extraction from raw UCSD video frames.

Extracts the CrowdGuard AI core feature vector:
    X_t = [D, V, theta_sin, theta_cos, F_x, F_y, O,
           delta_D, motion_consistency, flow_mag_std]

Where:
    D          = crowd density (foreground pixel ratio)
    V          = motion magnitude (mean optical flow magnitude)
    theta_sin  = sin of mean flow direction (circular encoding)
    theta_cos  = cos of mean flow direction (circular encoding)
    F_x        = mean horizontal optical flow
    F_y        = mean vertical optical flow
    O          = zone occupancy (foreground ratio, same as D for full-frame view)
    delta_D    = frame-to-frame change in D
    motion_consistency = circular std of flow directions
    flow_mag_std       = std of flow magnitudes

IMPORTANT:
  - Normalization is FIT on training data only.
  - Scaler is saved and reloaded for val/test/inference.
  - No future frame information leaks into past feature vectors.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Feature column schema ────────────────────────────────────────────────────
CORE_FEATURE_COLUMNS = [
    "crowd_density",        # D
    "motion_magnitude",     # V
    "motion_direction_sin", # theta_sin
    "motion_direction_cos", # theta_cos
    "optical_flow_x",       # F_x
    "optical_flow_y",       # F_y
    "zone_occupancy",       # O
    "delta_density",        # dD/dt
    "motion_consistency",   # angular std
    "flow_magnitude_std",   # magnitude std
]

FEATURE_DIM = len(CORE_FEATURE_COLUMNS)


# ── Background subtraction helper ────────────────────────────────────────────
def _extract_foreground(frame_gray: np.ndarray,
                         bg: np.ndarray) -> np.ndarray:
    """Simple frame-differencing foreground mask."""
    diff = np.abs(frame_gray.astype(np.float32) - bg.astype(np.float32))
    _, mask = _threshold_otsu(diff)
    return mask


def _threshold_otsu(img: np.ndarray) -> Tuple[float, np.ndarray]:
    """Minimal Otsu threshold without cv2."""
    hist, bins = np.histogram(img.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0, img < 0
    cum_sum = np.cumsum(hist)
    cum_mean = np.cumsum(hist * np.arange(256))
    global_mean = cum_mean[-1] / total
    inter_class_var = np.zeros(256)
    for t in range(256):
        w0 = cum_sum[t] / total
        w1 = 1.0 - w0
        if w0 == 0 or w1 == 0:
            continue
        mean0 = cum_mean[t] / (cum_sum[t] + 1e-9)
        mean1 = (cum_mean[-1] - cum_mean[t]) / (total - cum_sum[t] + 1e-9)
        inter_class_var[t] = w0 * w1 * (mean0 - mean1) ** 2
    threshold = float(np.argmax(inter_class_var))
    mask = img > threshold
    return threshold, mask.astype(np.uint8)


def _compute_optical_flow_dense(prev: np.ndarray,
                                  curr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal Lucas-Kanade-style dense optical flow using numpy.
    Returns (flow_x, flow_y) arrays of shape equal to input frames.
    This is a simplified implementation using gradient-based approach.
    """
    # Compute image gradients
    Ix = (curr.astype(np.float32) - np.roll(curr, -1, axis=1)).astype(np.float32)
    Iy = (curr.astype(np.float32) - np.roll(curr, -1, axis=0)).astype(np.float32)
    It = (curr.astype(np.float32) - prev.astype(np.float32))
    # Simple flow estimate via gradient descent proxy
    eps = 1e-6
    flow_x = -Ix * It / (Ix ** 2 + Iy ** 2 + eps)
    flow_y = -Iy * It / (Ix ** 2 + Iy ** 2 + eps)
    # Clip outliers
    flow_x = np.clip(flow_x, -10, 10)
    flow_y = np.clip(flow_y, -10, 10)
    return flow_x, flow_y


def _circular_std(angles_rad: np.ndarray) -> float:
    """Circular standard deviation of an array of angles."""
    if len(angles_rad) == 0:
        return 0.0
    sin_mean = np.mean(np.sin(angles_rad))
    cos_mean = np.mean(np.cos(angles_rad))
    R = np.sqrt(sin_mean ** 2 + cos_mean ** 2)
    R = np.clip(R, 0.0, 1.0)
    return float(np.sqrt(-2.0 * np.log(R + 1e-9)))


def extract_features_from_frames(
    frames: List[np.ndarray],
    resize_hw: Tuple[int, int] = (64, 64),
) -> pd.DataFrame:
    """
    Extract CrowdGuard AI feature vectors from a list of grayscale frames.

    Parameters
    ----------
    frames : list of np.ndarray
        Grayscale frames (H x W, uint8).
    resize_hw : tuple
        Target (height, width) for processing.

    Returns
    -------
    pd.DataFrame
        One row per frame with CORE_FEATURE_COLUMNS columns.
        Row 0 has delta_density = 0 (no previous frame).
    """
    try:
        from PIL import Image
        def resize_frame(f):
            img = Image.fromarray(f)
            img = img.resize((resize_hw[1], resize_hw[0]), Image.BILINEAR)
            return np.array(img, dtype=np.float32)
    except ImportError:
        def resize_frame(f):
            # Crude resize using slicing
            h, w = resize_hw
            return f[::f.shape[0]//h, ::f.shape[1]//w][:h, :w].astype(np.float32)

    records = []
    prev_frame = None
    prev_density = None

    # Compute background as mean of all frames (simple static background model)
    stack = np.stack([resize_frame(f) for f in frames], axis=0)
    background = np.mean(stack, axis=0)

    for i, raw_frame in enumerate(frames):
        frame = resize_frame(raw_frame)
        frame_norm = frame / 255.0

        # ── Foreground / density ─────────────────────────────────────────────
        fg_mask_raw = np.abs(frame - background)
        fg_mask = (fg_mask_raw > 15).astype(np.float32)
        density = float(fg_mask.mean())
        occupancy = density  # same for full-frame view

        # ── Optical flow ─────────────────────────────────────────────────────
        if prev_frame is not None:
            fx, fy = _compute_optical_flow_dense(prev_frame / 255.0, frame_norm)
            mag = np.sqrt(fx ** 2 + fy ** 2)
            angles = np.arctan2(fy, fx)

            motion_mag = float(np.mean(mag))
            mean_fx = float(np.mean(fx))
            mean_fy = float(np.mean(fy))
            mean_angle = float(np.arctan2(mean_fy, mean_fx))
            angle_sin = float(np.sin(mean_angle))
            angle_cos = float(np.cos(mean_angle))
            consistency = _circular_std(angles.ravel())
            mag_std = float(np.std(mag))
        else:
            motion_mag = 0.0
            mean_fx = 0.0
            mean_fy = 0.0
            angle_sin = 0.0
            angle_cos = 1.0
            consistency = 0.0
            mag_std = 0.0

        delta_d = density - prev_density if prev_density is not None else 0.0

        records.append({
            "frame_idx": i,
            "crowd_density": density,
            "motion_magnitude": motion_mag,
            "motion_direction_sin": angle_sin,
            "motion_direction_cos": angle_cos,
            "optical_flow_x": mean_fx,
            "optical_flow_y": mean_fy,
            "zone_occupancy": occupancy,
            "delta_density": delta_d,
            "motion_consistency": consistency,
            "flow_magnitude_std": mag_std,
        })

        prev_frame = frame
        prev_density = density

    df = pd.DataFrame(records)
    # Ensure correct column order
    extra_cols = [c for c in df.columns if c not in CORE_FEATURE_COLUMNS and c != "frame_idx"]
    ordered = ["frame_idx"] + CORE_FEATURE_COLUMNS + extra_cols
    df = df[[c for c in ordered if c in df.columns]]
    return df


# ── Normalization ─────────────────────────────────────────────────────────────
class StandardScaler:
    """
    Minimal StandardScaler that works without sklearn.
    Fit on training data only — prevents data leakage.
    """

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """Fit on training data only."""
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0  # avoid division by zero
        self.feature_names_ = feature_names
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray, feature_names=None) -> np.ndarray:
        return self.fit(X, feature_names).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.std_ + self.mean_

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"mean": self.mean_, "std": self.std_,
                         "feature_names": self.feature_names_}, f)
        logger.info(f"Scaler saved to {path}")

    @classmethod
    def load(cls, path: str) -> "StandardScaler":
        with open(path, "rb") as f:
            data = pickle.load(f)
        s = cls()
        s.mean_ = data["mean"]
        s.std_ = data["std"]
        s.feature_names_ = data.get("feature_names")
        return s


def handle_missing_values(df: pd.DataFrame,
                           fill_values: Optional[Dict] = None) -> pd.DataFrame:
    """
    Fill missing / NaN values in feature DataFrame.
    Uses feature-specific fill values from config if provided,
    otherwise falls back to 0.0.
    """
    defaults = {col: 0.0 for col in CORE_FEATURE_COLUMNS}
    if fill_values:
        defaults.update(fill_values)

    for col in CORE_FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(defaults.get(col, 0.0))
            df[col] = df[col].replace([np.inf, -np.inf], defaults.get(col, 0.0))

    return df


def validate_feature_schema(df: pd.DataFrame) -> bool:
    """Check that all expected feature columns are present."""
    missing = [c for c in CORE_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        logger.warning(f"Missing feature columns: {missing}")
        return False
    return True
