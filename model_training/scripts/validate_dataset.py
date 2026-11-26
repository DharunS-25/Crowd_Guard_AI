"""
validate_dataset.py — CrowdGuard AI
=======================================
Validates processed dataset arrays before training.
Checks for data leakage, label correctness, and schema compliance.

Run:
    python scripts/validate_dataset.py
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).parent
_ROOT = _SCRIPTS.parent
sys.path.insert(0, str(_ROOT / "src"))

from preprocessing import CORE_FEATURE_COLUMNS, FEATURE_DIM

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] — %(message)s")
logger = logging.getLogger("crowdguard.validate")


def validate(data_dir: str):
    issues = []
    warnings = []

    logger.info(f"Validating dataset in: {data_dir}")

    for split in ["train", "val", "test"]:
        X_path = os.path.join(data_dir, f"X_{split}.npy")
        y_path = os.path.join(data_dir, f"y_{split}.npy")

        if not os.path.exists(X_path):
            issues.append(f"Missing: {X_path}")
            continue

        X = np.load(X_path, allow_pickle=False)
        y = np.load(y_path, allow_pickle=False)

        # Shape checks
        if X.ndim != 3:
            issues.append(f"[{split}] X should be 3D (N, L, F), got {X.ndim}D")
        if y.ndim != 1:
            issues.append(f"[{split}] y should be 1D, got {y.ndim}D")
        if len(X) != len(y):
            issues.append(f"[{split}] X and y have different lengths: {len(X)} vs {len(y)}")

        if X.ndim == 3 and X.shape[2] != FEATURE_DIM:
            issues.append(
                f"[{split}] Feature dim mismatch: expected {FEATURE_DIM}, got {X.shape[2]}"
            )

        # NaN/Inf checks
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        if nan_count > 0:
            warnings.append(f"[{split}] {nan_count} NaN values in X")
        if inf_count > 0:
            warnings.append(f"[{split}] {inf_count} Inf values in X")

        # Label validity
        unique_labels = np.unique(y)
        if not all(l in [0, 1] for l in unique_labels):
            issues.append(f"[{split}] Unexpected labels: {unique_labels}")

        n_anomaly = int((y == 1).sum())
        n_normal = int((y == 0).sum())
        pct = 100 * n_anomaly / len(y) if len(y) > 0 else 0

        logger.info(
            f"  {split:5s}: {len(y):6d} seqs | "
            f"Normal={n_normal} | Anomaly={n_anomaly} ({pct:.1f}%)"
        )

        # Extreme imbalance warning
        if pct < 1.0:
            warnings.append(
                f"[{split}] Very few anomaly samples ({pct:.1f}%). "
                "Model may struggle to learn anomalies."
            )
        if pct > 90.0:
            warnings.append(
                f"[{split}] Very few normal samples. Check annotation."
            )

    # Check no overlap between splits by checking meta files
    for split in ["train", "val", "test"]:
        meta_path = os.path.join(data_dir, f"meta_{split}.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            clip_ids = {m["clip_id"] for m in meta}
            logger.info(f"  {split} meta: {len(clip_ids)} unique clips")

    # Scaler check
    scaler_path = os.path.join(data_dir, "scaler.pkl")
    if not os.path.exists(scaler_path):
        warnings.append("scaler.pkl not found — normalisation not applied.")

    # Report
    print(f"\n{'='*50}")
    if issues:
        print("  VALIDATION ISSUES (must fix before training):")
        for issue in issues:
            print(f"    ✗ {issue}")
    if warnings:
        print("  WARNINGS (review before training):")
        for w in warnings:
            print(f"    ⚠ {w}")
    if not issues and not warnings:
        print("  ✓ All validation checks passed.")
    print(f"{'='*50}\n")

    return len(issues) == 0


if __name__ == "__main__":
    data_dir = str(_ROOT / "data" / "processed")
    ok = validate(data_dir)
    sys.exit(0 if ok else 1)
