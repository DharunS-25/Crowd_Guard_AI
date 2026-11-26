"""
sequence_builder.py — CrowdGuard AI
=====================================
Builds sliding-window temporal sequences from per-clip feature DataFrames.

Key design decisions:
  - Sequences NEVER cross clip boundaries (no leakage across sessions).
  - Clip-level train/val/test splits prevent adjacent-frame leakage.
  - Sequence length (L) and stride are configurable.
  - Labels are taken at the END of each window (current-state estimation).
  - Short clips (< L frames) are discarded with a warning.

Prediction target: CURRENT-STATE ESTIMATION
    Given features from t-L+1 to t, predict the risk class at time t.
    (Not multi-step future prediction.)
"""

import json
import logging
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ClipMeta:
    """Metadata for one video clip."""
    clip_id: str          # e.g. "Peds1_Train001"
    subset: str           # "Peds1" or "Peds2"
    split: str            # "train", "val", or "test"
    n_frames: int
    n_sequences: int
    has_anomaly: bool


# ── Sequence Builder ──────────────────────────────────────────────────────────
class SequenceBuilder:
    """
    Converts per-clip feature DataFrames into fixed-length temporal windows.

    Parameters
    ----------
    seq_len : int
        Number of time steps per input window.
    stride : int
        Sliding window stride (frames to advance per step).
    feature_cols : list of str
        Ordered list of feature column names.
    label_col : str
        Name of the label column.
    """

    def __init__(
        self,
        seq_len: int = 20,
        stride: int = 1,
        feature_cols: Optional[List[str]] = None,
        label_col: str = "label",
    ):
        self.seq_len = seq_len
        self.stride = stride
        self.label_col = label_col

        from preprocessing import CORE_FEATURE_COLUMNS
        self.feature_cols = feature_cols or CORE_FEATURE_COLUMNS

    def build_from_clip(
        self,
        df: pd.DataFrame,
        clip_id: str,
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """
        Build sequences from a single clip DataFrame.

        Returns
        -------
        X : np.ndarray of shape (N, seq_len, n_features)
        y : np.ndarray of shape (N,)
        meta : list of dicts with per-sequence metadata
        """
        if len(df) < self.seq_len:
            logger.warning(
                f"Clip '{clip_id}' has {len(df)} frames < seq_len={self.seq_len}. "
                "Skipping."
            )
            return np.empty((0, self.seq_len, len(self.feature_cols))), np.empty((0,)), []

        features = df[self.feature_cols].values.astype(np.float32)
        labels = df[self.label_col].values.astype(np.int64)

        X_list, y_list, meta_list = [], [], []
        n = len(features)

        for start in range(0, n - self.seq_len + 1, self.stride):
            end = start + self.seq_len
            window = features[start:end]
            label = int(labels[end - 1])   # label at END of window

            # Skip windows with too many NaN-filled frames (>50%)
            if np.isnan(window).mean() > 0.5:
                continue

            # Replace any remaining NaN with 0
            window = np.nan_to_num(window, nan=0.0)

            X_list.append(window)
            y_list.append(label)
            meta_list.append({
                "clip_id": clip_id,
                "start_frame": start,
                "end_frame": end - 1,
                "label": label,
            })

        if not X_list:
            return np.empty((0, self.seq_len, len(self.feature_cols))), np.empty((0,)), []

        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int64)
        return X, y, meta_list


# ── Dataset Split Manager ─────────────────────────────────────────────────────
class DatasetSplitter:
    """
    Performs clip-level train/val/test splits ensuring both anomalous and
    normal clips appear in every split (no single-class splits).

    Strategy — "pooled stratified" split:
      All clips are pooled together regardless of the UCSD official train/test
      designation, then split into train/val/test with stratification by
      whether the clip contains ANY anomalous frame. This guarantees that the
      training set includes annotated anomaly examples, which is required for
      supervised classification.

    Why override the official UCSD split?
      The UCSD Train partition contains ONLY normal clips by design (anomaly
      detection benchmark convention). A model trained exclusively on normal
      clips cannot learn a supervised anomaly classifier — it would predict
      only class 0 and appear trivially accurate. This implementation pools
      ALL clips and performs a stratified split instead.

    NOTE: This means our "test" split is NOT the official UCSD test benchmark.
    Results are therefore NOT comparable to published UCSD benchmark numbers.
    They represent internal evaluation on our own held-out clips only.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def split_clips(
        self,
        all_clips: List[dict],
        val_fraction: float = 0.15,
        test_fraction: float = 0.20,
    ) -> Dict[str, List[dict]]:
        """
        Stratified clip-level split into train / val / test.

        Clips are stratified by whether they have a GT annotation file
        (proxy for containing anomalies). This ensures anomalous clips
        appear in all three splits.

        Parameters
        ----------
        all_clips : list of clip metadata dicts (from discover_clips)
        val_fraction : fraction of all clips for validation
        test_fraction : fraction of all clips for test

        Returns
        -------
        dict with keys 'train', 'val', 'test'
        """
        rng = random.Random(self.seed)

        # Stratify by whether clip has a GT file (anomaly indicator)
        anomaly_clips = [c for c in all_clips if c.get("gt_path")]
        normal_clips  = [c for c in all_clips if not c.get("gt_path")]

        rng.shuffle(anomaly_clips)
        rng.shuffle(normal_clips)

        def _do_split(clips, val_frac, test_frac):
            n = len(clips)
            n_test = max(1, int(n * test_frac))
            n_val  = max(1, int(n * val_frac))
            n_train = n - n_val - n_test
            if n_train < 1:
                # Not enough clips for all splits; put at least 1 in train
                n_train = max(1, n - 2)
                n_val   = max(0, (n - n_train) // 2)
                n_test  = n - n_train - n_val
            test  = clips[:n_test]
            val   = clips[n_test:n_test + n_val]
            train = clips[n_test + n_val:]
            return train, val, test

        a_train, a_val, a_test = _do_split(anomaly_clips, val_fraction, test_fraction)
        n_train, n_val,  n_test  = _do_split(normal_clips,  val_fraction, test_fraction)

        train_clips = a_train + n_train
        val_clips   = a_val   + n_val
        test_clips  = a_test  + n_test

        rng.shuffle(train_clips)
        rng.shuffle(val_clips)
        rng.shuffle(test_clips)

        for c in train_clips: c["split"] = "train"
        for c in val_clips:   c["split"] = "val"
        for c in test_clips:  c["split"] = "test"

        splits = {
            "train": train_clips,
            "val":   val_clips,
            "test":  test_clips,
        }

        # Count anomaly clips per split for logging
        def _count_anomaly(clips):
            return sum(1 for c in clips if c.get("gt_path"))

        logger.info(
            f"Stratified split summary — "
            f"train: {len(train_clips)} clips ({_count_anomaly(train_clips)} with GT), "
            f"val: {len(val_clips)} clips ({_count_anomaly(val_clips)} with GT), "
            f"test: {len(test_clips)} clips ({_count_anomaly(test_clips)} with GT)"
        )
        logger.warning(
            "NOTE: Using pooled stratified split (not the official UCSD train/test "
            "partition). Results are NOT comparable to published UCSD benchmarks."
        )
        return splits

    def save_manifests(self, splits: Dict[str, List[dict]], out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        for split_name, clips in splits.items():
            path = os.path.join(out_dir, f"{split_name}_manifest.json")
            with open(path, "w") as f:
                json.dump(clips, f, indent=2)
            logger.info(f"Saved {split_name} manifest → {path}")

    @staticmethod
    def load_manifest(path: str) -> List[dict]:
        with open(path) as f:
            return json.load(f)


# ── Full dataset assembly ─────────────────────────────────────────────────────
def assemble_split_arrays(
    clip_dfs: Dict[str, pd.DataFrame],
    split_manifest: List[dict],
    builder: SequenceBuilder,
    report: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Assemble (X, y, meta) arrays for a dataset split from per-clip DataFrames.

    Parameters
    ----------
    clip_dfs : dict mapping clip_id → DataFrame (features + label column)
    split_manifest : list of clip metadata dicts
    builder : SequenceBuilder instance
    report : print summary

    Returns
    -------
    X : (N, seq_len, n_features)
    y : (N,)
    meta : list of per-sequence metadata
    """
    X_parts, y_parts, meta_all = [], [], []

    for clip_meta in split_manifest:
        clip_id = clip_meta["clip_id"]
        if clip_id not in clip_dfs:
            logger.warning(f"No features found for clip '{clip_id}' — skipping.")
            continue

        df = clip_dfs[clip_id]
        X_c, y_c, meta_c = builder.build_from_clip(df, clip_id)
        if len(X_c) == 0:
            continue

        X_parts.append(X_c)
        y_parts.append(y_c)
        meta_all.extend(meta_c)

    if not X_parts:
        raise RuntimeError("No sequences assembled — check dataset and sequence length.")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    if report:
        n_anomaly = int(y.sum())
        n_normal = len(y) - n_anomaly
        logger.info(
            f"Assembled {len(y)} sequences | "
            f"Normal: {n_normal} ({100*n_normal/len(y):.1f}%) | "
            f"Anomaly: {n_anomaly} ({100*n_anomaly/len(y):.1f}%)"
        )

    return X, y, meta_all


def report_class_distribution(y: np.ndarray, split_name: str = "") -> Dict:
    """Report and return class distribution."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    report = {}
    lines = [f"  Class distribution [{split_name}] (total={total}):"]
    for cls, cnt in zip(unique, counts):
        pct = 100.0 * cnt / total
        label = "Normal" if cls == 0 else "Anomalous"
        lines.append(f"    class {cls} ({label}): {cnt} ({pct:.1f}%)")
        report[int(cls)] = {"count": int(cnt), "pct": float(pct)}
    logger.info("\n".join(lines))
    print("\n".join(lines))
    return report
