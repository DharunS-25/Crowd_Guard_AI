"""
inspect_dataset.py — CrowdGuard AI
=====================================
Phase 1 & 2: Inspect UCSD dataset, extract features, build processed arrays.

This is the PRIMARY data pipeline script. It must be run FIRST before training.

What it does:
  1. Scans the raw UCSD data directory (Peds1/Peds2 folder structure)
  2. Reads frame images (.tif files)
  3. Reads frame-level ground-truth anomaly labels from _gt files
  4. Extracts CrowdGuard AI features (optical flow, density, etc.)
  5. Performs clip-level train/val/test split (no frame-level leakage)
  6. Fits scaler on training data only
  7. Saves X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy
  8. Saves scaler.pkl, split manifests, and dataset summary

UCSD dataset structure expected:
  data/raw/ucsd/
    UCSDped1/
      Train/
        Train001/          ← folder of .tif frames
        Train001_gt        ← frame-level anomaly labels (1-indexed, space-separated)
        Train002/
        Train002_gt
        ...
      Test/
        Test001/
        Test001_gt
        ...
    UCSDped2/
      Train/
        ...
      Test/
        ...

Usage:
    python scripts/inspect_dataset.py
    python scripts/inspect_dataset.py --raw-dir data/raw/ucsd --seq-len 20
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).parent
_ROOT = _SCRIPTS.parent
sys.path.insert(0, str(_ROOT / "src"))

from preprocessing import (
    CORE_FEATURE_COLUMNS,
    FEATURE_DIM,
    StandardScaler,
    extract_features_from_frames,
    handle_missing_values,
    validate_feature_schema,
)
from sequence_builder import (
    DatasetSplitter,
    SequenceBuilder,
    assemble_split_arrays,
    report_class_distribution,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crowdguard.inspect")


# ── UCSD Label Parsing ────────────────────────────────────────────────────────
def parse_gt_file(gt_path: Optional[str], n_frames: int) -> np.ndarray:
    """
    Parse a UCSD ground-truth annotation.

    Supports two formats used in the UCSD dataset:

    1. **Mask directory** (*_gt/ folder containing per-frame .bmp images):
       Each BMP is a binary mask where pixel value > 0 indicates an anomalous
       region. A frame is labelled anomalous (1) if ANY pixel is non-zero.

    2. **Text file** (*_gt text file with start/end frame pairs):
       Lines contain pairs of 1-indexed frame numbers (start end).
       All frames in [start, end] inclusive are labelled anomalous.

    Returns
    -------
    np.ndarray of int64, shape (n_frames,): 0=normal, 1=anomalous
    """
    labels = np.zeros(n_frames, dtype=np.int64)

    if not gt_path or not os.path.exists(gt_path):
        logger.debug(f"No GT found at: {gt_path} — all frames labelled Normal")
        return labels

    # ── Case 1: GT is a directory of per-frame mask images ────────────────
    if os.path.isdir(gt_path):
        from PIL import Image
        mask_files = sorted(
            [f for f in Path(gt_path).iterdir()
             if f.suffix.lower() in {".bmp", ".png", ".tif", ".tiff"}],
            key=lambda f: f.name
        )
        for frame_idx, mask_path in enumerate(mask_files):
            if frame_idx >= n_frames:
                break
            try:
                mask = np.array(Image.open(mask_path))
                if mask.max() > 0:
                    labels[frame_idx] = 1
            except Exception as e:
                logger.warning(f"Failed to read GT mask {mask_path}: {e}")
        n_anomalous = int(labels.sum())
        logger.debug(
            f"GT dir {Path(gt_path).name}: {len(mask_files)} masks, "
            f"{n_anomalous}/{min(len(mask_files), n_frames)} anomalous frames"
        )
        return labels

    # ── Case 2: GT is a text file with (start, end) frame-range pairs ─────
    try:
        with open(gt_path) as f:
            content = f.read().strip()
        numbers = list(map(int, re.findall(r'\d+', content)))
        for i in range(0, len(numbers) - 1, 2):
            start = numbers[i] - 1      # convert 1-indexed to 0-indexed
            end = numbers[i + 1] - 1
            labels[max(0, start):min(n_frames, end + 1)] = 1
        logger.debug(
            f"GT text {Path(gt_path).name}: {int(labels.sum())}/{n_frames} "
            "anomalous frames"
        )
    except Exception as e:
        logger.warning(f"Failed to read GT text file {gt_path}: {e}")

    return labels


# ── Frame Loading ─────────────────────────────────────────────────────────────
def load_clip_frames(clip_dir: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """
    Load grayscale frames from a clip directory.
    Supports .tif, .png, .jpg files sorted by name.
    """
    clip_path = Path(clip_dir)
    extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    frame_files = sorted(
        [f for f in clip_path.iterdir()
         if f.suffix.lower() in extensions],
        key=lambda f: f.name
    )

    if max_frames:
        frame_files = frame_files[:max_frames]

    frames = []
    for fp in frame_files:
        try:
            from PIL import Image
            img = Image.open(fp).convert("L")  # grayscale
            frames.append(np.array(img, dtype=np.uint8))
        except Exception as e:
            logger.warning(f"Failed to load {fp}: {e}")

    return frames


# ── Clip Discovery ────────────────────────────────────────────────────────────
def discover_clips(ucsd_dir: str) -> List[dict]:
    """
    Walk the UCSD directory structure and return a list of clip metadata dicts.

    Returns list of dicts with:
        clip_id, subset, official_split, clip_dir, gt_path
    """
    clips = []
    ucsd_path = Path(ucsd_dir)

    # Try both naming conventions: UCSDped1/UCSDped2 and Ped1/Ped2
    subset_dirs = []
    for name in ["UCSDped1", "UCSDped2", "Ped1", "Ped2", "UCSD_Anomaly_Dataset/UCSDped1",
                  "UCSD_Anomaly_Dataset/UCSDped2"]:
        d = ucsd_path / name
        if d.exists():
            subset_dirs.append(d)

    if not subset_dirs:
        # Try to find any subdirectory with Train/Test structure
        for child in ucsd_path.iterdir():
            if child.is_dir():
                if (child / "Train").exists() or (child / "Test").exists():
                    subset_dirs.append(child)
        if not subset_dirs:
            # Also try top-level Train/Test
            for top_split in ["Train", "Test"]:
                if (ucsd_path / top_split).exists():
                    # Treat as single-subset dataset
                    subset_dirs = [ucsd_path]
                    break

    logger.info(f"Found subset directories: {[str(d) for d in subset_dirs]}")

    for subset_dir in subset_dirs:
        subset_name = subset_dir.name

        for split_name in ["Train", "Test"]:
            split_dir = subset_dir / split_name
            if not split_dir.exists():
                continue

            for item in sorted(split_dir.iterdir()):
                if not item.is_dir():
                    continue

                # Skip _gt directories/files
                if "gt" in item.name.lower():
                    continue

                clip_id = f"{subset_name}_{split_name}_{item.name}"
                # GT can be a text file OR a directory of per-frame mask images
                gt_candidates = [
                    split_dir / f"{item.name}_gt",       # dir (mask images) or text
                    split_dir / f"{item.name}_gt.txt",   # explicit text file
                    item.parent / f"{item.name}_gt",
                ]
                gt_path = next(
                    (str(g) for g in gt_candidates if g.exists()), None
                )

                clips.append({
                    "clip_id": clip_id,
                    "subset": subset_name,
                    "official_split": split_name.lower(),
                    "clip_dir": str(item),
                    "gt_path": gt_path,
                })

    logger.info(f"Discovered {len(clips)} clips total.")
    return clips


# ── Feature Extraction Pipeline ───────────────────────────────────────────────
def process_clip(
    clip: dict,
    seq_len: int,
    sampling_interval: int = 5,
    resize_hw: Tuple[int, int] = (64, 64),
) -> Optional[pd.DataFrame]:
    """Extract features for a single clip. Returns DataFrame or None."""
    frames = load_clip_frames(clip["clip_dir"])
    if not frames:
        logger.warning(f"No frames found in {clip['clip_dir']}")
        return None

    n_frames = len(frames)
    labels = parse_gt_file(clip["gt_path"], n_frames)

    # Sub-sample frames
    frame_indices = list(range(0, n_frames, sampling_interval))
    sampled_frames = [frames[i] for i in frame_indices]
    sampled_labels = labels[frame_indices]

    # Extract features
    feature_df = extract_features_from_frames(sampled_frames, resize_hw=resize_hw)
    feature_df["label"] = sampled_labels

    # Validate
    handle_missing_values(feature_df)

    return feature_df


def main():
    p = argparse.ArgumentParser(description="Inspect and process UCSD dataset")
    p.add_argument("--raw-dir", default=str(_ROOT / "data" / "raw" / "ucsd"))
    p.add_argument("--out-dir", default=str(_ROOT / "data" / "processed"))
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--sampling-interval", type=int, default=5)
    p.add_argument("--max-clips", type=int, default=None,
                   help="Limit clips for testing (None = all)")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    splits_dir = os.path.join(args.out_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    t0 = time.time()

    # ── 1. Discover clips ─────────────────────────────────────────────────
    logger.info(f"Scanning dataset: {args.raw_dir}")
    if not os.path.exists(args.raw_dir):
        logger.error(
            f"Dataset directory not found: {args.raw_dir}\n"
            "Download the UCSD Anomaly Dataset from Kaggle and extract to data/raw/ucsd/"
        )
        sys.exit(1)

    clips = discover_clips(args.raw_dir)
    if not clips:
        logger.error(
            "No clips found. Check the directory structure in data/raw/ucsd/\n"
            "Expected: UCSDped1/Train/Train001/, UCSDped1/Test/Test001/, etc."
        )
        sys.exit(1)

    if args.max_clips:
        clips = clips[:args.max_clips]
        logger.info(f"Limited to {args.max_clips} clips for testing")

    # ── 2. Split clips ────────────────────────────────────────────────────
    splitter = DatasetSplitter(seed=args.seed)
    splits = splitter.split_clips(clips, val_fraction=args.val_fraction,
                                   test_fraction=args.test_fraction)
    splitter.save_manifests(splits, splits_dir)

    # ── 3. Extract features ───────────────────────────────────────────────
    logger.info(f"\nExtracting features from {len(clips)} clips…")
    logger.info("This may take several minutes for the full dataset.")

    clip_dfs = {}
    failed_clips = []
    for i, clip in enumerate(clips):
        try:
            df = process_clip(
                clip,
                seq_len=args.seq_len,
                sampling_interval=args.sampling_interval,
            )
            if df is not None and len(df) > 0:
                clip_dfs[clip["clip_id"]] = df
                if (i + 1) % 10 == 0:
                    logger.info(f"  Processed {i+1}/{len(clips)} clips…")
            else:
                failed_clips.append(clip["clip_id"])
        except Exception as e:
            logger.error(f"Failed to process clip {clip['clip_id']}: {e}")
            failed_clips.append(clip["clip_id"])

    logger.info(
        f"Feature extraction complete: "
        f"{len(clip_dfs)} succeeded, {len(failed_clips)} failed"
    )

    if not clip_dfs:
        logger.error("No features extracted. Cannot continue.")
        sys.exit(1)

    # ── 4. Build sequences ────────────────────────────────────────────────
    builder = SequenceBuilder(
        seq_len=args.seq_len,
        stride=args.stride,
        feature_cols=CORE_FEATURE_COLUMNS,
    )

    X_parts, y_parts = {"train": [], "val": [], "test": []}, \
                        {"train": [], "val": [], "test": []}
    meta_parts = {"train": [], "val": [], "test": []}

    for split_name, split_clips in splits.items():
        logger.info(f"\nBuilding sequences for {split_name} split…")
        for clip in split_clips:
            clip_id = clip["clip_id"]
            if clip_id not in clip_dfs:
                continue
            X_c, y_c, meta_c = builder.build_from_clip(clip_dfs[clip_id], clip_id)
            if len(X_c) > 0:
                X_parts[split_name].append(X_c)
                y_parts[split_name].append(y_c)
                meta_parts[split_name].extend(meta_c)

    # Concatenate
    arrays = {}
    for split_name in ["train", "val", "test"]:
        if not X_parts[split_name]:
            logger.warning(f"No sequences for split: {split_name}")
            arrays[split_name] = (
                np.empty((0, args.seq_len, FEATURE_DIM), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                [],
            )
        else:
            X = np.concatenate(X_parts[split_name], axis=0)
            y = np.concatenate(y_parts[split_name], axis=0)
            arrays[split_name] = (X, y, meta_parts[split_name])
            report_class_distribution(y, split_name)

    # ── 5. Fit scaler on training data only ───────────────────────────────
    X_train_raw = arrays["train"][0]
    if len(X_train_raw) == 0:
        logger.error("No training sequences. Cannot fit scaler.")
        sys.exit(1)

    logger.info(f"\nFitting scaler on training data only…")
    X_train_2d = X_train_raw.reshape(-1, FEATURE_DIM)
    scaler = StandardScaler()
    scaler.fit(X_train_2d, feature_names=CORE_FEATURE_COLUMNS)

    scaler_path = os.path.join(args.out_dir, "scaler.pkl")
    scaler.save(scaler_path)

    # ── 6. Apply scaler and save arrays ──────────────────────────────────
    logger.info("Normalising and saving split arrays…")
    stats = {}

    for split_name in ["train", "val", "test"]:
        X_raw, y, meta = arrays[split_name]
        if len(X_raw) == 0:
            continue

        n, L, F = X_raw.shape
        X_norm = scaler.transform(X_raw.reshape(-1, F)).reshape(n, L, F)

        X_path = os.path.join(args.out_dir, f"X_{split_name}.npy")
        y_path = os.path.join(args.out_dir, f"y_{split_name}.npy")
        meta_path = os.path.join(args.out_dir, f"meta_{split_name}.json")

        np.save(X_path, X_norm.astype(np.float32))
        np.save(y_path, y.astype(np.int64))
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        n_anomaly = int(y.sum())
        n_normal = len(y) - n_anomaly
        stats[split_name] = {
            "n_sequences": len(y),
            "n_normal": n_normal,
            "n_anomaly": n_anomaly,
            "pct_anomaly": round(100 * n_anomaly / len(y), 1) if len(y) > 0 else 0,
            "shape_X": list(X_norm.shape),
        }
        logger.info(
            f"  {split_name}: {len(y)} sequences saved | "
            f"Normal={n_normal}, Anomaly={n_anomaly}"
        )

    # ── 7. Save dataset summary ───────────────────────────────────────────
    summary = {
        "dataset": "UCSD Anomaly Detection Dataset",
        "feature_columns": CORE_FEATURE_COLUMNS,
        "feature_dim": FEATURE_DIM,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "sampling_interval": args.sampling_interval,
        "total_clips": len(clips),
        "failed_clips": failed_clips,
        "splits": stats,
        "scaler": {"method": "StandardScaler", "fit_on": "train"},
        "elapsed_seconds": round(time.time() - t0, 1),
        "label_schema": {
            "0": "Normal",
            "1": "Anomalous",
            "note": (
                "Binary anomaly labels from UCSD dataset. "
                "NOT validated crowd-risk classes. "
                "Model performs anomaly detection as a proxy."
            ),
        },
    }

    summary_path = os.path.join(args.out_dir, "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nDataset summary saved → {summary_path}")

    # ── Final report ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  DATASET INSPECTION COMPLETE")
    print(f"{'='*60}")
    for split_name, s in stats.items():
        print(f"  {split_name:5s}: {s['n_sequences']:6d} sequences "
              f"| Normal={s['n_normal']} | Anomaly={s['n_anomaly']} "
              f"({s['pct_anomaly']}%)")
    print(f"  Feature dim  : {FEATURE_DIM}")
    print(f"  Sequence len : {args.seq_len}")
    print(f"  Elapsed      : {round(time.time()-t0, 1)}s")
    print(f"{'='*60}\n")
    print("Next step: python src/train_gru.py")


if __name__ == "__main__":
    main()
