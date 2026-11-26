"""
train_gru.py — CrowdGuard AI
==============================
Entry point to train the GRU model on the UCSD Anomaly Dataset.

Run:
    python train_gru.py

Or with custom config:
    python train_gru.py --config configs/gru.yaml --data-dir data/processed

What this script does:
  1. Loads processed numpy arrays (features + labels)
  2. Builds DataLoaders
  3. Constructs GRU model from config
  4. Trains with early stopping
  5. Evaluates on validation set
  6. Saves best checkpoint + training history
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

from dataset import build_dataloaders
from gru_model import build_gru_from_config
from preprocessing import CORE_FEATURE_COLUMNS, FEATURE_DIM
from train_engine import train_model
from reproducibility import set_seed, get_device, log_environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crowdguard.train_gru")


def parse_args():
    p = argparse.ArgumentParser(description="Train CrowdGuard GRU model")
    p.add_argument("--config", default=str(_ROOT / "configs" / "gru.yaml"))
    p.add_argument("--data-dir", default=str(_ROOT / "data" / "processed"))
    p.add_argument("--output-dir", default=str(_ROOT / "outputs"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true",
                   help="Run 2 epochs with small batch for quick validation")
    return p.parse_args()


def load_split_arrays(data_dir: str, split: str):
    """Load pre-built numpy arrays for a given split."""
    X_path = os.path.join(data_dir, f"X_{split}.npy")
    y_path = os.path.join(data_dir, f"y_{split}.npy")
    meta_path = os.path.join(data_dir, f"meta_{split}.json")

    if not os.path.exists(X_path):
        raise FileNotFoundError(
            f"Processed data not found: {X_path}\n"
            "Run scripts/inspect_dataset.py first to extract features."
        )

    X = np.load(X_path, allow_pickle=False)
    y = np.load(y_path, allow_pickle=False)
    meta = []
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return X, y, meta


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    # ── Load config ───────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_environment()
    logger.info(f"Config: {args.config}")
    logger.info(f"Data dir: {args.data_dir}")

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info("Loading processed dataset splits…")
    X_train, y_train, meta_train = load_split_arrays(args.data_dir, "train")
    X_val, y_val, meta_val = load_split_arrays(args.data_dir, "val")
    X_test, y_test, meta_test = load_split_arrays(args.data_dir, "test")

    logger.info(
        f"Shapes — Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )
    logger.info(
        f"Train labels — Normal: {(y_train==0).sum()}, Anomaly: {(y_train==1).sum()}"
    )

    input_dim = X_train.shape[2]
    logger.info(f"Input feature dim: {input_dim}")

    # ── Smoke test override ───────────────────────────────────────────────
    if args.smoke_test:
        logger.info("SMOKE TEST MODE — 2 epochs, 64 samples")
        X_train = X_train[:64]
        y_train = y_train[:64]
        X_val = X_val[:32]
        y_val = y_val[:32]
        cfg["training"]["epochs"] = 2

    # ── Build DataLoaders ─────────────────────────────────────────────────
    batch_size = cfg["training"]["batch_size"]
    train_loader, val_loader, test_loader, train_ds = build_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=batch_size,
        use_weighted_sampler=cfg["training"].get("use_class_weights", True),
        meta_train=meta_train,
        meta_val=meta_val,
        meta_test=meta_test,
    )

    # ── Build model ───────────────────────────────────────────────────────
    model = build_gru_from_config(cfg, input_dim=input_dim)
    logger.info(f"GRU model: {model.count_parameters():,} trainable parameters")

    # ── Class weights for loss ────────────────────────────────────────────
    class_weights = None
    if cfg["training"].get("use_class_weights", True):
        class_weights = train_ds.compute_class_weights(n_classes=2)
        logger.info(f"Loss class weights: {class_weights.tolist()}")
        # If training data is single-class, disable weighting to avoid
        # confusing the model (uniform weights = no weighting effect)
        unique_labels = set(int(v) for v in train_ds.y.unique())
        if len(unique_labels) < 2:
            logger.warning(
                "Training split contains only 1 class — class weighting "
                "disabled. Check your data pipeline (label extraction)."
            )
            class_weights = None

    # ── Train ─────────────────────────────────────────────────────────────
    save_dir = os.path.join(
        args.output_dir, "checkpoints", "gru"
    )
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        save_dir=save_dir,
        device=device,
        class_weights=class_weights,
    )

    logger.info("GRU training complete.")

    if not args.smoke_test:
        logger.info("Best checkpoint ready for evaluation via evaluate.py")


if __name__ == "__main__":
    main()
