"""
train_lstm.py — CrowdGuard AI
===============================
Entry point to train the LSTM model on the UCSD Anomaly Dataset.

Run:
    python train_lstm.py

Mirrors train_gru.py exactly (same data, same splits, same evaluation)
to ensure fair comparison. Only the model architecture differs.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import yaml

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

from dataset import build_dataloaders
from lstm_model import build_lstm_from_config
from train_engine import train_model
from reproducibility import set_seed, get_device, log_environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crowdguard.train_lstm")


def parse_args():
    p = argparse.ArgumentParser(description="Train CrowdGuard LSTM model")
    p.add_argument("--config", default=str(_ROOT / "configs" / "lstm.yaml"))
    p.add_argument("--data-dir", default=str(_ROOT / "data" / "processed"))
    p.add_argument("--output-dir", default=str(_ROOT / "outputs"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def load_split_arrays(data_dir: str, split: str):
    X = np.load(os.path.join(data_dir, f"X_{split}.npy"), allow_pickle=False)
    y = np.load(os.path.join(data_dir, f"y_{split}.npy"), allow_pickle=False)
    meta = []
    mp = os.path.join(data_dir, f"meta_{split}.json")
    if os.path.exists(mp):
        with open(mp) as f:
            meta = json.load(f)
    return X, y, meta


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_environment()
    logger.info(f"Config: {args.config}")

    X_train, y_train, meta_train = load_split_arrays(args.data_dir, "train")
    X_val, y_val, meta_val = load_split_arrays(args.data_dir, "val")
    X_test, y_test, meta_test = load_split_arrays(args.data_dir, "test")

    logger.info(
        f"Shapes — Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )

    input_dim = X_train.shape[2]

    if args.smoke_test:
        logger.info("SMOKE TEST MODE — 2 epochs")
        X_train = X_train[:64]
        y_train = y_train[:64]
        X_val = X_val[:32]
        y_val = y_val[:32]
        cfg["training"]["epochs"] = 2

    batch_size = cfg["training"]["batch_size"]
    train_loader, val_loader, test_loader, train_ds = build_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=batch_size,
        use_weighted_sampler=cfg["training"].get("use_class_weights", True),
        meta_train=meta_train,
        meta_val=meta_val,
        meta_test=meta_test,
    )

    model = build_lstm_from_config(cfg, input_dim=input_dim)
    logger.info(f"LSTM model: {model.count_parameters():,} trainable parameters")

    class_weights = None
    if cfg["training"].get("use_class_weights", True):
        class_weights = train_ds.compute_class_weights(n_classes=2)
        logger.info(f"Loss class weights: {class_weights.tolist()}")
        unique_labels = set(int(v) for v in train_ds.y.unique())
        if len(unique_labels) < 2:
            logger.warning(
                "Training split contains only 1 class — class weighting "
                "disabled. Check your data pipeline (label extraction)."
            )
            class_weights = None

    save_dir = os.path.join(args.output_dir, "checkpoints", "lstm")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        save_dir=save_dir,
        device=device,
        class_weights=class_weights,
    )

    logger.info("LSTM training complete.")


if __name__ == "__main__":
    main()
