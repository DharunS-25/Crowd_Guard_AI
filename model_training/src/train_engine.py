"""
train_gru.py / train_lstm.py — shared training engine for CrowdGuard AI
========================================================================
This module implements the shared training loop used by both GRU and LSTM.

Usage (GRU):
    python train_gru.py --config configs/gru.yaml

Usage (LSTM):
    python train_lstm.py --config configs/lstm.yaml
    (train_lstm.py is a one-liner wrapper around this module with model_type=lstm)

Training loop features:
  - AdamW optimiser
  - CrossEntropyLoss with optional class weights
  - ReduceLROnPlateau scheduler
  - Gradient clipping
  - Validation-based early stopping (monitors val_f1)
  - Best-checkpoint saving
  - Per-epoch training history logging
  - CPU / GPU support
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from dataset import CrowdSequenceDataset, build_dataloaders
from metrics import full_evaluation, save_metrics
from reproducibility import set_seed, get_device, log_environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crowdguard.train")


# ── Training utilities ────────────────────────────────────────────────────────
class EarlyStopping:
    """Validation-metric-based early stopping."""

    def __init__(
        self,
        patience: int = 15,
        mode: str = "max",
        min_delta: float = 0.001,
    ):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best = None
        self.stop = False

    def step(self, metric: float) -> bool:
        """Returns True if training should stop."""
        if self.best is None:
            self.best = metric
            return False

        if self.mode == "max":
            improved = metric > self.best + self.min_delta
        else:
            improved = metric < self.best - self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    grad_clip: Optional[float] = 1.0,
) -> Tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    n_classes: int = 2,
) -> Dict:
    """Run validation. Returns metric dict."""
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        total += len(y_batch)

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())
        all_probs.append(probs[:, 1].cpu().numpy())  # prob of anomaly class

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    metrics = full_evaluation(y_true, y_pred, y_prob, n_classes=n_classes)
    metrics["loss"] = total_loss / total
    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    metrics: Dict,
    save_dir: str,
    filename: str = "best_model.pt",
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "model_config": model.config,
        },
        path,
    )
    logger.info(f"Checkpoint saved → {path}")
    return path


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device,
    optimizer=None,
) -> Dict:
    """Load a saved checkpoint into model (and optionally optimizer)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info(
        f"Loaded checkpoint from epoch {ckpt['epoch']} "
        f"(val_f1={ckpt['metrics'].get('f1_macro', 'N/A')})"
    )
    return ckpt


# ── Main training function ────────────────────────────────────────────────────
def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: Dict,
    save_dir: str,
    device: Optional[torch.device] = None,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[nn.Module, Dict]:
    """
    Full training loop for GRU or LSTM model.

    Parameters
    ----------
    model : the GRU or LSTM module
    train_loader : training DataLoader
    val_loader : validation DataLoader
    cfg : parsed training config (from gru.yaml / lstm.yaml)
    save_dir : directory to save checkpoints
    device : torch device
    class_weights : optional class weight tensor for imbalanced datasets

    Returns
    -------
    model : model loaded with best weights
    history : dict with lists of per-epoch metrics
    """
    if device is None:
        device = get_device()
    model = model.to(device)

    train_cfg = cfg.get("training", {})
    n_epochs = train_cfg.get("epochs", 80)
    lr = train_cfg.get("learning_rate", 0.001)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    grad_clip = train_cfg.get("gradient_clip_norm", 1.0)
    n_classes = cfg.get("architecture", {}).get("output_dim", 2)

    # Loss
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimiser
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    sched_cfg = train_cfg.get("scheduler", {})
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=sched_cfg.get("patience", 8),
        factor=sched_cfg.get("factor", 0.5),
        min_lr=sched_cfg.get("min_lr", 1e-6),
    )

    # Early stopping
    es_cfg = train_cfg.get("early_stopping", {})
    early_stopper = EarlyStopping(
        patience=es_cfg.get("patience", 15),
        mode=es_cfg.get("mode", "max"),
        min_delta=es_cfg.get("min_delta", 0.001),
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "val_f1_macro": [], "val_f1_anomaly": [],
        "val_recall_anomaly": [], "val_roc_auc": [],
        "lr": [],
    }

    best_val_f1 = -1.0
    best_epoch = 0
    t0 = time.time()

    logger.info(
        f"Starting training — {n_epochs} epochs, lr={lr}, "
        f"device={device}, params={model.count_parameters():,}"
    )

    for epoch in range(1, n_epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        t_loss, t_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, grad_clip
        )

        # ── Validate ───────────────────────────────────────────────────────
        v_metrics = validate(model, val_loader, criterion, device, n_classes)
        v_loss = v_metrics["loss"]
        v_acc = v_metrics["accuracy"]
        v_f1 = v_metrics.get("f1_macro", 0.0)
        v_f1_anom = v_metrics.get("f1_anomaly", 0.0)
        v_recall_anom = v_metrics.get("recall_anomaly", 0.0)
        v_auc = v_metrics.get("roc_auc", 0.0)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Record history ─────────────────────────────────────────────────
        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["val_f1_macro"].append(v_f1)
        history["val_f1_anomaly"].append(v_f1_anom)
        history["val_recall_anomaly"].append(v_recall_anom)
        history["val_roc_auc"].append(v_auc)
        history["lr"].append(current_lr)

        # ── Print epoch summary ────────────────────────────────────────────
        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:03d}/{n_epochs} | "
            f"TrLoss={t_loss:.4f} TrAcc={t_acc:.4f} | "
            f"VaLoss={v_loss:.4f} VaAcc={v_acc:.4f} | "
            f"F1_mac={v_f1:.4f} F1_anom={v_f1_anom:.4f} | "
            f"AUC={v_auc:.4f} LR={current_lr:.6f} | "
            f"Time={elapsed:.0f}s"
        )

        # ── Save best checkpoint ───────────────────────────────────────────
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, v_metrics, save_dir)

        # ── LR scheduler ──────────────────────────────────────────────────
        scheduler.step(v_f1)

        # ── Early stopping ─────────────────────────────────────────────────
        if early_stopper.step(v_f1):
            logger.info(
                f"Early stopping at epoch {epoch} "
                f"(best val_f1={best_val_f1:.4f} at epoch {best_epoch})"
            )
            break

    total_time = time.time() - t0
    logger.info(
        f"Training complete — {epoch} epochs in {total_time:.1f}s. "
        f"Best val_f1={best_val_f1:.4f} at epoch {best_epoch}."
    )

    # ── Reload best weights ────────────────────────────────────────────────
    best_ckpt = os.path.join(save_dir, "best_model.pt")
    if os.path.exists(best_ckpt):
        load_checkpoint(model, best_ckpt, device)

    history["best_epoch"] = best_epoch
    history["best_val_f1"] = best_val_f1
    history["total_training_seconds"] = total_time

    # Save history
    hist_path = os.path.join(save_dir, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved → {hist_path}")

    return model, history
