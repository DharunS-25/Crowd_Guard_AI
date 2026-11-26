"""
evaluate.py — CrowdGuard AI
==============================
Evaluates a trained model (GRU or LSTM) on the held-out test set.

Produces:
  - JSON metrics file
  - Confusion matrix plot
  - Per-class performance bar chart
  - Risk timeline plot
  - ROC curve

Usage:
    python evaluate.py --model-type gru
    python evaluate.py --model-type lstm
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

from dataset import CrowdSequenceDataset
from gru_model import CrowdGuardGRU, build_gru_from_config
from lstm_model import CrowdGuardLSTM, build_lstm_from_config
from metrics import full_evaluation, save_metrics
from reproducibility import set_seed, get_device
from train_engine import load_checkpoint, validate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crowdguard.evaluate")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CrowdGuard model")
    p.add_argument("--model-type", choices=["gru", "lstm"], required=True)
    p.add_argument("--data-dir", default=str(_ROOT / "data" / "processed"))
    p.add_argument("--output-dir", default=str(_ROOT / "outputs"))
    p.add_argument("--config-dir", default=str(_ROOT / "configs"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model_from_checkpoint(
    model_type: str,
    ckpt_path: str,
    device: torch.device,
):
    """Reconstruct and load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ckpt["model_config"]

    if model_type == "gru":
        model = CrowdGuardGRU(**{k: v for k, v in model_cfg.items()
                                  if k != "model_type"})
    else:
        model = CrowdGuardLSTM(**{k: v for k, v in model_cfg.items()
                                   if k != "model_type"})

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_all(
    model,
    X: np.ndarray,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
):
    """Run inference on all sequences. Returns (y_pred, y_prob)."""
    device = device or torch.device("cpu")
    model.eval()
    all_preds, all_probs = [], []
    n = len(X)

    for start in range(0, n, batch_size):
        batch = torch.tensor(X[start:start+batch_size], dtype=torch.float32).to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs[:, 1].cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_probs)


def plot_confusion_matrix(cm, class_names, title, save_path):
    """Plot and save confusion matrix."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=12)
        ax.set_yticklabels(class_names, fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=13)
        ax.set_ylabel("True Label", fontsize=13)
        ax.set_title(title, fontsize=14, fontweight="bold")

        thresh = cm.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=14, fontweight="bold",
                        color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Confusion matrix saved → {save_path}")
    except Exception as e:
        logger.warning(f"Could not plot confusion matrix: {e}")


def plot_training_curves(history_path: str, save_dir: str, model_name: str):
    """Plot training/validation loss and F1 curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with open(history_path) as f:
            history = json.load(f)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"{model_name} Training Curves", fontsize=14, fontweight="bold")

        # Loss
        axes[0].plot(history["train_loss"], label="Train Loss", color="#2196F3")
        axes[0].plot(history["val_loss"], label="Val Loss", color="#FF5722")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # F1
        axes[1].plot(history["val_f1_macro"], label="Val F1 (macro)", color="#4CAF50")
        axes[1].plot(history["val_f1_anomaly"], label="Val F1 (anomaly)", color="#FF9800")
        if "best_epoch" in history:
            axes[1].axvline(x=history["best_epoch"]-1, color="red",
                            linestyle="--", alpha=0.7, label=f"Best epoch")
        axes[1].set_title("F1 Score")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Recall anomaly
        axes[2].plot(history["val_recall_anomaly"], label="Val Recall (anomaly)",
                     color="#9C27B0")
        axes[2].set_title("Anomaly Recall (Safety Metric)")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{model_name.lower()}_training_curves.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Training curves saved → {save_path}")
    except Exception as e:
        logger.warning(f"Could not plot training curves: {e}")


def plot_risk_timeline(y_true, y_pred, y_prob, save_path: str, model_name: str,
                       n_frames: int = 300):
    """Plot predicted vs actual risk labels over time."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = min(n_frames, len(y_true))
        t = np.arange(n)

        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(f"{model_name} — Risk Prediction Timeline (first {n} sequences)",
                     fontsize=13, fontweight="bold")

        axes[0].fill_between(t, y_true[:n], alpha=0.6, color="#F44336", label="True Label")
        axes[0].set_ylabel("True Label")
        axes[0].set_ylim(-0.1, 1.4)
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        axes[1].fill_between(t, y_pred[:n], alpha=0.6, color="#2196F3", label="Predicted")
        axes[1].set_ylabel("Predicted Label")
        axes[1].set_ylim(-0.1, 1.4)
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t, y_prob[:n], color="#FF9800", linewidth=0.8, label="P(anomaly)")
        axes[2].axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
        axes[2].set_ylabel("Anomaly Probability")
        axes[2].set_xlabel("Sequence Index")
        axes[2].set_ylim(0, 1)
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Risk timeline saved → {save_path}")
    except Exception as e:
        logger.warning(f"Could not plot risk timeline: {e}")


def measure_inference_latency(model, input_dim: int, seq_len: int,
                               device: torch.device, n_runs: int = 100) -> dict:
    """Measure per-batch inference latency."""
    import time
    model.eval()
    dummy = torch.randn(1, seq_len, input_dim, device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(dummy)

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000  # ms
    return {
        "latency_mean_ms": round(float(np.mean(times)), 4),
        "latency_std_ms": round(float(np.std(times)), 4),
        "latency_p95_ms": round(float(np.percentile(times, 95)), 4),
        "device": str(device),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    model_type = args.model_type
    ckpt_path = os.path.join(
        args.output_dir, "checkpoints", model_type, "best_model.pt"
    )

    if not os.path.exists(ckpt_path):
        logger.error(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run train_{model_type}.py first."
        )
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────
    model, ckpt = load_model_from_checkpoint(model_type, ckpt_path, device)
    model_cfg = ckpt["model_config"]
    input_dim = model_cfg["input_dim"]
    n_classes = model_cfg["output_dim"]
    seq_len = ckpt.get("seq_len", 20)  # fallback
    logger.info(f"Loaded {model_type.upper()} — {model.count_parameters():,} params")

    # ── Load test data ────────────────────────────────────────────────────
    X_test = np.load(os.path.join(args.data_dir, "X_test.npy"), allow_pickle=False)
    y_test = np.load(os.path.join(args.data_dir, "y_test.npy"), allow_pickle=False)
    logger.info(f"Test set: {X_test.shape}")

    # ── Predict ───────────────────────────────────────────────────────────
    y_pred, y_prob = predict_all(model, X_test, batch_size=64, device=device)

    # ── Evaluate ──────────────────────────────────────────────────────────
    model_name = f"CrowdGuard_{model_type.upper()}"
    metrics = full_evaluation(
        y_test, y_pred, y_prob,
        n_classes=n_classes,
        model_name=model_name,
    )

    # Latency
    latency = measure_inference_latency(model, input_dim, seq_len, device)
    metrics["inference_latency"] = latency

    # Model size
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    metrics["model_size_kb"] = round(buf.tell() / 1024, 2)
    metrics["param_count"] = model.count_parameters()

    # ── Save metrics ──────────────────────────────────────────────────────
    metrics_path = os.path.join(
        args.output_dir, "metrics", f"{model_type}_metrics.json"
    )
    save_metrics(metrics, metrics_path)

    # ── Plots ─────────────────────────────────────────────────────────────
    plots_dir = os.path.join(args.output_dir, "plots")
    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(
        cm,
        class_names=["Normal", "Anomalous"],
        title=f"{model_name} — Confusion Matrix (Test Set)",
        save_path=os.path.join(plots_dir, f"{model_type}_confusion_matrix.png"),
    )

    plot_risk_timeline(
        y_test, y_pred, y_prob,
        save_path=os.path.join(plots_dir, f"{model_type}_risk_timeline.png"),
        model_name=model_name,
    )

    # Training curves
    hist_path = os.path.join(
        args.output_dir, "checkpoints", model_type, "training_history.json"
    )
    if os.path.exists(hist_path):
        plot_training_curves(hist_path, plots_dir, model_name)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {model_name} — Test Set Evaluation")
    print(f"{'='*60}")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  F1 (macro)     : {metrics['f1_macro']:.4f}")
    print(f"  F1 (anomaly)   : {metrics.get('f1_anomaly', 'N/A'):.4f}")
    print(f"  Recall (anomaly): {metrics.get('recall_anomaly', 'N/A'):.4f}")
    print(f"  FNR (anomaly)  : {metrics.get('false_negative_rate_anomaly', 'N/A'):.4f}")
    print(f"  ROC-AUC        : {metrics.get('roc_auc', 'N/A')}")
    print(f"  Latency (mean) : {latency['latency_mean_ms']:.2f} ms")
    print(f"  Model size     : {metrics['model_size_kb']:.1f} KB")
    print(f"  Params         : {metrics['param_count']:,}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
