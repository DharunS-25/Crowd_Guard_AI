"""
metrics.py — CrowdGuard AI
============================
Evaluation metrics for crowd risk / anomaly classification.

Supports binary and multi-class tasks.
All metrics are computed from numpy arrays — no sklearn dependency assumed,
but sklearn is used if available for richer outputs.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 2,
) -> np.ndarray:
    """Compute confusion matrix without sklearn."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_metrics(cm: np.ndarray) -> Dict[int, Dict]:
    """Compute precision, recall, F1 per class from confusion matrix."""
    n_classes = cm.shape[0]
    results = {}
    for c in range(n_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum()) - tp
        fn = int(cm[c, :].sum()) - tp
        tn = int(cm.sum()) - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        results[c] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "false_negative_rate": round(false_negative_rate, 4),
        }
    return results


def macro_metrics(per_class: Dict[int, Dict]) -> Dict:
    """Macro-averaged precision, recall, F1."""
    classes = list(per_class.keys())
    precisions = [per_class[c]["precision"] for c in classes]
    recalls = [per_class[c]["recall"] for c in classes]
    f1s = [per_class[c]["f1"] for c in classes]
    return {
        "precision_macro": round(float(np.mean(precisions)), 4),
        "recall_macro": round(float(np.mean(recalls)), 4),
        "f1_macro": round(float(np.mean(f1s)), 4),
    }


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def compute_roc_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    positive_class: int = 1,
) -> Optional[float]:
    """
    Compute ROC-AUC for binary classification.
    y_prob should be probability of positive class.
    """
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        pass

    # Minimal implementation
    sorted_idx = np.argsort(-y_prob)
    y_sorted = y_true[sorted_idx]
    n_pos = (y_true == positive_class).sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    tps = np.cumsum(y_sorted == positive_class)
    fps = np.cumsum(y_sorted != positive_class)
    tpr = tps / n_pos
    fpr = fps / n_neg
    auc = float(np.trapz(tpr, fpr))
    return round(abs(auc), 4)


def full_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    n_classes: int = 2,
    positive_class: int = 1,
    model_name: str = "model",
) -> Dict:
    """
    Compute full evaluation suite.

    Parameters
    ----------
    y_true : ground truth labels
    y_pred : predicted class labels
    y_prob : predicted probabilities for positive class (binary)
    n_classes : number of classes
    positive_class : index of positive (anomalous) class
    model_name : identifier for logging

    Returns
    -------
    dict with all metrics
    """
    results = {"model": model_name}

    # Accuracy
    acc = compute_accuracy(y_true, y_pred)
    results["accuracy"] = round(acc, 4)

    # Confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred, n_classes=n_classes)
    results["confusion_matrix"] = cm.tolist()

    # Per-class metrics
    per_class = per_class_metrics(cm)
    results["per_class"] = per_class

    # Macro averages
    macro = macro_metrics(per_class)
    results.update(macro)

    # Anomaly class specifically
    if positive_class in per_class:
        results["recall_anomaly"] = per_class[positive_class]["recall"]
        results["precision_anomaly"] = per_class[positive_class]["precision"]
        results["f1_anomaly"] = per_class[positive_class]["f1"]
        results["false_negative_rate_anomaly"] = per_class[positive_class][
            "false_negative_rate"
        ]

    # ROC-AUC
    if y_prob is not None:
        auc = compute_roc_auc(y_true, y_prob, positive_class)
        results["roc_auc"] = auc

    # Summary log
    logger.info(
        f"[{model_name}] "
        f"Acc={acc:.4f} | "
        f"F1_macro={results.get('f1_macro', 'N/A'):.4f} | "
        f"F1_anomaly={results.get('f1_anomaly', 'N/A'):.4f} | "
        f"ROC-AUC={results.get('roc_auc', 'N/A')}"
    )

    return results


def save_metrics(results: Dict, path: str):
    """Save evaluation results dict to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    logger.info(f"Metrics saved to {path}")


def load_metrics(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)
