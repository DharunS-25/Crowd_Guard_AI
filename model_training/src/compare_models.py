"""
compare_models.py -- CrowdGuard AI
====================================
Side-by-side comparison of GRU vs LSTM on the held-out test set.

Reads pre-computed metrics JSON files and:
  1. Prints a formatted comparison table
  2. Saves outputs/metrics/model_comparison.csv
  3. Generates a comparison bar chart

Usage:
    python compare_models.py
"""

import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crowdguard.compare")

METRICS_DIR = _ROOT / "outputs" / "metrics"
PLOTS_DIR = _ROOT / "outputs" / "plots"


def load_metrics(model_type: str) -> dict:
    path = METRICS_DIR / f"{model_type}_metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics not found: {path}\n"
            f"Run evaluate.py --model-type {model_type} first."
        )
    with open(path) as f:
        return json.load(f)


def comparison_table(gru: dict, lstm: dict) -> list:
    """Build comparison rows."""
    rows = []

    metric_labels = [
        ("accuracy", "Accuracy"),
        ("f1_macro", "F1 (macro)"),
        ("precision_macro", "Precision (macro)"),
        ("recall_macro", "Recall (macro)"),
        ("f1_anomaly", "F1 (anomaly class)"),
        ("recall_anomaly", "Recall (anomaly)  [<--] Safety Critical"),
        ("precision_anomaly", "Precision (anomaly)"),
        ("false_negative_rate_anomaly", "False Negative Rate (anomaly)  [<--] Lower is better"),
        ("roc_auc", "ROC-AUC"),
        ("param_count", "Parameter Count"),
        ("model_size_kb", "Model Size (KB)"),
    ]

    for key, label in metric_labels:
        gru_val = gru.get(key, "N/A")
        lstm_val = lstm.get(key, "N/A")

        # For latency, drill into nested dict
        if key == "latency_mean_ms":
            gru_val = gru.get("inference_latency", {}).get("latency_mean_ms", "N/A")
            lstm_val = lstm.get("inference_latency", {}).get("latency_mean_ms", "N/A")

        # Determine winner
        if gru_val == "N/A" or lstm_val == "N/A":
            winner = "N/A"
        elif key in ("param_count", "model_size_kb", "false_negative_rate_anomaly"):
            winner = "GRU" if gru_val <= lstm_val else "LSTM"
        else:
            winner = "GRU" if gru_val >= lstm_val else "LSTM"

        rows.append({
            "metric": label,
            "gru": gru_val,
            "lstm": lstm_val,
            "winner": winner,
        })

    # Add latency
    gru_lat = gru.get("inference_latency", {})
    lstm_lat = lstm.get("inference_latency", {})
    for lat_key, lat_label in [
        ("latency_mean_ms", "Inference Latency Mean (ms)  [<--] Lower is better"),
        ("latency_p95_ms", "Inference Latency P95 (ms)   [<--] Lower is better"),
    ]:
        gv = gru_lat.get(lat_key, "N/A")
        lv = lstm_lat.get(lat_key, "N/A")
        winner = "N/A"
        if gv != "N/A" and lv != "N/A":
            winner = "GRU" if gv <= lv else "LSTM"
        rows.append({"metric": lat_label, "gru": gv, "lstm": lv, "winner": winner})

    return rows


def print_comparison(rows: list):
    header = f"{'Metric':<50} {'GRU':>12} {'LSTM':>12} {'Better':>8}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print("  CrowdGuard AI -- GRU vs LSTM Comparison (Test Set)")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        gv = r["gru"]
        lv = r["lstm"]
        gstr = f"{gv:.4f}" if isinstance(gv, float) else str(gv)
        lstr = f"{lv:.4f}" if isinstance(lv, float) else str(lv)
        print(f"  {r['metric']:<48} {gstr:>12} {lstr:>12} {r['winner']:>8}")
    print(sep + "\n")


def save_comparison_csv(rows: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "gru", "lstm", "winner"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Comparison CSV saved [-->] {path}")


def plot_comparison(rows: list, save_dir: str):
    """Bar chart comparison of key metrics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        key_metrics = [
            "Accuracy", "F1 (macro)", "F1 (anomaly class)",
            "Recall (anomaly)  [<--] Safety Critical", "ROC-AUC"
        ]
        filtered = [r for r in rows if any(k in r["metric"] for k in key_metrics)]

        labels = [r["metric"].split("[<--]")[0].strip() for r in filtered]
        gru_vals = [float(r["gru"]) if isinstance(r["gru"], (int, float)) else 0 for r in filtered]
        lstm_vals = [float(r["lstm"]) if isinstance(r["lstm"], (int, float)) else 0 for r in filtered]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 5))
        bars_gru = ax.bar(x - width/2, gru_vals, width, label="GRU",
                          color="#2196F3", alpha=0.85, edgecolor="white")
        bars_lstm = ax.bar(x + width/2, lstm_vals, width, label="LSTM",
                           color="#FF5722", alpha=0.85, edgecolor="white")

        ax.set_ylim(0, 1.15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("CrowdGuard AI -- GRU vs LSTM Comparison", fontsize=14,
                     fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        for bar in bars_gru:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8, color="#1565C0")
        for bar in bars_lstm:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8, color="#BF360C")

        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "gru_vs_lstm_comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Comparison chart saved [-->] {save_path}")
    except Exception as e:
        logger.warning(f"Could not generate comparison chart: {e}")


def recommendation(gru: dict, lstm: dict) -> str:
    """Generate a model deployment recommendation."""
    gru_f1 = gru.get("f1_macro", 0) or 0
    lstm_f1 = lstm.get("f1_macro", 0) or 0
    gru_recall = gru.get("recall_anomaly", 0) or 0
    lstm_recall = lstm.get("recall_anomaly", 0) or 0
    gru_auc = gru.get("roc_auc", 0) or 0
    lstm_auc = lstm.get("roc_auc", 0) or 0
    gru_params = gru.get("param_count", 1e9)
    lstm_params = lstm.get("param_count", 1e9)

    # Score: weight recall_anomaly highest (safety), then F1, then AUC
    gru_score = 0.5 * gru_recall + 0.3 * gru_f1 + 0.2 * gru_auc
    lstm_score = 0.5 * lstm_recall + 0.3 * lstm_f1 + 0.2 * lstm_auc

    lines = [
        "DEPLOYMENT RECOMMENDATION",
        "=" * 60,
        "",
        "Primary evaluation criteria (safety-oriented):",
        "  50% weight: Anomaly Recall (minimise missed events)",
        "  30% weight: Macro F1 (overall classification quality)",
        "  20% weight: ROC-AUC (discrimination ability)",
        "",
        f"  GRU score:  {gru_score:.4f}",
        f"  LSTM score: {lstm_score:.4f}",
        "",
    ]

    if abs(gru_score - lstm_score) < 0.005:
        lines += [
            "VERDICT: Models are statistically comparable in performance.",
            f"  [-->] Recommend GRU for deployment (fewer parameters: "
            f"{gru_params:,} vs {lstm_params:,}, faster inference).",
        ]
    elif gru_score > lstm_score:
        lines += [
            f"VERDICT: GRU outperforms LSTM on safety-weighted score "
            f"({gru_score:.4f} vs {lstm_score:.4f}).",
            "  [-->] Recommend GRU for deployment.",
        ]
    else:
        lines += [
            f"VERDICT: LSTM outperforms GRU on safety-weighted score "
            f"({lstm_score:.4f} vs {gru_score:.4f}).",
            "  [-->] Recommend LSTM for deployment.",
        ]

    lines += [
        "",
        "IMPORTANT DISCLAIMER:",
        "  These are experimental results on the UCSD anomaly dataset.",
        "  The model has NOT been validated on independent controlled-event",
        "  venue recordings. Do not claim production readiness until",
        "  sufficient real-world venue validation is completed.",
        "=" * 60,
    ]

    return "\n".join(lines)


def main():
    try:
        gru_metrics = load_metrics("gru")
        lstm_metrics = load_metrics("lstm")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    rows = comparison_table(gru_metrics, lstm_metrics)
    print_comparison(rows)

    csv_path = str(METRICS_DIR / "model_comparison.csv")
    save_comparison_csv(rows, csv_path)
    plot_comparison(rows, str(PLOTS_DIR))

    rec = recommendation(gru_metrics, lstm_metrics)
    print(rec)

    rec_path = str(METRICS_DIR / "recommendation.txt")
    with open(rec_path, "w") as f:
        f.write(rec)
    logger.info(f"Recommendation saved [-->] {rec_path}")


if __name__ == "__main__":
    main()
