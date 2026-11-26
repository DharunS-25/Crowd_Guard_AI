"""
inference.py — CrowdGuard AI
==============================
Shared inference interface for GRU and LSTM models.

This module provides a unified CrowdGuardPredictor class that:
  - Loads either GRU or LSTM from checkpoint
  - Validates input sequence shape and completeness
  - Returns structured prediction output including:
      * Predicted class (0=Normal, 1=Anomalous)
      * Class probabilities
      * Prediction timestamp
      * Camera/zone identifiers
      * Model identifier
      * Data quality warnings
      * Confidence level

IMPORTANT:
  All predictions are labelled EXPERIMENTAL.
  This model has not been validated for real-world crowd-risk assessment.
  Do not use for safety-critical decisions without independent validation.

Usage:
    predictor = CrowdGuardPredictor(model_type="gru")
    result = predictor.predict(sequence, camera_id="cam01", zone_id="entry")
    print(result)
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

from gru_model import CrowdGuardGRU
from lstm_model import CrowdGuardLSTM
from preprocessing import StandardScaler, CORE_FEATURE_COLUMNS, handle_missing_values

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crowdguard.inference")

CLASS_NAMES = {0: "Normal", 1: "Anomalous"}
RISK_DESCRIPTION = {
    0: "No anomaly detected — normal pedestrian movement.",
    1: "EXPERIMENTAL: Anomalous pattern detected — human review recommended.",
}


class DataQualityChecker:
    """Validates input sequence quality before inference."""

    def __init__(self, feature_cols: List[str], nan_threshold: float = 0.3):
        self.feature_cols = feature_cols
        self.nan_threshold = nan_threshold

    def check(self, sequence: np.ndarray) -> Dict:
        """
        Parameters
        ----------
        sequence : np.ndarray of shape (seq_len, n_features)

        Returns
        -------
        dict with 'ok': bool and 'warnings': list of str
        """
        warnings = []
        ok = True

        nan_frac = np.isnan(sequence).mean()
        if nan_frac > self.nan_threshold:
            warnings.append(
                f"Input sequence has {nan_frac:.1%} missing values "
                f"(threshold: {self.nan_threshold:.1%}). "
                "Prediction reliability is reduced."
            )
            ok = False
        elif nan_frac > 0:
            warnings.append(
                f"Input sequence has {nan_frac:.1%} missing values — filled with 0."
            )

        if np.any(np.isinf(sequence)):
            warnings.append("Input sequence contains infinite values — clipped.")

        return {"ok": ok, "nan_fraction": float(nan_frac), "warnings": warnings}


class CrowdGuardPredictor:
    """
    Unified inference interface for CrowdGuard GRU/LSTM models.

    Parameters
    ----------
    model_type : 'gru' or 'lstm'
    checkpoint_dir : directory containing best_model.pt
    scaler_path : path to saved StandardScaler
    confidence_threshold : probability threshold for positive class
    device : torch device
    """

    def __init__(
        self,
        model_type: str = "gru",
        checkpoint_dir: Optional[str] = None,
        scaler_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        self.model_type = model_type.lower()
        self.confidence_threshold = confidence_threshold
        self.device = device or torch.device("cpu")

        # Default paths
        if checkpoint_dir is None:
            checkpoint_dir = str(_ROOT / "outputs" / "checkpoints" / model_type)
        if scaler_path is None:
            scaler_path = str(_ROOT / "data" / "processed" / "scaler.pkl")

        self.checkpoint_dir = checkpoint_dir
        self.scaler_path = scaler_path

        self.model = None
        self.scaler = None
        self.model_config = None
        self.seq_len = None
        self.input_dim = None
        self.feature_cols = CORE_FEATURE_COLUMNS

        self._load()

    def _load(self):
        """Load model and scaler from disk."""
        ckpt_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"No checkpoint found at {ckpt_path}.\n"
                f"Run train_{self.model_type}.py first."
            )

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model_config = ckpt["model_config"]
        self.input_dim = self.model_config["input_dim"]

        if self.model_type == "gru":
            self.model = CrowdGuardGRU(**{
                k: v for k, v in self.model_config.items() if k != "model_type"
            })
        else:
            self.model = CrowdGuardLSTM(**{
                k: v for k, v in self.model_config.items() if k != "model_type"
            })

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        if os.path.exists(self.scaler_path):
            self.scaler = StandardScaler.load(self.scaler_path)
            logger.info(f"Scaler loaded from {self.scaler_path}")
        else:
            logger.warning(
                f"Scaler not found at {self.scaler_path} — "
                "input will not be normalised."
            )

        self.quality_checker = DataQualityChecker(self.feature_cols)
        logger.info(
            f"[{self.model_type.upper()}] Loaded — "
            f"{self.model.count_parameters():,} params"
        )

    @torch.no_grad()
    def predict(
        self,
        sequence: np.ndarray,
        camera_id: str = "unknown",
        zone_id: str = "unknown",
        timestamp: Optional[str] = None,
    ) -> Dict:
        """
        Run inference on a single temporal sequence.

        Parameters
        ----------
        sequence : np.ndarray of shape (seq_len, n_features)
            Time-ordered feature vectors. Must match training feature schema.
        camera_id : identifier for the source camera
        zone_id : identifier for the monitored zone
        timestamp : ISO 8601 timestamp string (defaults to now)

        Returns
        -------
        dict with prediction results and metadata
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        t_start = time.perf_counter()

        # ── Input validation ──────────────────────────────────────────────
        if not isinstance(sequence, np.ndarray):
            sequence = np.array(sequence, dtype=np.float32)

        quality = self.quality_checker.check(sequence)

        # Sanitise
        sequence = np.nan_to_num(sequence, nan=0.0)
        sequence = np.clip(sequence, -1e6, 1e6)

        # Shape check
        if sequence.ndim != 2:
            raise ValueError(
                f"Expected sequence shape (seq_len, n_features), got {sequence.shape}"
            )

        # ── Normalise ─────────────────────────────────────────────────────
        if self.scaler is not None:
            sequence = self.scaler.transform(sequence.astype(np.float32))

        # ── Inference ─────────────────────────────────────────────────────
        x = torch.tensor(sequence[np.newaxis, :, :], dtype=torch.float32)
        x = x.to(self.device)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        anomaly_prob = float(probs[1])

        latency_ms = (time.perf_counter() - t_start) * 1000

        # ── Build output ──────────────────────────────────────────────────
        result = {
            "status": "experimental",
            "disclaimer": (
                "EXPERIMENTAL PREDICTION. Model trained on UCSD anomaly dataset. "
                "Not validated for real-world crowd-risk assessment."
            ),
            "timestamp": timestamp,
            "camera_id": camera_id,
            "zone_id": zone_id,
            "model_id": f"crowdguard_{self.model_type}_v1",
            "prediction": {
                "class": pred_class,
                "label": CLASS_NAMES[pred_class],
                "description": RISK_DESCRIPTION[pred_class],
                "confidence": round(confidence, 4),
                "anomaly_probability": round(anomaly_prob, 4),
                "probabilities": {
                    CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)
                },
            },
            "data_quality": quality,
            "inference_latency_ms": round(latency_ms, 3),
        }

        return result

    def predict_batch(
        self,
        sequences: np.ndarray,
        camera_ids: Optional[List[str]] = None,
        zone_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Run inference on a batch of sequences.

        Parameters
        ----------
        sequences : np.ndarray of shape (N, seq_len, n_features)

        Returns
        -------
        list of prediction dicts
        """
        results = []
        n = len(sequences)
        camera_ids = camera_ids or ["unknown"] * n
        zone_ids = zone_ids or ["unknown"] * n

        for i, seq in enumerate(sequences):
            r = self.predict(seq, camera_ids[i], zone_ids[i])
            results.append(r)
        return results


def demo():
    """Quick demo using random data."""
    print("\n[CrowdGuard AI Inference Demo]")
    print("Loading GRU predictor…")

    try:
        predictor = CrowdGuardPredictor(model_type="gru")
    except FileNotFoundError as e:
        print(f"Cannot run demo: {e}")
        return

    from preprocessing import FEATURE_DIM
    dummy_seq = np.random.randn(20, FEATURE_DIM).astype(np.float32)
    result = predictor.predict(
        dummy_seq, camera_id="cam01", zone_id="main_entry"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    demo()
