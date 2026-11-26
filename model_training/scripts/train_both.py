"""
train_both.py — CrowdGuard AI
================================
Master script that runs the full pipeline:

  1. Validates processed data exists
  2. Trains GRU model
  3. Trains LSTM model
  4. Evaluates both on held-out test set
  5. Generates comparison report

Run:
    python scripts/train_both.py

Prerequisites:
    python scripts/inspect_dataset.py  (must run first)
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
import time

_SCRIPTS = Path(__file__).parent
_ROOT = _SCRIPTS.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crowdguard.train_both")


def run(cmd: list, name: str) -> bool:
    logger.info(f"\n{'='*60}")
    logger.info(f"  Running: {name}")
    logger.info(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(_ROOT), capture_output=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        logger.error(f"FAILED: {name} (exit code {result.returncode})")
        return False
    logger.info(f"  Done: {name} ({elapsed:.1f}s)")
    return True


def main():
    data_dir = _ROOT / "data" / "processed"

    # Check preprocessed data
    for split in ["train", "val", "test"]:
        xp = data_dir / f"X_{split}.npy"
        if not xp.exists():
            logger.error(
                f"Processed data not found: {xp}\n"
                "Run first: python scripts/inspect_dataset.py"
            )
            sys.exit(1)

    src_dir = _ROOT / "src"
    python = sys.executable

    steps = [
        ([python, str(src_dir / "train_gru.py")], "Train GRU model"),
        ([python, str(src_dir / "train_lstm.py")], "Train LSTM model"),
        ([python, str(src_dir / "evaluate.py"), "--model-type", "gru"], "Evaluate GRU"),
        ([python, str(src_dir / "evaluate.py"), "--model-type", "lstm"], "Evaluate LSTM"),
        ([python, str(src_dir / "compare_models.py")], "Compare GRU vs LSTM"),
    ]

    for cmd, name in steps:
        ok = run(cmd, name)
        if not ok:
            logger.error(f"Pipeline stopped at: {name}")
            sys.exit(1)

    logger.info("\n" + "="*60)
    logger.info("  Full training pipeline completed successfully!")
    logger.info(f"  Results in: {_ROOT / 'outputs'}")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    main()
