"""
evaluate_both.py — CrowdGuard AI
=====================================
Evaluate GRU and LSTM on the test set in one command.
Equivalent to running evaluate.py twice then compare_models.py.

Run:
    python scripts/evaluate_both.py
"""

import logging
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).parent
_ROOT = _SCRIPTS.parent
SRC = _ROOT / "src"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] — %(message)s",
)
logger = logging.getLogger("crowdguard.evaluate_both")


def main():
    python = sys.executable
    steps = [
        ([python, str(SRC / "evaluate.py"), "--model-type", "gru"], "Evaluate GRU"),
        ([python, str(SRC / "evaluate.py"), "--model-type", "lstm"], "Evaluate LSTM"),
        ([python, str(SRC / "compare_models.py")], "Compare models"),
    ]
    for cmd, name in steps:
        logger.info(f"Running: {name}")
        r = subprocess.run(cmd, cwd=str(_ROOT))
        if r.returncode != 0:
            logger.error(f"Failed: {name}")
            sys.exit(1)
    logger.info("Evaluation complete. Check outputs/metrics/ and outputs/plots/")


if __name__ == "__main__":
    main()
