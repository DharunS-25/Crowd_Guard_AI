"""
reproducibility.py — CrowdGuard AI
===================================
Centralised seed management and determinism helpers.
Call set_seed() at the start of every training run.
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms (may slow training slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch not installed yet — handled at training time


def get_device() -> "torch.device":
    """Return the best available device (CUDA > CPU)."""
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] CUDA not available — using CPU.")
    return device


def log_environment(logger=None) -> dict:
    """Log key environment details for reproducibility record."""
    import torch
    import platform

    info = {
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        ),
        "numpy_version": np.__version__,
    }

    try:
        import sklearn
        info["sklearn_version"] = sklearn.__version__
    except ImportError:
        info["sklearn_version"] = "not installed"

    msg = "\n".join(f"  {k}: {v}" for k, v in info.items())
    if logger:
        logger.info(f"Environment:\n{msg}")
    else:
        print(f"[Environment]\n{msg}")

    return info
