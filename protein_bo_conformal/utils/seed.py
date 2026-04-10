"""Random seed management helpers."""

from __future__ import annotations

import os
import random
from typing import Any


def set_global_seed(seed: int, deterministic: bool = True, logger: Any | None = None) -> dict[str, Any]:
    """Seed every available randomness source and return a seed report."""
    report: dict[str, Any] = {
        "seed": seed,
        "python_random": True,
        "numpy": False,
        "torch": False,
        "torch_deterministic": False,
    }

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
        report["numpy"] = True
    except Exception:
        report["numpy"] = False

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        report["torch"] = True

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            report["torch_deterministic"] = True
    except Exception:
        report["torch"] = False
        report["torch_deterministic"] = False

    if logger is not None:
        logger.info("Seed report: %s", report)
    return report
