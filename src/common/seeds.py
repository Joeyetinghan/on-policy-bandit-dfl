"""Random seeding helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_python(seed: int) -> None:
    """Seed Python and NumPy RNGs."""
    random.seed(seed)
    np.random.seed(seed)


def seed_torch(seed: int) -> None:
    """Seed torch RNGs."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_seed(seed: int, include_torch: bool = True) -> None:
    """Seed Python/NumPy and optionally torch."""
    seed_python(seed)
    if include_torch:
        seed_torch(seed)
