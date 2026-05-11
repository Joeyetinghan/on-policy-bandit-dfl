"""Shared round-batch accumulation helpers for trainer updates."""

from __future__ import annotations

import numpy as np
from torch.nn.utils import clip_grad_norm_


class RoundBatchAccumulator:
    """Accumulate round-level gradient contributions until a batch is ready."""

    def __init__(self, batch_rounds: int):
        self.batch_rounds = max(1, int(batch_rounds))
        self.count = 0
        self._numpy_sum = None

    def add_numpy(self, grad) -> None:
        grad_arr = np.asarray(grad, dtype=float)
        if self._numpy_sum is None:
            self._numpy_sum = grad_arr.copy()
        else:
            self._numpy_sum += grad_arr
        self.count += 1

    def mark_round(self) -> None:
        self.count += 1

    def should_flush(self, *, is_final_round: bool) -> bool:
        return self.count > 0 and (self.count >= self.batch_rounds or is_final_round)

    def consume_numpy_mean(self):
        if self.count <= 0 or self._numpy_sum is None:
            self.reset()
            return None
        mean_grad = self._numpy_sum / float(self.count)
        self.reset()
        return mean_grad

    def average_torch_grads_(self, params) -> int:
        if self.count <= 0:
            return 0
        scale = 1.0 / float(self.count)
        for param in params:
            if param.grad is not None:
                param.grad.mul_(scale)
        batch_count = self.count
        self.reset()
        return batch_count

    def reset(self) -> None:
        self.count = 0
        self._numpy_sum = None


def clip_torch_grads(params, max_norm) -> None:
    """Clip torch gradients for any parameters that currently have grads."""
    if max_norm is None:
        return
    params_with_grads = [param for param in params if param.grad is not None]
    if params_with_grads:
        clip_grad_norm_(params_with_grads, float(max_norm))
