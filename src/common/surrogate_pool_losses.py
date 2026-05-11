"""Point-model surrogate losses built on a cached solution pool.

References:
- PyEPO `SPOPlus` in `pkg/pyepo/func/surrogate.py`
- PyEPO ranking losses in `pkg/pyepo/func/rank.py`
- PyEPO contrastive losses in `pkg/pyepo/func/contrastive.py`
- PredOpt's cached-solution benchmark pattern in `predopt-benchmarks`

This module adapts the point-model ranking/contrastive losses to the repo's
maximize-score interface. Benchmarks with native minimization objectives are
converted to score vectors before reaching these helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


POOL_SURROGATE_LOSSES = {"pointwise_ltr", "pairwise_ltr", "listwise_ltr", "nce", "map"}


@dataclass
class SolutionPool:
    """FIFO set of unique feasible actions used by ranking/contrastive losses."""

    q: int
    max_size: int = 128
    decimals: int = 8
    _keys: list[tuple[float, ...]] = field(default_factory=list)
    _sols: dict[tuple[float, ...], np.ndarray] = field(default_factory=dict)

    def _key(self, sol: np.ndarray) -> tuple[float, ...]:
        arr = np.asarray(sol, dtype=float).reshape(self.q)
        return tuple(np.round(arr, self.decimals))

    def add(self, sol: np.ndarray) -> None:
        arr = np.asarray(sol, dtype=float).reshape(self.q).copy()
        key = self._key(arr)
        if key in self._sols:
            self._keys.remove(key)
        self._sols[key] = arr
        self._keys.append(key)
        while len(self._keys) > self.max_size:
            stale = self._keys.pop(0)
            self._sols.pop(stale, None)

    def add_many(self, sols) -> None:
        for sol in sols:
            self.add(sol)

    def as_array(self) -> np.ndarray:
        if not self._keys:
            return np.zeros((0, self.q), dtype=float)
        return np.vstack([self._sols[key] for key in self._keys])


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_shifted = np.exp(np.clip(shifted, -50.0, 50.0))
    denom = np.sum(exp_shifted)
    if not np.isfinite(denom) or denom <= 0.0:
        return np.full_like(exp_shifted, 1.0 / max(exp_shifted.size, 1))
    return exp_shifted / denom


def _pointwise_grad(pred_obj: np.ndarray, true_obj: np.ndarray, solpool: np.ndarray) -> np.ndarray:
    coeffs = 2.0 * (pred_obj - true_obj) / max(len(solpool), 1)
    return coeffs @ solpool


def _pairwise_grad(pred_obj: np.ndarray, true_obj: np.ndarray, solpool: np.ndarray, margin: float) -> np.ndarray:
    if len(solpool) <= 1:
        return np.zeros(solpool.shape[1], dtype=float)
    best_idx = int(np.argmax(true_obj))
    mask = np.ones(len(solpool), dtype=bool)
    mask[best_idx] = False
    rest_obj = pred_obj[mask]
    margins = rest_obj - pred_obj[best_idx] + float(margin)
    active = margins > 0.0
    if not np.any(active):
        return np.zeros(solpool.shape[1], dtype=float)
    coeffs = np.zeros(len(solpool), dtype=float)
    rest_indices = np.flatnonzero(mask)
    coeffs[rest_indices[active]] = 1.0 / max(len(rest_obj), 1)
    coeffs[best_idx] = -float(np.sum(active)) / max(len(rest_obj), 1)
    return coeffs @ solpool


def _listwise_grad(pred_obj: np.ndarray, true_obj: np.ndarray, solpool: np.ndarray, temperature: float) -> np.ndarray:
    tau = max(float(temperature), 1.0e-8)
    pred_probs = _softmax(pred_obj / tau)
    true_probs = _softmax(true_obj / tau)
    coeffs = (pred_probs - true_probs) / tau
    return coeffs @ solpool


def _nce_grad(solpool: np.ndarray, true_sol: np.ndarray) -> np.ndarray:
    return np.mean(solpool, axis=0) - true_sol


def _map_grad(pred_obj: np.ndarray, solpool: np.ndarray, true_sol: np.ndarray) -> np.ndarray:
    best_idx = int(np.argmax(pred_obj))
    return solpool[best_idx] - true_sol


def pool_surrogate_grad(
    *,
    loss_type: str,
    pred_vec,
    target_vec,
    oracle_solve,
    solution_pool: SolutionPool,
    oracle_context=None,
    pairwise_margin: float = 0.0,
    listwise_temperature: float = 1.0,
) -> np.ndarray:
    """Gradient of a pool-based surrogate w.r.t. the predicted score vector."""

    pred = np.asarray(pred_vec, dtype=float).reshape(-1)
    target = np.asarray(target_vec, dtype=float).reshape(-1)
    if pred.shape != target.shape:
        raise ValueError(f"pred_vec and target_vec must have the same shape, got {pred.shape} vs {target.shape}")

    true_sol = np.asarray(oracle_solve(target, oracle_context=oracle_context), dtype=float).reshape(-1)
    pred_sol = np.asarray(oracle_solve(pred, oracle_context=oracle_context), dtype=float).reshape(-1)
    solution_pool.add_many([true_sol, pred_sol])
    solpool = solution_pool.as_array()
    if solpool.size == 0:
        return np.zeros_like(pred)

    pred_obj = solpool @ pred
    true_obj = solpool @ target

    if loss_type == "pointwise_ltr":
        return _pointwise_grad(pred_obj, true_obj, solpool)
    if loss_type == "pairwise_ltr":
        return _pairwise_grad(pred_obj, true_obj, solpool, pairwise_margin)
    if loss_type == "listwise_ltr":
        return _listwise_grad(pred_obj, true_obj, solpool, listwise_temperature)
    if loss_type == "nce":
        return _nce_grad(solpool, true_sol)
    if loss_type == "map":
        return _map_grad(pred_obj, solpool, true_sol)
    raise ValueError(f"Unknown pool surrogate loss_type: {loss_type}")
