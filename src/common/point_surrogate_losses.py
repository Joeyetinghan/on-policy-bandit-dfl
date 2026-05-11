"""Shared point-model surrogate gradients.

References:
- PyEPO `pkg/pyepo/func/blackbox.py`
- PyEPO `pkg/pyepo/func/perturbed.py`
- PyEPO `pkg/pyepo/func/surrogate.py`
- PredOpt `ShortestPath/Trainer/CacheLosses.py`
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.common.surrogate_loss_names import CANONICAL_POINT_LOSS_TYPES, canonical_surrogate_loss_name
from src.common.surrogate_pool_losses import POOL_SURROGATE_LOSSES, SolutionPool, pool_surrogate_grad


POINT_SURROGATE_LOSSES = set(CANONICAL_POINT_LOSS_TYPES)
NON_GENERATIVE_POINT_SURROGATE_LOSSES = POINT_SURROGATE_LOSSES - {
    "mse",
    "spo_plus",
    "weighted_mse_spo_plus",
}


@dataclass
class SumGammaDistribution:
    """Sum-of-Gamma noise used by PyEPO's IMLE variants."""

    kappa: float = 5.0
    n_iterations: int = 10
    seed: int = 135
    rng: np.random.RandomState = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.RandomState(self.seed)

    def sample(self, size) -> np.ndarray:
        samples = 0.0
        for idx in range(1, self.n_iterations + 1):
            samples += self.rng.gamma(1.0 / self.kappa, self.kappa / idx, size)
        samples -= np.log(self.n_iterations)
        samples /= self.kappa
        return np.asarray(samples, dtype=float)


@dataclass
class PointSurrogateState:
    """Mutable state for cached and stochastic point surrogate losses."""

    q: int
    max_pool_size: int = 128
    seed: int = 42
    aimle_alpha: float = 1.0
    aimle_grad_norm_avg: float = 1.0
    aimle_step: float = 1.0e-3
    solution_pool: SolutionPool = field(init=False)
    rng: np.random.RandomState = field(init=False)
    imle_distribution: SumGammaDistribution = field(init=False)

    def __post_init__(self) -> None:
        self.solution_pool = SolutionPool(self.q, max_size=self.max_pool_size)
        self.rng = np.random.RandomState(self.seed)
        self.imle_distribution = SumGammaDistribution(seed=self.seed + 1)


def _as_vec(value, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional after reshape")
    return arr


def _solve_score(oracle_solve, score: np.ndarray, *, oracle_context=None) -> np.ndarray:
    return _as_vec(oracle_solve(score, oracle_context=oracle_context), name="solution")


def _normal_perturbation_solutions(
    pred_score: np.ndarray,
    *,
    oracle_solve,
    state: PointSurrogateState,
    oracle_context,
    num_samples: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    noises = state.rng.normal(0.0, 1.0, size=(num_samples, pred_score.size))
    sols = np.vstack(
        [
            _solve_score(oracle_solve, pred_score - sigma * noise, oracle_context=oracle_context)
            for noise in noises
        ]
    )
    return noises, sols


def _imle_perturbation_solutions(
    pred_score: np.ndarray,
    *,
    oracle_solve,
    state: PointSurrogateState,
    oracle_context,
    num_samples: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    noises = state.imle_distribution.sample(size=(num_samples, pred_score.size))
    sols = np.vstack(
        [
            _solve_score(oracle_solve, pred_score - sigma * noise, oracle_context=oracle_context)
            for noise in noises
        ]
    )
    return noises, sols


def _spo_plus_grad(
    pred_score: np.ndarray,
    target_score: np.ndarray,
    *,
    oracle_solve,
    oracle_context,
    target_sol: np.ndarray,
) -> np.ndarray:
    shifted_sol = _solve_score(oracle_solve, 2.0 * pred_score - target_score, oracle_context=oracle_context)
    return 2.0 * (shifted_sol - target_sol)


def _dbb_grad(
    pred_score: np.ndarray,
    target_score: np.ndarray,
    *,
    oracle_solve,
    oracle_context,
    dbb_lambda: float,
) -> np.ndarray:
    pred_sol = _solve_score(oracle_solve, pred_score, oracle_context=oracle_context)
    shifted_sol = _solve_score(oracle_solve, pred_score + dbb_lambda * target_score, oracle_context=oracle_context)
    return (pred_sol - shifted_sol) / (dbb_lambda + 1.0e-7)


def _nid_grad(target_score: np.ndarray) -> np.ndarray:
    return -target_score


def _dpo_grad(
    pred_score: np.ndarray,
    target_score: np.ndarray,
    *,
    oracle_solve,
    state: PointSurrogateState,
    oracle_context,
    num_samples: int,
    sigma: float,
) -> np.ndarray:
    noises, perturbed_sols = _normal_perturbation_solutions(
        pred_score,
        oracle_solve=oracle_solve,
        state=state,
        oracle_context=oracle_context,
        num_samples=num_samples,
        sigma=sigma,
    )
    coeffs = perturbed_sols @ target_score
    return np.mean(noises * coeffs[:, None], axis=0) / (sigma + 1.0e-7)


def _pfyl_grad(
    pred_score: np.ndarray,
    *,
    oracle_solve,
    state: PointSurrogateState,
    oracle_context,
    target_sol: np.ndarray,
    num_samples: int,
    sigma: float,
) -> np.ndarray:
    _, perturbed_sols = _normal_perturbation_solutions(
        pred_score,
        oracle_solve=oracle_solve,
        state=state,
        oracle_context=oracle_context,
        num_samples=num_samples,
        sigma=sigma,
    )
    expected_sol = np.mean(perturbed_sols, axis=0)
    return expected_sol - target_sol


def _imle_grad(
    pred_score: np.ndarray,
    target_score: np.ndarray,
    *,
    oracle_solve,
    state: PointSurrogateState,
    oracle_context,
    imle_lambda: float,
    num_samples: int,
    sigma: float,
    two_sides: bool,
) -> np.ndarray:
    noises, base_sols = _imle_perturbation_solutions(
        pred_score,
        oracle_solve=oracle_solve,
        state=state,
        oracle_context=oracle_context,
        num_samples=num_samples,
        sigma=sigma,
    )
    pos_sols = np.vstack(
        [
            _solve_score(
                oracle_solve,
                pred_score + imle_lambda * target_score - sigma * noise,
                oracle_context=oracle_context,
            )
            for noise in noises
        ]
    )
    if two_sides:
        neg_sols = np.vstack(
            [
                _solve_score(
                    oracle_solve,
                    pred_score - imle_lambda * target_score - sigma * noise,
                    oracle_context=oracle_context,
                )
                for noise in noises
            ]
        )
        return np.mean(neg_sols - pos_sols, axis=0) / (2.0 * imle_lambda + 1.0e-7)
    return np.mean(base_sols - pos_sols, axis=0) / (imle_lambda + 1.0e-7)


def _aimle_grad(
    pred_score: np.ndarray,
    target_score: np.ndarray,
    *,
    oracle_solve,
    state: PointSurrogateState,
    oracle_context,
    num_samples: int,
    sigma: float,
    two_sides: bool,
) -> np.ndarray:
    target_norm = float(np.linalg.norm(target_score))
    pred_norm = float(np.linalg.norm(pred_score))
    if target_norm <= 0.0:
        grad = np.zeros_like(pred_score)
    else:
        imle_lambda = state.aimle_alpha * pred_norm / target_norm
        grad = _imle_grad(
            pred_score,
            target_score,
            oracle_solve=oracle_solve,
            state=state,
            oracle_context=oracle_context,
            imle_lambda=imle_lambda,
            num_samples=num_samples,
            sigma=sigma,
            two_sides=two_sides,
        )
    grad_norm = float(np.mean(np.abs(grad) > 1.0e-7))
    state.aimle_grad_norm_avg = 0.9 * state.aimle_grad_norm_avg + 0.1 * grad_norm
    if state.aimle_grad_norm_avg < 1.0:
        state.aimle_alpha += state.aimle_step
    else:
        state.aimle_alpha = max(0.0, state.aimle_alpha - state.aimle_step)
    return grad


def _pg_grad(
    pred_score: np.ndarray,
    target_score: np.ndarray,
    *,
    oracle_solve,
    oracle_context,
    pg_sigma: float,
    two_sides: bool,
) -> np.ndarray:
    if two_sides:
        plus_sol = _solve_score(oracle_solve, pred_score + pg_sigma * target_score, oracle_context=oracle_context)
        minus_sol = _solve_score(oracle_solve, pred_score - pg_sigma * target_score, oracle_context=oracle_context)
        return (minus_sol - plus_sol) / (2.0 * pg_sigma + 1.0e-7)
    pred_sol = _solve_score(oracle_solve, pred_score, oracle_context=oracle_context)
    minus_sol = _solve_score(oracle_solve, pred_score - pg_sigma * target_score, oracle_context=oracle_context)
    return (minus_sol - pred_sol) / (pg_sigma + 1.0e-7)


def _pairwise_diff_grad(pred_obj: np.ndarray, true_obj: np.ndarray, solpool: np.ndarray) -> np.ndarray:
    if len(solpool) <= 1:
        return np.zeros(solpool.shape[1], dtype=float)
    _, indices = np.unique(-true_obj, return_index=True)
    if len(indices) <= 1:
        return np.zeros(solpool.shape[1], dtype=float)
    best_idx = int(indices[0])
    other_idx = np.asarray(indices[1:], dtype=int)
    deltas = (pred_obj[best_idx] - pred_obj[other_idx]) - (true_obj[best_idx] - true_obj[other_idx])
    pair_dirs = solpool[best_idx] - solpool[other_idx]
    return 2.0 * np.mean(deltas[:, None] * pair_dirs, axis=0)


def _nce_c_grad(solpool: np.ndarray, true_sol: np.ndarray) -> np.ndarray:
    return np.mean(solpool, axis=0) - true_sol


def _map_c_grad(centered_obj: np.ndarray, solpool: np.ndarray, true_sol: np.ndarray) -> np.ndarray:
    return solpool[int(np.argmax(centered_obj))] - true_sol


def _spo_caching_grad(shifted_obj: np.ndarray, solpool: np.ndarray, true_sol: np.ndarray) -> np.ndarray:
    return solpool[int(np.argmax(shifted_obj))] - true_sol


def point_surrogate_grad(
    *,
    loss_type: str,
    pred_vec,
    target_vec,
    oracle_solve,
    state: PointSurrogateState,
    oracle_context=None,
    target_sol=None,
    mse_weight: float = 0.5,
    pairwise_margin: float = 0.0,
    listwise_temperature: float = 1.0,
    dbb_lambda: float = 10.0,
    imle_lambda: float = 10.0,
    num_samples: int = 10,
    sigma: float = 1.0,
    pg_sigma: float = 0.1,
    two_sides: bool = False,
) -> np.ndarray:
    """Gradient of a point surrogate loss with respect to the predicted score."""

    loss = canonical_surrogate_loss_name(loss_type)
    pred = _as_vec(pred_vec, name="pred_vec")
    target = _as_vec(target_vec, name="target_vec")
    if pred.shape != target.shape:
        raise ValueError(f"pred_vec and target_vec must match, got {pred.shape} vs {target.shape}")
    if pred.size != state.q:
        raise ValueError(f"pred_vec must have length {state.q}, got {pred.size}")

    if loss == "mse":
        return 2.0 * (pred - target)

    if loss in {
        "spo_plus",
        "weighted_mse_spo_plus",
        "pfyl",
        *POOL_SURROGATE_LOSSES,
        "pairwise_diff",
        "nce_c",
        "map_c",
        "spo_caching",
    }:
        if target_sol is None:
            target_sol = _solve_score(oracle_solve, target, oracle_context=oracle_context)
        else:
            target_sol = _as_vec(target_sol, name="target_sol")

    if loss == "spo_plus":
        return _spo_plus_grad(
            pred,
            target,
            oracle_solve=oracle_solve,
            oracle_context=oracle_context,
            target_sol=target_sol,
        )
    if loss == "weighted_mse_spo_plus":
        mse_grad = 2.0 * (pred - target)
        spo_grad = _spo_plus_grad(
            pred,
            target,
            oracle_solve=oracle_solve,
            oracle_context=oracle_context,
            target_sol=target_sol,
        )
        return mse_weight * mse_grad + (1.0 - mse_weight) * spo_grad
    if loss in POOL_SURROGATE_LOSSES:
        return pool_surrogate_grad(
            loss_type=loss,
            pred_vec=pred,
            target_vec=target,
            oracle_solve=oracle_solve,
            solution_pool=state.solution_pool,
            oracle_context=oracle_context,
            pairwise_margin=pairwise_margin,
            listwise_temperature=listwise_temperature,
        )
    if loss == "dbb":
        return _dbb_grad(
            pred,
            target,
            oracle_solve=oracle_solve,
            oracle_context=oracle_context,
            dbb_lambda=dbb_lambda,
        )
    if loss == "nid":
        return _nid_grad(target)
    if loss == "dpo":
        return _dpo_grad(
            pred,
            target,
            oracle_solve=oracle_solve,
            state=state,
            oracle_context=oracle_context,
            num_samples=num_samples,
            sigma=sigma,
        )
    if loss == "pfyl":
        return _pfyl_grad(
            pred,
            oracle_solve=oracle_solve,
            state=state,
            oracle_context=oracle_context,
            target_sol=target_sol,
            num_samples=num_samples,
            sigma=sigma,
        )
    if loss == "imle":
        return _imle_grad(
            pred,
            target,
            oracle_solve=oracle_solve,
            state=state,
            oracle_context=oracle_context,
            imle_lambda=imle_lambda,
            num_samples=num_samples,
            sigma=sigma,
            two_sides=two_sides,
        )
    if loss == "aimle":
        return _aimle_grad(
            pred,
            target,
            oracle_solve=oracle_solve,
            state=state,
            oracle_context=oracle_context,
            num_samples=num_samples,
            sigma=sigma,
            two_sides=two_sides,
        )
    if loss == "pg":
        return _pg_grad(
            pred,
            target,
            oracle_solve=oracle_solve,
            oracle_context=oracle_context,
            pg_sigma=pg_sigma,
            two_sides=two_sides,
        )

    pred_sol = _solve_score(oracle_solve, pred, oracle_context=oracle_context)
    state.solution_pool.add_many([target_sol, pred_sol])
    if loss == "map_c":
        centered_sol = _solve_score(oracle_solve, pred - target, oracle_context=oracle_context)
        state.solution_pool.add(centered_sol)
    if loss == "spo_caching":
        shifted_sol = _solve_score(oracle_solve, 2.0 * pred - target, oracle_context=oracle_context)
        state.solution_pool.add(shifted_sol)

    solpool = state.solution_pool.as_array()
    if solpool.size == 0:
        return np.zeros_like(pred)
    pred_obj = solpool @ pred
    true_obj = solpool @ target

    if loss == "pairwise_diff":
        return _pairwise_diff_grad(pred_obj, true_obj, solpool)
    if loss == "nce_c":
        return _nce_c_grad(solpool, target_sol)
    if loss == "map_c":
        centered_obj = solpool @ (pred - target)
        return _map_c_grad(centered_obj, solpool, target_sol)
    if loss == "spo_caching":
        shifted_obj = solpool @ (2.0 * pred - target)
        return _spo_caching_grad(shifted_obj, solpool, target_sol)

    raise ValueError(f"Unknown point surrogate loss_type: {loss}")
