"""Common interface for contextual bandit optimization algorithms."""

from __future__ import annotations

import abc
from typing import Any

import numpy as np


class BaseAlgo(abc.ABC):
    """Base class for bandit algorithms."""

    def __init__(self, config, oracle):
        self.config = config
        self.oracle = oracle
        self.total_rounds = max(1, int(config.get("T", 1)))
        self._oracle_solution_cache = {}

    def _init_random_exploration(self, config, rng, q) -> None:
        epsilon = float(config.get("exploration_epsilon", 0.0))
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("exploration_epsilon must be in [0, 1]")
        self.exploration_epsilon = epsilon
        self._exploration_rng = rng
        self._exploration_q = int(q)
        self.last_random_exploration = False

    def _sample_action_score(self, default_score):
        use_random = self.exploration_epsilon > 0.0 and self._exploration_rng.rand() < self.exploration_epsilon
        self.last_random_exploration = use_random
        if use_random:
            return self._exploration_rng.randn(self._exploration_q)
        return default_score

    def _is_final_round(self) -> bool:
        return self.t_curr >= self.total_rounds

    def _reset_oracle_solution_cache(self) -> None:
        self._oracle_solution_cache = {}

    def _oracle_cache_key(self, score: Any, oracle_context=None):
        if hasattr(score, "detach") and hasattr(score, "cpu"):
            score_arr = score.detach().cpu().numpy()
        else:
            score_arr = np.asarray(score)
        score_arr = np.ascontiguousarray(np.asarray(score_arr, dtype=np.float64))
        return (tuple(score_arr.shape), score_arr.tobytes(), id(oracle_context))

    def _cached_oracle_solve(self, score: Any, oracle_context=None):
        key = self._oracle_cache_key(score, oracle_context=oracle_context)
        cached = self._oracle_solution_cache.get(key)
        if cached is None:
            raw_solved = self.oracle.solve(score, oracle_context=oracle_context)
            if hasattr(raw_solved, "linear_observation_matrix"):
                solved = raw_solved.copy()
            else:
                solved = np.asarray(raw_solved, dtype=float).copy()
            self._oracle_solution_cache[key] = solved
            return solved.copy()
        return cached.copy()

    def _extract_linear_observation_feedback(self, info, *, objective_values: bool = False):
        """Return linear observation matrix and values when semi-bandit feedback supplies them."""
        if info is None:
            return None, None
        matrix = info.get("linear_observation_matrix")
        values_key = "linear_observed_objective_values" if objective_values else "linear_observation_values"
        values = info.get(values_key)
        if matrix is None or values is None:
            return None, None
        matrix_arr = np.asarray(matrix, dtype=float)
        if matrix_arr.ndim == 1:
            matrix_arr = matrix_arr.reshape(1, -1)
        values_arr = np.asarray(values, dtype=float).reshape(-1)
        if matrix_arr.ndim != 2:
            raise ValueError(f"linear_observation_matrix must be 2D, got shape {matrix_arr.shape}")
        if matrix_arr.shape[0] != values_arr.shape[0]:
            raise ValueError(
                "linear_observation_matrix row count must match linear_observation_values length, "
                f"got {matrix_arr.shape[0]} and {values_arr.shape[0]}"
            )
        return matrix_arr, values_arr

    def _extract_observed_score_feedback(self, info):
        """Return observed score components and their mask for semi/full feedback when available."""
        if info is None:
            return None, None
        observed = info.get("observed_score_vector")
        mask = info.get("observation_mask")
        if observed is None and "r_t" in info:
            observed_arr = np.asarray(info["r_t"], dtype=float).reshape(-1)
            return observed_arr, np.ones_like(observed_arr)
        if observed is None or mask is None:
            return None, None
        observed_arr = np.asarray(observed, dtype=float).reshape(-1)
        mask_arr = np.asarray(mask, dtype=float).reshape(-1)
        return observed_arr, mask_arr

    @abc.abstractmethod
    def select_action(self, context: Any, oracle_context=None, true_reward=None):
        """Return an action for the current context."""

    @abc.abstractmethod
    def update(self, context: Any, action, reward: float, info=None) -> None:
        """Update internal state from scalar bandit feedback."""
