"""Base environment interface for contextual optimization benchmarks."""

from __future__ import annotations

import abc

import numpy as np


class BaseEnv(abc.ABC):
    """Abstract base class for contextual optimization environments."""

    def __init__(self, config, seed):
        self.config = config
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.t = 0
        self.T = config["T"]

    @abc.abstractmethod
    def reset(self):
        """Reset state and return the initial context."""

    @abc.abstractmethod
    def step(self, action):
        """Apply an action and return scalar bandit feedback plus auxiliary info."""

    @abc.abstractmethod
    def get_context(self):
        """Return the current context."""

    def get_oracle_context(self):
        """Return optional per-round metadata required by the oracle."""
        return None

    def get_true_objective_vector(self):
        """Return the current oracle-score vector used by score-maximizing solvers."""
        return getattr(self, "current_r_t", None)

    def get_step_eval_info(self):
        """Return evaluator-only info for the most recent completed step."""
        return None

    def get_resolved_config_updates(self):
        """Return environment-derived config fields such as resolved dimensions."""
        return {}

    def sample_eval_instance(self):
        """Draw one independent evaluation sample from the environment's data source."""
        context = self.reset()
        score_vector = self.get_true_objective_vector()
        if context is None or score_vector is None:
            raise RuntimeError("Environment must expose context and true objective vectors for evaluation sampling.")
        oracle_context = self.get_oracle_context()
        context_arr = np.asarray(context, dtype=float).copy()
        score_arr = np.asarray(score_vector, dtype=float).reshape(-1).copy()
        if isinstance(oracle_context, np.ndarray):
            oracle_context = oracle_context.copy()
        return {
            "context": context_arr,
            "score_vector": score_arr,
            "oracle_context": oracle_context,
        }
