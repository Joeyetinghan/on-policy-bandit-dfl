"""Baseline algorithms for contextual optimization experiments."""

from __future__ import annotations

import numpy as np

from src.algos.base_algo import BaseAlgo


class RandomOracleAlgo(BaseAlgo):
    """Random-score feasible baseline."""

    def __init__(self, config, oracle):
        super().__init__(config, oracle)
        self.rng = np.random.RandomState(config.get("seed", 42))
        self.q = getattr(oracle, "d", None)
        if self.q is None:
            self.q = getattr(oracle, "q")

    def select_action(self, context, oracle_context=None, true_reward=None):
        del context, true_reward
        score = self.rng.randn(self.q)
        return self.oracle.solve(score, oracle_context=oracle_context)

    def update(self, context, action, reward, info=None) -> None:
        del context, action, reward, info


class TrueModelAlgo(BaseAlgo):
    """Offline-opt per-round oracle baseline using the true reward vector."""

    def __init__(self, config, oracle):
        super().__init__(config, oracle)
        self.uses_true_reward = True

    def select_action(self, context, oracle_context=None, true_reward=None):
        del context
        if true_reward is None:
            raise ValueError("TrueModelAlgo requires true_reward for action selection")
        return self.oracle.solve(true_reward, oracle_context=oracle_context)

    def update(self, context, action, reward, info=None) -> None:
        del context, action, reward, info
