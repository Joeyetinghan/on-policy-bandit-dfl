"""Fractional exact-cardinality top-k benchmark with min-cost evaluation semantics."""

from __future__ import annotations

import numpy as np

from src.envs.base_env import BaseEnv
from src.envs.datagen import DataGenerator


class TopKEnv(BaseEnv):
    """Contextual benchmark over the fractional exact-cardinality top-k polytope."""

    objective_sense = "min"
    objective_name = "cost"

    def __init__(self, config, seed):
        super().__init__(config, seed)
        self.p = config["p"]
        self.d = config["d"]
        self.k = float(config["k"])
        self.feedback_mode = str(config.get("feedback_mode", "bandit")).lower()
        if self.feedback_mode not in {"bandit", "semi_bandit", "full_feedback"}:
            raise ValueError("TopKEnv feedback_mode must be one of {'bandit', 'semi_bandit', 'full_feedback'}")
        self.data_generator_family = str(config.get("data_generator_family", "current_global_poly")).lower()
        if self.data_generator_family == "current_global_poly":
            self.datagen = DataGenerator(
                p=self.p,
                q=self.d,
                deg=config["deg"],
                eps_bar=config["eps_bar"],
                seed=seed,
            )
        else:
            raise ValueError("TopKEnv data_generator_family must be 'current_global_poly'")
        self.current_context = None
        self.current_c_t = None
        self.current_r_t = None
        self._last_step_eval_info = None

    def _validate_action(self, action):
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape != (self.d,):
            raise ValueError(f"Expected action shape {(self.d,)}, got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains non-finite values")
        if np.any(action < -1.0e-8) or np.any(action > 1.0 + 1.0e-8):
            raise ValueError("Top-k action must satisfy 0 <= w <= 1")
        action = np.clip(action, 0.0, 1.0)
        if not np.isclose(float(np.sum(action)), self.k, atol=1.0e-8):
            raise ValueError("Top-k action must satisfy the exact-cardinality constraint sum_i w_i = k")
        return action

    def _refresh_round_state(self):
        self.current_context = self.datagen.generate_context()
        self.current_c_t = self.datagen.get_latent_vec(self.current_context)
        # Algorithms still consume a maximize-score interface.
        self.current_r_t = -self.current_c_t

    def reset(self):
        self.t = 0
        self._last_step_eval_info = None
        self._refresh_round_state()
        return self.current_context

    def step(self, action):
        action_used = self._validate_action(action)
        c_t = self.current_c_t
        score_t = self.current_r_t
        realized_cost = self.datagen.get_reward(c_t, action_used)
        reward = -realized_cost
        eval_info = {
            "objective_vector": c_t.copy(),
            "objective_value": float(realized_cost),
            "expected_objective": float(c_t @ action_used),
            "objective_sense": self.objective_sense,
            "objective_name": self.objective_name,
            "action_taken": action_used.copy(),
        }
        self._last_step_eval_info = eval_info
        self.t += 1

        if self.t < self.T:
            self._refresh_round_state()

        if self.feedback_mode == "bandit":
            info = {"action_taken": action_used.copy()}
        elif self.feedback_mode == "semi_bandit":
            observation_mask = (action_used > 1.0e-8).astype(float)
            info = {
                "action_taken": action_used.copy(),
                "observation_mask": observation_mask,
                "observed_score_vector": score_t.copy() * observation_mask,
                "observed_objective_vector": c_t.copy() * observation_mask,
            }
        else:
            info = {
                "r_t": score_t.copy(),
                "score_t": score_t.copy(),
                "c_t": c_t.copy(),
                "objective_vector": c_t.copy(),
                "objective_value": float(realized_cost),
                "expected_objective": float(c_t @ action_used),
                "objective_sense": self.objective_sense,
                "objective_name": self.objective_name,
                "action_taken": action_used.copy(),
            }
        return reward, info

    def get_resolved_config_updates(self):
        return {"p": self.p, "d": self.d}

    def get_context(self):
        return self.current_context

    def get_step_eval_info(self):
        if self._last_step_eval_info is None:
            return None
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in self._last_step_eval_info.items()
        }
