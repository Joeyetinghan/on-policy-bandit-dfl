"""Shortest-path benchmark with online contextual costs.

The default polynomial family uses the calibrated shortest-path polynomial generator.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from src.envs.base_env import BaseEnv
from src.envs.datagen import ShortestPathCalibratedPolyGenerator


class ShortestPathEnv(BaseEnv):
    """Contextual shortest-path benchmark on a fixed right/down grid DAG."""

    objective_sense = "min"
    objective_name = "cost"

    def __init__(self, config, seed):
        super().__init__(config, seed)
        self.p = config["p"]
        self.grid_size = int(config.get("grid_size", 5))
        self.n_nodes = self.grid_size * self.grid_size
        self.feedback_mode = str(config.get("feedback_mode", "bandit")).lower()
        if self.feedback_mode not in {"bandit", "semi_bandit", "full_feedback"}:
            raise ValueError("ShortestPathEnv feedback_mode must be one of {'bandit', 'semi_bandit', 'full_feedback'}")

        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(self.n_nodes))

        edges = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                node = row * self.grid_size + col
                if col < self.grid_size - 1:
                    edges.append((node, row * self.grid_size + (col + 1)))
                if row < self.grid_size - 1:
                    edges.append((node, (row + 1) * self.grid_size + col))

        self.G.add_edges_from(edges)
        self.edges = list(self.G.edges())
        self.edge_to_idx = {edge: idx for idx, edge in enumerate(self.edges)}
        self.d = len(self.edges)
        expected_edges = 2 * self.grid_size * (self.grid_size - 1)
        if self.d != expected_edges:
            raise ValueError(f"Expected {expected_edges} edges for {self.grid_size}x{self.grid_size} grid, got {self.d}")

        self.A_eq = np.zeros((self.n_nodes, self.d))
        self.b_eq = np.zeros(self.n_nodes)
        self.source = 0
        self.sink = self.n_nodes - 1
        self.b_eq[self.source] = -1.0
        self.b_eq[self.sink] = 1.0
        for idx, (u, v) in enumerate(self.edges):
            self.A_eq[u, idx] -= 1.0
            self.A_eq[v, idx] += 1.0

        self.data_generator_family = str(config.get("data_generator_family", "shortest_path_calibrated_poly")).lower()
        if self.data_generator_family == "shortest_path_calibrated_poly":
            self.datagen = ShortestPathCalibratedPolyGenerator(
                p=self.p,
                q=self.d,
                deg=config["deg"],
                eps_bar=config["eps_bar"],
                poly_offset=float(config.get("poly_offset", 1.0)),
                poly_scale=float(config.get("poly_scale", 0.7)),
                seed=seed,
            )
        else:
            raise ValueError("ShortestPathEnv data_generator_family must be 'shortest_path_calibrated_poly'")
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
            raise ValueError("Shortest-path action must satisfy 0 <= w <= 1")
        residual = self.A_eq @ action - self.b_eq
        if float(np.max(np.abs(residual))) > 1.0e-6:
            raise ValueError("Shortest-path action violates unit-flow conservation constraints")
        return np.clip(action, 0.0, 1.0)

    def _refresh_round_state(self):
        self.current_context = self.datagen.generate_context()
        self.current_c_t = self.datagen.get_latent_vec(self.current_context)
        # Algorithms are still written against a maximize-score interface.
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

    def get_context(self):
        return self.current_context

    def get_step_eval_info(self):
        if self._last_step_eval_info is None:
            return None
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in self._last_step_eval_info.items()
        }

    def get_resolved_config_updates(self):
        return {"p": self.p, "d": self.d}
