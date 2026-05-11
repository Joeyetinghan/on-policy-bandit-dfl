"""Online contextual pricing benchmark with bandit, semi-bandit, or full feedback."""

from __future__ import annotations

import numpy as np

from src.envs.base_env import BaseEnv


def _softplus(values):
    arr = np.asarray(values, dtype=float)
    clipped = np.clip(arr, -50.0, 50.0)
    return np.log1p(np.exp(clipped))


class PricingEnv(BaseEnv):
    """Contextual pricing benchmark with a full latent demand matrix and configurable feedback."""

    objective_sense = "min"
    objective_name = "cost"

    def __init__(self, config, seed):
        super().__init__(config, seed)
        self.n_products = int(config.get("n_products", 20))
        self.num_price_levels = int(config.get("num_price_levels", 4))
        self.pricing_context_dim = int(config.get("pricing_context_dim", 25))
        self.pricing_context_degree = int(config.get("pricing_context_degree", 2))
        if self.pricing_context_degree < 1:
            raise ValueError("pricing_context_degree must be a positive integer")
        self.pricing_price_degree = int(config.get("pricing_price_degree", 2))
        if self.pricing_price_degree < 1:
            raise ValueError("pricing_price_degree must be a positive integer")
        self.promotion_budget = min(int(config.get("promotion_budget", 10)), self.n_products)
        self.pricing_num_low_levels = int(config.get("pricing_num_low_levels", 2))
        if self.pricing_num_low_levels < 1 or self.pricing_num_low_levels > self.num_price_levels:
            raise ValueError("pricing_num_low_levels must be in [1, num_price_levels]")

        self.feedback_mode = str(config.get("feedback_mode", "bandit")).lower()
        if self.feedback_mode not in {"bandit", "semi_bandit", "full_feedback"}:
            raise ValueError("PricingEnv feedback_mode must be one of {'bandit', 'semi_bandit', 'full_feedback'}")
        self.feedback_delay = int(config.get("feedback_delay", 0))
        if self.feedback_delay != 0:
            raise ValueError("PricingEnv requires feedback_delay=0")

        self.pricing_common_shock_sigma = float(config.get("pricing_common_shock_sigma", 0.1))
        if self.pricing_common_shock_sigma < 0.0:
            raise ValueError("pricing_common_shock_sigma must be nonnegative")

        # Keep exogenous context generation independent of action-dependent reward noise so
        # all algorithms see the same pricing instance stream under a fixed seed.
        self.context_rng = np.random.RandomState(seed + 1009)
        self.shock_rng = np.random.RandomState(seed + 2003)
        self.demand_rng = np.random.RandomState(seed + 3001)

        self.normalized_price_grid = np.linspace(0.1, 0.9, self.num_price_levels, dtype=float)
        self.d = self.n_products * self.num_price_levels
        self.p = 2 * self.pricing_context_dim + self.num_price_levels + 6

        raw_directions = self.rng.randn(self.n_products, self.pricing_context_dim) / np.sqrt(
            float(self.pricing_context_dim)
        )
        self.product_directions = self._normalize_rows(raw_directions)
        self.price_response = (1.0 - self.normalized_price_grid) ** self.pricing_price_degree
        self.low_level_mask = np.zeros(self.num_price_levels, dtype=float)
        self.low_level_mask[: self.pricing_num_low_levels] = 1.0
        self.low_option_mask = np.tile(self.low_level_mask, self.n_products)

        self.current_context = None
        self.current_x_t = None
        self.current_mu_t = None
        self.current_c_t = None
        self.current_r_t = None
        self.current_mean_demand = None
        self.current_dot = None
        self.current_context_term = None
        self.current_shocked_mean_demand = None
        self.current_demand_matrix = None
        self.current_common_shock = None
        self._last_step_eval_info = None

    @staticmethod
    def _normalize_rows(matrix):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1.0e-12, None)

    def _map_context(self, x_t, dot, context_term):
        price_levels = np.tile(self.normalized_price_grid, self.n_products)
        price_term = np.tile(self.price_response, self.n_products)
        low_indicator = self.low_option_mask
        repeated_dot = np.repeat(dot, self.num_price_levels)
        repeated_context_term = np.repeat(context_term, self.num_price_levels)

        x_block = np.repeat(x_t[None, :], self.d, axis=0)
        direction_block = np.repeat(self.product_directions, self.num_price_levels, axis=0)
        price_onehot = np.tile(np.eye(self.num_price_levels, dtype=float), (self.n_products, 1))
        bias = np.ones((self.d, 1), dtype=float)

        return np.concatenate(
            [
                x_block,
                direction_block,
                repeated_dot[:, None],
                repeated_context_term[:, None],
                price_levels[:, None],
                price_term[:, None],
                low_indicator[:, None],
                price_onehot,
                bias,
            ],
            axis=1,
        )

    def _expected_mean_demand(self, x_t):
        dot = self.product_directions @ x_t
        context_term = np.abs(dot) ** self.pricing_context_degree
        base = context_term[:, None] + self.price_response[None, :]
        mean_demand = np.maximum(_softplus(base), 1.0e-6)
        return dot, context_term, mean_demand

    def _refresh_round_state(self):
        x_t = self.context_rng.randn(self.pricing_context_dim)
        dot, context_term, mean_demand = self._expected_mean_demand(x_t)
        common_shock = self._sample_common_shock()
        shocked_mean_demand = mean_demand * common_shock
        demand_matrix = self.demand_rng.poisson(shocked_mean_demand).astype(float)
        revenue_t = (self.normalized_price_grid[None, :] * mean_demand).reshape(-1)
        cost_t = -revenue_t

        self.current_x_t = x_t
        self.current_dot = dot
        self.current_context_term = context_term
        self.current_mean_demand = mean_demand
        self.current_shocked_mean_demand = shocked_mean_demand
        self.current_demand_matrix = demand_matrix
        self.current_common_shock = common_shock
        self.current_mu_t = revenue_t
        self.current_r_t = revenue_t
        self.current_c_t = cost_t
        self.current_context = self._map_context(x_t, dot, context_term)

    def _sample_common_shock(self):
        if self.pricing_common_shock_sigma <= 0.0:
            return 1.0
        zeta_t = float(self.shock_rng.randn())
        sigma = self.pricing_common_shock_sigma
        return float(np.exp(sigma * zeta_t - 0.5 * (sigma**2)))

    def _validate_action(self, action):
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape != (self.d,):
            raise ValueError(f"Expected action shape {(self.d,)}, got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains non-finite values")

        rounded = np.rint(action)
        if not np.allclose(action, rounded, atol=1.0e-6):
            raise ValueError("Pricing action must be binary")
        rounded = rounded.astype(float)
        if np.any((rounded < -1.0e-8) | (rounded > 1.0 + 1.0e-8)):
            raise ValueError("Pricing action must satisfy 0 <= y <= 1")

        matrix = rounded.reshape(self.n_products, self.num_price_levels)
        row_sums = matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1.0e-6):
            raise ValueError("Pricing action must choose exactly one price per product")

        low_count = float(np.sum(rounded * self.low_option_mask))
        if low_count > self.promotion_budget + 1.0e-6:
            raise ValueError("Pricing action violates the promotion budget")
        return rounded

    def reset(self):
        self.t = 0
        self._last_step_eval_info = None
        self._refresh_round_state()
        return self.current_context

    def step(self, action):
        action_used = self._validate_action(action)
        chosen_matrix = action_used.reshape(self.n_products, self.num_price_levels)
        chosen_levels = np.argmax(chosen_matrix, axis=1)

        realized_demand = self.current_demand_matrix[np.arange(self.n_products), chosen_levels]
        realized_revenue = self.normalized_price_grid[chosen_levels] * realized_demand
        total_revenue = float(np.sum(realized_revenue))
        realized_cost = -total_revenue
        expected_cost = float(self.current_c_t @ action_used)
        realized_revenue_vector = (
            self.normalized_price_grid[None, :] * self.current_demand_matrix
        ).reshape(-1)
        realized_cost_vector = -realized_revenue_vector

        self._last_step_eval_info = {
            "objective_vector": self.current_c_t.copy(),
            "objective_value": realized_cost,
            "expected_objective": expected_cost,
            "objective_sense": self.objective_sense,
            "objective_name": self.objective_name,
            "chosen_levels": chosen_levels.copy(),
            "realized_demand": np.asarray(realized_demand, dtype=float),
            "realized_revenue": realized_revenue.copy(),
            "realized_cost": realized_cost,
            "realized_score_vector": realized_revenue_vector.copy(),
            "realized_objective_vector": realized_cost_vector.copy(),
            "latent_demand_matrix": self.current_demand_matrix.copy(),
            "common_shock": float(self.current_common_shock),
        }

        self.t += 1
        if self.t < self.T:
            self._refresh_round_state()

        if self.feedback_mode == "bandit":
            info = {"action_taken": action_used.copy()}
        elif self.feedback_mode == "semi_bandit":
            observation_mask = action_used.copy()
            info = {
                "action_taken": action_used.copy(),
                "observation_mask": observation_mask,
                "observed_score_vector": realized_revenue_vector.copy() * observation_mask,
                "observed_objective_vector": realized_cost_vector.copy() * observation_mask,
            }
        else:
            info = {
                "r_t": realized_revenue_vector.copy(),
                "score_t": realized_revenue_vector.copy(),
                "c_t": realized_cost_vector.copy(),
                "objective_vector": realized_cost_vector.copy(),
                "objective_value": realized_cost,
                "expected_objective": expected_cost,
                "objective_sense": self.objective_sense,
                "objective_name": self.objective_name,
                "action_taken": action_used.copy(),
                "chosen_levels": chosen_levels.copy(),
                "realized_demand": np.asarray(realized_demand, dtype=float),
                "realized_revenue": realized_revenue.copy(),
            }

        return total_revenue, info

    def get_context(self):
        return self.current_context

    def get_true_objective_vector(self):
        if self.current_r_t is None:
            return None
        return self.current_r_t.copy()

    def get_step_eval_info(self):
        if self._last_step_eval_info is None:
            return None
        out = {}
        for key, value in self._last_step_eval_info.items():
            out[key] = value.copy() if isinstance(value, np.ndarray) else value
        return out

    def get_resolved_config_updates(self):
        return {"p": self.p, "d": self.d}
