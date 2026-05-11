"""Pricing oracle."""

from __future__ import annotations

import numpy as np


class PricingOracle:
    """Exact pure-pricing oracle with a promotion-budget coupling constraint."""

    def __init__(self, n_products, num_price_levels, promotion_budget, pricing_num_low_levels):
        self.n_products = int(n_products)
        self.num_price_levels = int(num_price_levels)
        self.promotion_budget = min(int(promotion_budget), self.n_products)
        self.pricing_num_low_levels = int(pricing_num_low_levels)
        if self.pricing_num_low_levels < 1 or self.pricing_num_low_levels > self.num_price_levels:
            raise ValueError("pricing_num_low_levels must be in [1, num_price_levels]")
        self.d = self.n_products * self.num_price_levels

    def solve(self, score, oracle_context=None):
        del oracle_context
        values = np.asarray(score, dtype=float).reshape(self.n_products, self.num_price_levels)
        dp = np.full((self.n_products + 1, self.promotion_budget + 1), -np.inf, dtype=float)
        parent_budget = np.full((self.n_products + 1, self.promotion_budget + 1), -1, dtype=int)
        parent_choice = np.full((self.n_products + 1, self.promotion_budget + 1), -1, dtype=int)
        dp[0, 0] = 0.0

        for prod_idx in range(self.n_products):
            for used_budget in range(self.promotion_budget + 1):
                if not np.isfinite(dp[prod_idx, used_budget]):
                    continue
                for level_idx in range(self.num_price_levels):
                    cost = 1 if level_idx < self.pricing_num_low_levels else 0
                    next_budget = used_budget + cost
                    if next_budget > self.promotion_budget:
                        continue
                    candidate = dp[prod_idx, used_budget] + float(values[prod_idx, level_idx])
                    if candidate > dp[prod_idx + 1, next_budget]:
                        dp[prod_idx + 1, next_budget] = candidate
                        parent_budget[prod_idx + 1, next_budget] = used_budget
                        parent_choice[prod_idx + 1, next_budget] = level_idx

        final_budget = int(np.argmax(dp[self.n_products]))
        action = np.zeros((self.n_products, self.num_price_levels), dtype=float)
        budget = final_budget
        for prod_idx in range(self.n_products, 0, -1):
            level_idx = parent_choice[prod_idx, budget]
            if level_idx < 0:
                raise RuntimeError("PricingOracle failed to recover a complete action")
            action[prod_idx - 1, level_idx] = 1.0
            budget = parent_budget[prod_idx, budget]
        return action.reshape(-1)
