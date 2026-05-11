"""Top-k oracle."""

from __future__ import annotations

import numpy as np

from src.oracles.common import _require_gurobi


class TopKOracle:
    """Exact oracle for the fractional exact-cardinality top-k polytope."""

    def __init__(self, d, k, backend="native"):
        self.d = int(d)
        self.k = float(k)
        self.backend = str(backend).lower()
        if self.backend not in {"native", "gurobi"}:
            raise ValueError(f"Unknown TopKOracle backend: {self.backend}")

        self._gp = None
        self._model = None
        self._vars = None
        if self.backend == "gurobi":
            self._init_gurobi_model()

    def _init_gurobi_model(self):
        gp = _require_gurobi()
        self._gp = gp
        model = gp.Model("topk_oracle")
        model.Params.OutputFlag = 0
        vars_ = model.addVars(self.d, lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="w")
        model.addConstr(gp.quicksum(vars_[i] for i in range(self.d)) == self.k, name="topk_cardinality")
        model.ModelSense = gp.GRB.MAXIMIZE
        model.update()
        self._model = model
        self._vars = vars_

    def _solve_native(self, score):
        score = np.asarray(score, dtype=float).reshape(self.d)
        action = np.zeros(self.d, dtype=float)
        order = np.argsort(score)[::-1]
        remaining = self.k
        for idx in order:
            if remaining <= 1.0e-12:
                break
            take = min(1.0, remaining)
            action[idx] = take
            remaining -= take
        return action

    def _solve_gurobi(self, score):
        score = np.asarray(score, dtype=float).reshape(self.d)
        for i in range(self.d):
            self._vars[i].Obj = float(score[i])
        self._model.optimize()
        if self._model.Status != self._gp.GRB.OPTIMAL:
            raise RuntimeError(f"TopKOracle Gurobi solve failed with status {self._model.Status}")
        return np.array([self._vars[i].X for i in range(self.d)], dtype=float)

    def solve(self, score, oracle_context=None):
        del oracle_context
        if self.backend == "gurobi":
            return self._solve_gurobi(score)
        return self._solve_native(score)
