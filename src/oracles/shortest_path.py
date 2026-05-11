"""Shortest-path oracle."""

from __future__ import annotations

import networkx as nx
import numpy as np

from src.oracles.common import _require_gurobi


class ShortestPathOracle:
    """Source-sink path oracle on a DAG under a maximize-score interface."""

    def __init__(self, G, edges, backend="native"):
        self.G = G
        self.edges = edges
        self.q = len(edges)
        self.n_nodes = G.number_of_nodes()
        self.source = 0
        self.sink = self.n_nodes - 1
        self.edge_to_idx = {edge: idx for idx, edge in enumerate(self.edges)}
        self.backend = str(backend).lower()
        if self.backend not in {"native", "gurobi"}:
            raise ValueError(f"Unknown ShortestPathOracle backend: {self.backend}")

        self.A_eq = np.zeros((self.n_nodes, self.q))
        self.b_eq = np.zeros(self.n_nodes)
        self.b_eq[self.source] = -1.0
        self.b_eq[self.sink] = 1.0
        for idx, (u, v) in enumerate(self.edges):
            self.A_eq[u, idx] -= 1.0
            self.A_eq[v, idx] += 1.0

        self.topological_order = list(nx.topological_sort(self.G))

        self._gp = None
        self._model = None
        self._vars = None
        if self.backend == "gurobi":
            self._init_gurobi_model()

    def _init_gurobi_model(self):
        gp = _require_gurobi()
        self._gp = gp
        model = gp.Model("shortest_path_oracle")
        model.Params.OutputFlag = 0
        vars_ = model.addVars(self.q, lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="w")
        for node in range(self.n_nodes):
            expr = gp.quicksum(self.A_eq[node, idx] * vars_[idx] for idx in range(self.q))
            model.addConstr(expr == float(self.b_eq[node]), name=f"flow_{node}")
        model.ModelSense = gp.GRB.MAXIMIZE
        model.update()
        self._model = model
        self._vars = vars_

    def _solve_native(self, score):
        score = np.asarray(score, dtype=float).reshape(self.q)
        best_value = np.full(self.n_nodes, -np.inf, dtype=float)
        parent_edge = {}
        best_value[self.source] = 0.0

        for node in self.topological_order:
            if not np.isfinite(best_value[node]):
                continue
            for succ in self.G.successors(node):
                edge_idx = self.edge_to_idx[(node, succ)]
                candidate = best_value[node] + score[edge_idx]
                if candidate > best_value[succ]:
                    best_value[succ] = candidate
                    parent_edge[succ] = (node, edge_idx)

        if self.sink not in parent_edge:
            return np.zeros(self.q, dtype=float)

        action = np.zeros(self.q, dtype=float)
        node = self.sink
        while node != self.source:
            prev_node, edge_idx = parent_edge[node]
            action[edge_idx] = 1.0
            node = prev_node
        return action

    def _solve_gurobi(self, score):
        score = np.asarray(score, dtype=float).reshape(self.q)
        for idx in range(self.q):
            self._vars[idx].Obj = float(score[idx])
        self._model.optimize()
        if self._model.Status != self._gp.GRB.OPTIMAL:
            raise RuntimeError(f"ShortestPathOracle Gurobi solve failed with status {self._model.Status}")
        return np.array([self._vars[idx].X for idx in range(self.q)], dtype=float)

    def solve(self, score, oracle_context=None):
        del oracle_context
        if self.backend == "gurobi":
            return self._solve_gurobi(score)
        return self._solve_native(score)
