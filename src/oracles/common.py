"""Shared optional-solver imports for exact oracles."""

from __future__ import annotations

import gurobipy as gp


def _require_gurobi():
    return gp
