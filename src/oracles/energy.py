"""Energy scheduling oracle."""

from __future__ import annotations

import os

import numpy as np

from src.oracles.common import _require_gurobi


class EnergyAction(np.ndarray):
    """48-slot load profile with selected primitive schedule metadata."""

    def __new__(cls, load_profile, *, selected_primitives=(), linear_observation_matrix=None):
        obj = np.asarray(load_profile, dtype=float).reshape(-1).view(cls)
        obj.selected_primitives = tuple(tuple(int(part) for part in primitive) for primitive in selected_primitives)
        if linear_observation_matrix is None:
            matrix = np.zeros((0, obj.shape[0]), dtype=float)
        else:
            matrix = np.asarray(linear_observation_matrix, dtype=float)
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != obj.shape[0]:
            raise ValueError(
                "EnergyAction linear_observation_matrix must have shape "
                f"(num_selected, {obj.shape[0]}), got {matrix.shape}"
            )
        obj.linear_observation_matrix = matrix.copy()
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.selected_primitives = getattr(obj, "selected_primitives", ())
        self.linear_observation_matrix = getattr(obj, "linear_observation_matrix", None)

    def copy(self, order="C"):
        return EnergyAction(
            np.asarray(self).copy(order=order),
            selected_primitives=self.selected_primitives,
            linear_observation_matrix=self.linear_observation_matrix,
        )


class EnergyOracle:
    """Exact load-profile oracle for the energy scheduling benchmark."""

    def __init__(self, instance_data, backend="gurobi"):
        self.instance_data = instance_data
        self.backend = str(backend).lower()
        if self.backend != "gurobi":
            raise ValueError("EnergyOracle requires backend='gurobi'")

        self.nb_machines = int(instance_data["nbMachines"])
        self.nb_tasks = int(instance_data["nbTasks"])
        self.nb_resources = int(instance_data["nbResources"])
        self.mc = instance_data["MC"]
        self.usage = instance_data["U"]
        self.duration = instance_data["D"]
        self.earliest = instance_data["E"]
        self.latest = instance_data["L"]
        self.power = instance_data["P"]
        self.q_minutes = int(instance_data["q"])
        self.d = 1440 // self.q_minutes

        self._gp = None
        self._model = None
        self._vars = {}
        self._init_gurobi_model()

    @staticmethod
    def _gurobi_threads_from_env():
        raw = os.environ.get("GUROBI_THREADS")
        if raw is None or str(raw).strip() == "":
            return None
        try:
            threads = int(raw)
        except ValueError as exc:
            raise ValueError(f"GUROBI_THREADS must be an integer, got {raw!r}") from exc
        if threads < 1:
            raise ValueError(f"GUROBI_THREADS must be positive, got {threads}")
        return threads

    def _init_gurobi_model(self):
        gp = _require_gurobi()
        self._gp = gp
        model = gp.Model("energy_oracle")
        model.Params.OutputFlag = 0
        gurobi_threads = self._gurobi_threads_from_env()
        if gurobi_threads is not None:
            model.Params.Threads = gurobi_threads
        tasks = range(self.nb_tasks)
        machines = range(self.nb_machines)
        time_slots = range(self.d)

        vars_ = model.addVars(tasks, machines, time_slots, vtype=gp.GRB.BINARY, name="x")
        model.ModelSense = gp.GRB.MAXIMIZE

        for task in tasks:
            model.addConstr(
                gp.quicksum(vars_[task, machine, t] for machine in machines for t in time_slots) == 1.0,
                name=f"assign_{task}",
            )
            if self.earliest[task] > 0:
                model.addConstr(
                    gp.quicksum(vars_[task, machine, t] for machine in machines for t in range(self.earliest[task])) == 0.0,
                    name=f"earliest_{task}",
                )
            latest_start = self.latest[task] - self.duration[task]
            if latest_start + 1 < self.d:
                model.addConstr(
                    gp.quicksum(
                        vars_[task, machine, t]
                        for machine in machines
                        for t in range(max(latest_start + 1, 0), self.d)
                    )
                    == 0.0,
                    name=f"latest_{task}",
                )

        for resource in range(self.nb_resources):
            for machine in machines:
                for slot in time_slots:
                    model.addConstr(
                        gp.quicksum(
                            gp.quicksum(
                                vars_[task, machine, start]
                                for start in range(max(0, slot - self.duration[task] + 1), slot + 1)
                            )
                            * self.usage[task][resource]
                            for task in tasks
                        )
                        <= float(self.mc[machine][resource]),
                        name=f"cap_r{resource}_m{machine}_t{slot}",
                    )

        model.update()
        self._model = model
        self._vars = vars_

    def _set_objective(self, score):
        score = np.asarray(score, dtype=float).reshape(self.d)
        objective = self._gp.quicksum(
            self._vars[task, machine, start]
            * float(np.sum(score[start:min(self.d, start + self.duration[task])]) * self.power[task] * self.q_minutes / 60.0)
            for task in range(self.nb_tasks)
            for machine in range(self.nb_machines)
            for start in range(self.d)
        )
        self._model.setObjective(objective, self._gp.GRB.MAXIMIZE)

    def _extract_energy_action(self):
        load = np.zeros(self.d, dtype=float)
        selected_primitives = []
        observation_rows = []
        for task in range(self.nb_tasks):
            task_duration = self.duration[task]
            task_load = float(self.power[task] * self.q_minutes / 60.0)
            for machine in range(self.nb_machines):
                for start in range(self.d):
                    if self._vars[task, machine, start].X > 0.5:
                        stop = min(self.d, start + task_duration)
                        load[start:stop] += task_load
                        selected_primitives.append((task, machine, start))
                        row = np.zeros(self.d, dtype=float)
                        row[start:stop] = task_load
                        observation_rows.append(row)
        matrix = np.vstack(observation_rows) if observation_rows else np.zeros((0, self.d), dtype=float)
        return EnergyAction(
            load,
            selected_primitives=selected_primitives,
            linear_observation_matrix=matrix,
        )

    def solve(self, score, oracle_context=None):
        del oracle_context
        self._set_objective(score)
        self._model.optimize()
        if self._model.Status != self._gp.GRB.OPTIMAL:
            raise RuntimeError(f"EnergyOracle Gurobi solve failed with status {self._model.Status}")
        return self._extract_energy_action()
