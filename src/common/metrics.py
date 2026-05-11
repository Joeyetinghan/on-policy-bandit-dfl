"""Metric tracking and lightweight plots."""

from __future__ import annotations

import json
import os
import time

import numpy as np


def atomic_json_write(path, data, *, retries=3, retry_delay=0.1):
    tmp_path = f"{path}.tmp.{os.getpid()}"
    last_error = None
    for attempt in range(retries):
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
            os.replace(tmp_path, path)
            return
        except OSError as exc:
            last_error = exc
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            if attempt + 1 < retries:
                time.sleep(retry_delay)
                continue
            raise
    if last_error is not None:
        raise last_error


class MetricTracker:
    """Collect stepwise scalar feedback and objective summaries."""

    def __init__(self):
        self.rewards = []
        self.objectives = []
        self.expected_objectives = []
        self.violations = []
        self.objective_sense = "min"
        self.objective_name = "cost"
        self.diagnostics = []

    def log_step(self, reward, objective_value, expected_objective, violation=0):
        self.rewards.append(float(reward))
        self.objectives.append(float(objective_value))
        self.expected_objectives.append(float(expected_objective))
        self.violations.append(float(violation))

    def log_diagnostics(self, diagnostics):
        if not diagnostics:
            return
        self.diagnostics.append({key: _json_scalar(value) for key, value in dict(diagnostics).items()})

    def to_dict(self):
        cum_feedback = np.cumsum(self.rewards).tolist()
        data = {
            "cum_feedback": cum_feedback,
            "feedback_values": np.asarray(self.rewards).tolist(),
            # Compatibility keys for analysis scripts that read historical metric names.
            "cum_reward": cum_feedback,
            "rewards": np.asarray(self.rewards).tolist(),
            "cum_objective": np.cumsum(self.objectives).tolist(),
            "cum_expected_objective": np.cumsum(self.expected_objectives).tolist(),
            "objective_values": np.asarray(self.objectives).tolist(),
            "expected_objectives": np.asarray(self.expected_objectives).tolist(),
            "objective_sense": self.objective_sense,
            "objective_name": self.objective_name,
            "violations": np.asarray(self.violations).tolist(),
        }
        if self.diagnostics:
            data["diagnostics"] = self.diagnostics
        return data

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        data = self.to_dict()
        metrics_path = os.path.join(output_dir, "metrics.json")
        atomic_json_write(metrics_path, data)

        try:
            self.plot(output_dir)
        except OSError:
            # Plot artifacts are helpful for debugging, but failures on shared
            # filesystems should not invalidate the experiment metrics.
            pass

    def plot(self, output_dir):
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)
        plt.figure()
        plt.plot(np.cumsum(self.objectives), label=f"Cumulative {self.objective_name.title()}")
        plt.xlabel("Time")
        plt.ylabel(self.objective_name.title())
        plt.legend()
        plt.savefig(os.path.join(output_dir, "cum_objective.png"))
        plt.close()


def _json_scalar(value):
    if value is None:
        return None
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, (float, int)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (str, bool)):
        return value
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return value if np.isfinite(value) else None
