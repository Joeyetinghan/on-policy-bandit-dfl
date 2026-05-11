"""Empirical energy scheduling benchmark with exact load-profile actions."""

from __future__ import annotations

import os
import shlex
from typing import Dict

import numpy as np

from src.envs.base_env import BaseEnv

_ENERGY_DATA_CACHE: Dict[str, Dict[str, np.ndarray]] = {}

_ENERGY_SPLIT_ALIASES = {
    "all": "all",
    "validation": "validation",
    "experiment": "experiment",
}


def _interpolate_nans(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    mask = np.isnan(result)
    if not np.any(mask):
        return result
    valid = np.flatnonzero(~mask)
    if valid.size == 0:
        raise ValueError("Cannot interpolate a vector with no valid values")
    result[mask] = np.interp(np.flatnonzero(mask), valid, result[valid])
    return result


def load_energy_price_dataset(data_root: str) -> Dict[str, np.ndarray]:
    """Load and standardize the SEMO energy price dataset."""
    root = os.path.abspath(data_root)
    cached = _ENERGY_DATA_CACHE.get(root)
    if cached is not None:
        return cached

    path = os.path.join(root, "prices2013.dat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Energy price data not found: {path}")

    feature_names = [
        "HolidayFlag",
        "DayOfWeek",
        "WeekOfYear",
        "Month",
        "ForecastWindProduction",
        "SystemLoadEA",
        "SMPEA",
        "CO2Intensity",
    ]
    price_name = "SMPEP2"

    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().strip().split()
        header_map = {name: idx for idx, name in enumerate(header)}
        missing = [name for name in feature_names + [price_name] if name not in header_map]
        if missing:
            raise ValueError(f"Energy price file is missing required columns: {missing}")

        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            tokens = shlex.split(stripped)
            rows.append(tokens)

    if len(rows) % 48 != 0:
        raise ValueError(f"Expected a multiple of 48 half-hour rows, got {len(rows)}")

    features = np.array(
        [[float(tokens[header_map[name]]) for name in feature_names] for tokens in rows],
        dtype=float,
    )
    prices = np.array([float(tokens[header_map[price_name]]) for tokens in rows], dtype=float)

    # Match the PredOpt preprocessing: treat zero CO2 values as missing and interpolate.
    co2_idx = feature_names.index("CO2Intensity")
    features[features[:, co2_idx] == 0.0, co2_idx] = np.nan
    features[:, co2_idx] = _interpolate_nans(features[:, co2_idx])

    day_features = []
    day_prices = []
    for start in range(0, len(rows), 48):
        stop = start + 48
        feat_block = features[start:stop]
        price_block = prices[start:stop]
        if feat_block.shape[0] != 48:
            continue
        if np.isnan(feat_block).any() or np.isnan(price_block).any():
            continue
        day_features.append(feat_block)
        day_prices.append(price_block)

    contexts = np.asarray(day_features, dtype=float)
    cost_vectors = np.asarray(day_prices, dtype=float)
    if contexts.ndim != 3 or contexts.shape[1:] != (48, len(feature_names)):
        raise ValueError(f"Unexpected energy context shape: {contexts.shape}")
    if cost_vectors.shape != (contexts.shape[0], 48):
        raise ValueError(f"Unexpected energy price shape: {cost_vectors.shape}")

    # The online benchmark streams over the full empirical pool, so standardize
    # using all observed days rather than maintaining a separate train split.
    all_rows = contexts.reshape(-1, contexts.shape[-1])
    mean = all_rows.mean(axis=0)
    std = all_rows.std(axis=0)
    std[std < 1.0e-8] = 1.0
    contexts = (contexts - mean) / std

    dataset = {
        "contexts": contexts,
        "cost_vectors": cost_vectors,
    }
    _ENERGY_DATA_CACHE[root] = dataset
    return dataset


def _energy_split_indices(n_days: int, split: str, validation_fraction: float) -> tuple[int, int, str]:
    split_key = _ENERGY_SPLIT_ALIASES.get(str(split).lower())
    if split_key is None:
        valid = ", ".join(sorted(_ENERGY_SPLIT_ALIASES))
        raise ValueError(f"Unknown energy dataset_split={split!r}; expected one of: {valid}")
    if split_key == "all":
        return 0, n_days, split_key

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("energy_validation_fraction must be strictly between 0 and 1")
    cut = int(round(n_days * validation_fraction))
    cut = min(max(cut, 1), n_days - 1)
    if split_key == "validation":
        return 0, cut, split_key
    return cut, n_days, split_key


def load_energy_instance(path: str) -> Dict[str, object]:
    """Parse one scheduling instance file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Energy instance file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    q = int(lines[0])
    nb_resources = int(lines[1])
    nb_machines = int(lines[2])

    idle = [None] * nb_machines
    up = [None] * nb_machines
    down = [None] * nb_machines
    mc = [None] * nb_machines
    for machine in range(nb_machines):
        machine_row = lines[2 * machine + 3].split()
        idle[machine] = int(machine_row[1])
        up[machine] = float(machine_row[2])
        down[machine] = float(machine_row[3])
        mc[machine] = list(map(int, lines[2 * (machine + 2)].split()))

    lines_read = 2 * nb_machines + 2
    nb_tasks = int(lines[lines_read + 1])
    usage = [None] * nb_tasks
    duration = [None] * nb_tasks
    earliest = [None] * nb_tasks
    latest = [None] * nb_tasks
    power = [None] * nb_tasks
    for task in range(nb_tasks):
        task_row = lines[2 * task + lines_read + 2].split()
        duration[task] = int(task_row[1])
        earliest[task] = int(task_row[2])
        latest[task] = int(task_row[3])
        power[task] = float(task_row[4])
        usage[task] = list(map(int, lines[2 * task + lines_read + 3].split()))

    return {
        "nbMachines": nb_machines,
        "nbTasks": nb_tasks,
        "nbResources": nb_resources,
        "MC": mc,
        "U": usage,
        "D": duration,
        "E": earliest,
        "L": latest,
        "P": power,
        "idle": idle,
        "up": up,
        "down": down,
        "q": q,
    }


class EnergyEnv(BaseEnv):
    """Empirical energy cost benchmark with primitive schedule semi-bandit feedback.

    Actions remain 48-slot induced load profiles.  In ``semi_bandit`` mode the
    environment only accepts ``EnergyAction`` instances from ``EnergyOracle`` and
    reveals one linear primitive coefficient per selected task-start variable.
    Slot-mask semi-bandit feedback is intentionally unsupported for energy.
    """

    objective_sense = "min"
    objective_name = "cost"

    def __init__(self, config, seed):
        super().__init__(config, seed)
        self.data_root = str(config.get("data_root", "data/Energy/SchedulingInstances"))
        self.stream_mode = str(config.get("stream_mode", "chronological")).lower()
        if self.stream_mode not in {"chronological", "fixed_pass", "replay_with_replacement"}:
            raise ValueError(f"Unknown energy stream_mode: {self.stream_mode}")
        self.feedback_mode = str(config.get("feedback_mode", "bandit")).lower()
        if self.feedback_mode not in {"bandit", "semi_bandit", "full_feedback"}:
            raise ValueError("EnergyEnv feedback_mode must be one of {'bandit', 'semi_bandit', 'full_feedback'}")
        self.energy_feedback_protocol = str(
            config.get("energy_feedback_protocol", "primitive_schedule_semibandit_v2")
        ).lower()
        if self.feedback_mode == "semi_bandit" and self.energy_feedback_protocol not in {
            "primitive_schedule_semibandit",
            "primitive_schedule_semibandit_v2",
        }:
            raise ValueError(
                "EnergyEnv semi_bandit supports only primitive schedule feedback. "
                "Set energy_feedback_protocol='primitive_schedule_semibandit_v2' or omit the key."
            )

        dataset = load_energy_price_dataset(self.data_root)
        split = str(config.get("dataset_split", config.get("energy_dataset_split", "all")))
        validation_fraction = float(config.get("energy_validation_fraction", 0.5))
        split_start, split_stop, split_key = _energy_split_indices(
            int(dataset["contexts"].shape[0]),
            split,
            validation_fraction,
        )
        self.dataset_split = split_key
        self.energy_validation_fraction = validation_fraction
        self.energy_split_start_index = split_start
        self.energy_split_stop_index = split_stop
        self.contexts = dataset["contexts"][split_start:split_stop]
        self.cost_vectors = dataset["cost_vectors"][split_start:split_stop]

        self.p = self.contexts.shape[-1]
        self.d = self.cost_vectors.shape[-1]
        if int(config.get("p", self.p)) != self.p:
            raise ValueError(f"Energy benchmark expects p={self.p}, got {config.get('p')}")
        if int(config.get("d", self.d)) != self.d:
            raise ValueError(f"Energy benchmark expects d={self.d}, got {config.get('d')}")

        instance_name = str(config.get("energy_instance", "load3"))
        self.instance_name = instance_name
        self.instance_path = os.path.join(self.data_root, instance_name, "day01.txt")
        self.instance_data = load_energy_instance(self.instance_path)

        self.current_context = None
        self.current_c_t = None
        self.current_r_t = None
        self._last_step_eval_info = None

    def _round_index(self, round_idx: int) -> int:
        if self.stream_mode == "replay_with_replacement":
            return int(self.rng.randint(self.contexts.shape[0]))
        return int(round_idx % self.contexts.shape[0])

    def _refresh_round_state(self, round_idx: int) -> None:
        data_idx = self._round_index(round_idx)
        self.current_context = np.array(self.contexts[data_idx], dtype=float, copy=True)
        self.current_c_t = np.array(self.cost_vectors[data_idx], dtype=float, copy=True)
        self.current_r_t = -self.current_c_t

    def _validate_action(self, action):
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape != (self.d,):
            raise ValueError(f"Expected energy action shape {(self.d,)}, got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("Energy action contains non-finite values")
        if np.any(action < -1.0e-8):
            raise ValueError("Energy load profile must be nonnegative")
        return np.clip(action, 0.0, None)

    def _primitive_semi_bandit_info(self, raw_action, action_used, score_t, c_t):
        matrix = getattr(raw_action, "linear_observation_matrix", None)
        selected_primitives = getattr(raw_action, "selected_primitives", None)
        if matrix is None or selected_primitives is None:
            raise ValueError(
                "Energy semi-bandit feedback requires an EnergyAction returned by EnergyOracle.solve; "
                "plain 48-slot load profiles and slot-mask observations are unsupported."
            )
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.d:
            raise ValueError(
                "EnergyAction linear_observation_matrix must have shape "
                f"(num_selected, {self.d}), got {matrix.shape}"
            )
        return {
            "action_taken": action_used.copy(),
            "linear_observation_matrix": matrix.copy(),
            "linear_observation_values": matrix @ score_t,
            "linear_observed_objective_values": matrix @ c_t,
            "selected_primitives": [tuple(int(part) for part in primitive) for primitive in selected_primitives],
        }

    def reset(self):
        self.t = 0
        self._last_step_eval_info = None
        self._refresh_round_state(0)
        return self.current_context

    def step(self, action):
        raw_action = action
        action_used = self._validate_action(action)
        c_t = self.current_c_t
        score_t = self.current_r_t
        realized_cost = float(c_t @ action_used)
        reward = -realized_cost
        self.t += 1

        eval_info = {
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
        self._last_step_eval_info = eval_info

        if self.t < self.T:
            self._refresh_round_state(self.t)

        if self.feedback_mode == "bandit":
            info = {"action_taken": action_used.copy()}
        elif self.feedback_mode == "semi_bandit":
            info = self._primitive_semi_bandit_info(raw_action, action_used, score_t, c_t)
        else:
            info = eval_info
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

    def sample_eval_instance(self):
        """Sample one empirical context-cost pair with replacement for evaluation."""
        data_idx = int(self.rng.randint(self.contexts.shape[0]))
        context = np.array(self.contexts[data_idx], dtype=float, copy=True)
        cost_vector = np.array(self.cost_vectors[data_idx], dtype=float, copy=True)
        return {
            "context": context,
            "score_vector": -cost_vector,
            "oracle_context": None,
        }
