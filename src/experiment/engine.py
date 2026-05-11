"""Experiment execution engine."""

from __future__ import annotations

import copy
import json
import os
import shutil
import time
from typing import Any, Dict

import yaml

from src.common.metrics import MetricTracker
from src.common.seeds import seed_torch, set_seed
from src.experiment.config import resolve_output_dir
from src.experiment.registry import create_algorithms, create_env_and_oracle, create_env_for_algo_run


def _requires_torch_runtime(config: Dict[str, Any]) -> bool:
    model_type = str(config.get("model_type", "linear")).lower()
    if model_type in {
        "nn",
        "two_layer_nn",
        "shared_nn",
        "diffusion",
        "cnf",
        "shared_diffusion",
        "shared_cnf",
    }:
        return True
    if any(
        bool(config.get(flag, False))
        for flag in (
            "run_hybrid_bandit",
        )
    ):
        return True
    return False


def run_simulation(env, algo, tracker: MetricTracker):
    """Run one algorithm on one environment trajectory."""
    context = env.reset()
    objective_sense = getattr(env, "objective_sense", "min")
    objective_name = getattr(env, "objective_name", "cost")

    while env.t < env.T:
        true_reward = env.get_true_objective_vector()
        oracle_context = env.get_oracle_context()
        if getattr(algo, "uses_true_reward", False):
            action = algo.select_action(context, oracle_context=oracle_context, true_reward=true_reward)
        else:
            action = algo.select_action(context, oracle_context=oracle_context)

        feedback, info = env.step(action)
        action_used = info.get("action_taken", action)
        algo.update(context, action_used, feedback, info=info)
        diagnostics_fn = getattr(algo, "get_step_diagnostics", None)
        if callable(diagnostics_fn):
            tracker.log_diagnostics(diagnostics_fn())

        eval_info = env.get_step_eval_info()
        if eval_info is None:
            eval_info = info

        objective_value = float(eval_info.get("objective_value", info.get("objective_value", feedback)))
        expected_objective = float(eval_info.get("expected_objective", info.get("expected_objective", objective_value)))
        objective_sense = str(eval_info.get("objective_sense", info.get("objective_sense", objective_sense))).lower()
        objective_name = str(eval_info.get("objective_name", info.get("objective_name", objective_name)))
        tracker.objective_sense = objective_sense
        tracker.objective_name = objective_name
        tracker.log_step(feedback, objective_value, expected_objective)

        context = env.get_context()

    return {
        "objective_sense": objective_sense,
        "objective_name": objective_name,
    }


class ExperimentRunner:
    """Orchestrates one full experiment run from an in-memory config dict."""

    def __init__(self, config: Dict[str, Any], *, verbose: bool = True):
        self.config = copy.deepcopy(config)
        self.verbose = bool(verbose)
        self._output_lock_path: str | None = None

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _acquire_output_lock(self, output_dir: str) -> None:
        lock_path = f"{output_dir}.lock"
        lock_parent = os.path.dirname(lock_path)
        if lock_parent:
            os.makedirs(lock_parent, exist_ok=True)
        lock_payload = {
            "pid": os.getpid(),
            "time": time.time(),
            "experiment_name": self.config.get("experiment_name"),
            "seed": self.config.get("seed"),
        }
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Output directory is locked by another active run: {output_dir}. "
                f"If no run is active, remove the stale lock file {lock_path} and retry."
            ) from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(lock_payload, handle, indent=2)
        self._output_lock_path = lock_path

    def _release_output_lock(self) -> None:
        if self._output_lock_path and os.path.exists(self._output_lock_path):
            os.remove(self._output_lock_path)
        self._output_lock_path = None

    def _prepare_output_dir(self) -> str:
        output_dir = resolve_output_dir(self.config)
        self._acquire_output_lock(output_dir)
        overwrite_output = bool(self.config.get("overwrite_output", True))
        merge_output_dir = bool(self.config.get("merge_output_dir", False))
        if os.path.exists(output_dir):
            if merge_output_dir:
                os.makedirs(output_dir, exist_ok=True)
            elif overwrite_output:
                shutil.rmtree(output_dir)
            else:
                raise FileExistsError(
                    f"Output directory already exists: {output_dir}. "
                    "Set overwrite_output: true to overwrite."
                )
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _prepare_algorithm_output_dir(self, output_dir: str, algo_name: str) -> None:
        if not bool(self.config.get("merge_output_dir", False)):
            return
        algo_dir = os.path.join(output_dir, algo_name)
        if os.path.isdir(algo_dir):
            shutil.rmtree(algo_dir)

    def _save_config(self, output_dir: str) -> None:
        with open(os.path.join(output_dir, "config.yaml"), "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.config, handle, sort_keys=False)

    def run(self) -> Dict[str, Any]:
        try:
            output_dir = self._prepare_output_dir()

            seed = int(self.config["seed"])
            set_seed(seed, include_torch=False)
            env_template, oracle = create_env_and_oracle(self.config, seed)
            resolved_updates = env_template.get_resolved_config_updates()
            if resolved_updates:
                self.config.update(resolved_updates)
            self._save_config(output_dir)
            if _requires_torch_runtime(self.config):
                seed_torch(seed)
            algorithms = create_algorithms(self.config, oracle)

            results: Dict[str, Any] = {}
            for name, algo in algorithms.items():
                self._log(f"Running {name}...")
                self._prepare_algorithm_output_dir(output_dir, name)
                env_run = create_env_for_algo_run(self.config, seed)

                tracker = MetricTracker()
                start_time = time.time()
                sim_result = run_simulation(env_run, algo, tracker)
                duration = time.time() - start_time
                self._log(f"Finished {name} in {duration:.2f}s")

                tracker.save(os.path.join(output_dir, name))
                results[name] = {"tracker": tracker, **sim_result}

            return {"output_dir": output_dir, "results": results}
        finally:
            self._release_output_lock()
