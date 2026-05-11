"""Run multiple experiments from one or more YAML sweep manifests."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, Iterable, List, Set

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.experiment.config import load_config
from src.experiment.engine import ExperimentRunner


BASELINE_DEFAULTS = {
    "run_random_oracle": True,
    "run_true_model": True,
}

ALGORITHM_TOGGLES = [
    "run_greedy_contextual_bandit",
    "run_epsilon_greedy_contextual_bandit",
    "run_thompson_contextual_bandit",
    "run_hybrid_bandit",
    "run_random_oracle",
    "run_true_model",
]


def _to_hashable(value: Any):
    if isinstance(value, dict):
        return tuple(sorted((k, _to_hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_to_hashable(v) for v in value)
    return value


def _cfg_key(config: Dict[str, Any], keys: Iterable[str]):
    return tuple(_to_hashable(config.get(k, None)) for k in keys)


def _baseline_signature(config: Dict[str, Any], algo_name: str):
    env_key = _cfg_key(
        config,
        [
            "benchmark",
            "solver_backend",
            "T",
            "p",
            "d",
            "k",
            "deg",
            "eps_bar",
            "data_generator_family",
            "energy_instance",
            "energy_dataset_split",
            "energy_validation_fraction",
            "power_data_source",
            "power_context_mode",
            "power_data_root",
            "power_data_start_index",
            "n_products",
            "num_price_levels",
            "pricing_context_dim",
            "pricing_context_degree",
            "pricing_price_degree",
            "promotion_budget",
            "pricing_num_low_levels",
            "pricing_common_shock_sigma",
            "feedback_mode",
            "feedback_delay",
            "dataset_split",
            "stream_mode",
            "seed",
        ],
    )
    return (algo_name, env_key)


def _load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return json.loads(text)
    return yaml.safe_load(text)


def _has_enabled_algorithm(config: Dict[str, Any]) -> bool:
    for toggle in ALGORITHM_TOGGLES:
        if bool(config.get(toggle, BASELINE_DEFAULTS.get(toggle, False))):
            return True
    return False


def _resolve_relative(base_file: str, target: str) -> str:
    if os.path.isabs(target):
        return os.path.normpath(target)
    candidate = os.path.normpath(os.path.join(os.path.dirname(base_file), target))
    if os.path.exists(candidate):
        return candidate
    return os.path.normpath(target)


def _normalize_experiment_paths(exp: Dict[str, Any], manifest_path: str) -> Dict[str, Any]:
    out = copy.deepcopy(exp)
    if "config" in out:
        out["config"] = _resolve_relative(manifest_path, str(out["config"]))
    if "base" in out:
        out["base"] = _resolve_relative(manifest_path, str(out["base"]))
    return out


def _load_manifest(path: str, stack: Set[str] | None = None) -> List[Dict[str, Any]]:
    if stack is None:
        stack = set()
    norm_path = os.path.normpath(path)
    if norm_path in stack:
        raise ValueError(f"Detected cyclic include while loading manifests: {norm_path}")
    stack.add(norm_path)

    data = _load_yaml(norm_path) or {}
    experiments: List[Dict[str, Any]] = []

    for include in data.get("include", []):
        include_path = _resolve_relative(norm_path, str(include))
        experiments.extend(_load_manifest(include_path, stack=stack))

    for exp in data.get("experiments", []):
        experiments.append(_normalize_experiment_paths(exp, norm_path))

    if "include" not in data and "experiments" not in data:
        raise ValueError(f"Manifest {norm_path} must define at least one of: include, experiments")
    stack.remove(norm_path)
    return experiments


def _build_run_config(
    base_config: Dict[str, Any],
    *,
    seed: int | None = None,
    experiment_name: str | None = None,
    config_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)
    if seed is not None:
        config["seed"] = seed
    if experiment_name is not None:
        config["experiment_name"] = experiment_name
    if config_overrides:
        config.update(config_overrides)
    return config


def run_experiment(config: Dict[str, Any], dry_run: bool = False) -> bool:
    if dry_run:
        summary = {
            "experiment_name": config.get("experiment_name"),
            "benchmark": config.get("benchmark"),
            "T": config.get("T"),
            "model_type": config.get("model_type"),
            "seed": config.get("seed"),
        }
        print(f"[DRY RUN] Would execute: {summary}")
        return True

    print("\n" + "=" * 60)
    print(
        "Running: benchmark={benchmark}, T={T}, model={model}, seed={seed}, name={name}".format(
            benchmark=config.get("benchmark"),
            T=config.get("T"),
            model=config.get("model_type"),
            seed=config.get("seed"),
            name=config.get("experiment_name"),
        )
    )
    print("=" * 60 + "\n")

    try:
        runner = ExperimentRunner(config, verbose=True)
        runner.run()
        return True
    except Exception as exc:
        print(f"Error running experiment: {exc}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Run experiment manifests")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="Path to root experiment manifest",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print runs without executing")
    parser.add_argument(
        "--filter",
        action="append",
        default=None,
        help="Filter experiments by name (substring). Can be passed multiple times.",
    )
    parser.add_argument(
        "--split-seeds",
        action="store_true",
        help="Treat each seed in each experiment group as its own experiment group before filtering/sharding.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="1-based runtime shard index into the loaded experiment groups.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=None,
        help="Number of experiment groups per runtime shard.",
    )
    parser.add_argument(
        "--dedup-baselines",
        dest="dedup_baselines",
        action="store_true",
        default=True,
        help="Deduplicate RandomOracle/TrueModel runs across equivalent settings (default: enabled)",
    )
    parser.add_argument(
        "--no-dedup-baselines",
        dest="dedup_baselines",
        action="store_false",
        help="Disable baseline deduplication",
    )
    args = parser.parse_args()

    manifest_path = os.path.normpath(args.config)
    all_experiments = _load_manifest(manifest_path)
    if not all_experiments:
        print(f"No experiments defined in {manifest_path}")
        return

    if args.split_seeds:
        split_experiments: List[Dict[str, Any]] = []
        for exp in all_experiments:
            seeds = exp.get("seeds", [None])
            for seed in seeds:
                split_exp = copy.deepcopy(exp)
                split_exp["seeds"] = [seed]
                if seed is not None:
                    split_exp["name"] = f"{exp.get('name', 'exp')}_seed{seed}"
                split_experiments.append(split_exp)
        all_experiments = split_experiments

    filter_tokens: List[str] = []
    if args.filter:
        for raw in args.filter:
            filter_tokens.extend([token.strip() for token in str(raw).split(",") if token.strip()])

    if filter_tokens:
        experiments = [
            e
            for e in all_experiments
            if any(token in str(e.get("name", "")) for token in filter_tokens)
        ]
        print(f"Filtered experiments for {filter_tokens}: {len(experiments)}/{len(all_experiments)}")
    else:
        experiments = all_experiments

    if args.shard_index is not None:
        if args.shard_index < 1:
            raise ValueError("--shard-index must be 1-based and positive")
        if args.shard_size is None or args.shard_size < 1:
            raise ValueError("--shard-size must be positive when --shard-index is set")
        start = (args.shard_index - 1) * args.shard_size
        end = start + args.shard_size
        before_shard = len(experiments)
        experiments = experiments[start:end]
        print(
            f"Runtime shard {args.shard_index}: experiments[{start}:{end}] "
            f"-> {len(experiments)}/{before_shard}"
        )
        if not experiments:
            print("Shard is empty; nothing to run.")
            return

    print(f"Found {len(experiments)} experiment groups to run")
    print(f"Baseline deduplication: {'enabled' if args.dedup_baselines else 'disabled'}")
    if args.dry_run:
        print("[DRY RUN MODE - No experiments will be executed]\n")

    total_runs = sum(len(exp.get("seeds", [None])) for exp in experiments)
    print(f"Total runs: {total_runs}\n")

    completed = 0
    failed = 0
    start_time = time.time()
    seen_baseline_sigs = {
        "RandomOracle": set(),
        "TrueModel": set(),
    }

    for index, exp in enumerate(experiments, start=1):
        exp_name = exp.get("name", f"exp_{index}")
        print("\n" + "#" * 60)
        print(f"# Experiment Group {index}/{len(experiments)}: {exp_name}")
        print("#" * 60)

        if "config" in exp:
            base_config = load_config(exp["config"])
        elif "base" in exp:
            base_config = load_config(exp["base"])
            base_config.update(exp.get("overrides", {}))
        else:
            print(f"Skipping {exp_name}: no config or base specified")
            continue

        seeds = exp.get("seeds", [None])
        for run_idx, seed in enumerate(seeds, start=1):
            print(f"\nRun {run_idx}/{len(seeds)} (seed={seed})")
            cfg_for_seed = _build_run_config(base_config, seed=seed)
            toggles: Dict[str, Any] = {}
            suppressed: List[str] = []

            if args.dedup_baselines:
                for algo_name, toggle_name in [
                    ("RandomOracle", "run_random_oracle"),
                    ("TrueModel", "run_true_model"),
                ]:
                    enabled = bool(cfg_for_seed.get(toggle_name, BASELINE_DEFAULTS[toggle_name]))
                    if not enabled:
                        continue
                    signature = _baseline_signature(cfg_for_seed, algo_name)
                    if signature in seen_baseline_sigs[algo_name]:
                        toggles[toggle_name] = False
                        suppressed.append(algo_name)
                    else:
                        seen_baseline_sigs[algo_name].add(signature)

            if suppressed:
                deduped_config = _build_run_config(cfg_for_seed, config_overrides=toggles)
                # Never suppress a baseline if it would leave the experiment
                # with no runnable algorithm at all.
                if not _has_enabled_algorithm(deduped_config):
                    toggles = {}
                    suppressed = []

            if suppressed:
                print(f"Suppressing duplicated baselines: {', '.join(suppressed)}")

            run_config = _build_run_config(
                cfg_for_seed,
                experiment_name=exp_name,
                config_overrides=toggles,
            )
            success = run_experiment(run_config, dry_run=args.dry_run)
            if success:
                completed += 1
            else:
                failed += 1
            print(f"Status: {'[PASS]' if success else '[FAIL]'}")

    elapsed = time.time() - start_time

    print("\n\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Total runs: {total_runs}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("=" * 60 + "\n")

    if not args.dry_run:
        print("Results saved to: outputs/")


if __name__ == "__main__":
    main()
