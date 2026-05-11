#!/usr/bin/env python3
"""Prepare energy-specific paper experiment runs and references.

This is intentionally separate from the default six paper blocks.  Energy uses
the empirical load3 time-series split: tune on validation days, evaluate on the
held-out experiment days, and average seeds only as algorithm randomness.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lib.io import loss_mix_overrides, write_yaml  # noqa: E402
from scripts.paper.paper_runs import (  # noqa: E402
    BuildOptions,
    FIXED_ALPHA_DIAGNOSTIC_GRID,
    GENERATIVE_MODEL_TYPES,
    PAPER_LOSSES,
    TUNED_CONFIG_KEYS,
    TUNED_METADATA_KEYS,
    _apply_default_adaptive_alpha,
    _base_flags,
    _experiment,
    _feedback_override,
    _model_label,
    _repo_rel,
    campaign_paths,
    ensure_result_block_dirs,
    effective_seeds,
    safe_name,
)
from src.common.surrogate_loss_names import (  # noqa: E402
    canonical_surrogate_loss_name,
    surrogate_loss_display_name,
)


DEFAULT_TUNED_CONFIGS = Path("configs/tuned/energy.yaml")


ENERGY_BLOCK_IDS = (
    "01_main_point_models",
    "02_generative_ablation",
    "03_alpha_diagnostics",
    "04_feedback_ablation",
    "05_surrogate_loss_grid",
)
ENERGY_BLOCK_ALIASES = {
    "energy_main": "01_main_point_models",
    "energy_distribution_family": "02_generative_ablation",
    **{block_id: block_id for block_id in ENERGY_BLOCK_IDS},
}
ENERGY_BLOCK_CHOICES = tuple(ENERGY_BLOCK_ALIASES)
ENERGY_BLOCK_TITLES = {
    "01_main_point_models": "energy main point-model comparison",
    "02_generative_ablation": "energy distribution-family comparison",
    "03_alpha_diagnostics": "energy adaptive-alpha diagnostics",
    "04_feedback_ablation": "energy feedback-regime comparison",
    "05_surrogate_loss_grid": "energy surrogate-loss comparison",
}
ENERGY_HORIZON = 395
ENERGY_QUICK_HORIZON = 25


def _timestamped_campaign_name() -> str:
    return "paper_energy_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _canonical_energy_block_id(block_id: str) -> str:
    try:
        return ENERGY_BLOCK_ALIASES[block_id]
    except KeyError as exc:
        raise ValueError(f"Unknown energy block: {block_id}") from exc


def _energy_tuned_key(energy_instance: str, tuning_split: str, model_type: str) -> str:
    return f"energy/{energy_instance}_{tuning_split}/{model_type}"


def _apply_energy_tuned_config(
    overrides: dict[str, Any],
    *,
    model_type: str,
    tuned_configs: dict[str, Any] | None,
    tuned_config_path: Path | None,
    require_tuned_configs: bool,
    energy_instance: str,
    tuning_split: str,
) -> None:
    selected_key = _energy_tuned_key(energy_instance, tuning_split, model_type)
    if tuned_configs is None:
        if require_tuned_configs:
            raise ValueError("--require-tuned-configs was set but no --tuned-configs file was provided")
        overrides.update({"paper_tuned_config_applied": False, "paper_tuned_config_missing_key": selected_key})
        return

    selected = tuned_configs.get(selected_key)
    if selected is None:
        if require_tuned_configs:
            raise KeyError(f"Missing tuned energy config {selected_key!r} in {tuned_config_path}.")
        overrides.update({"paper_tuned_config_applied": False, "paper_tuned_config_missing_key": selected_key})
        return

    applied_keys: list[str] = []
    is_generative = model_type in GENERATIVE_MODEL_TYPES
    for key in TUNED_CONFIG_KEYS:
        value = selected.get(key)
        if value is None:
            continue
        if key == "policy_sampling_scale" and is_generative:
            continue
        overrides[key] = value
        applied_keys.append(key)

    if is_generative and selected.get("theta_lr") is not None:
        overrides["generative_lr"] = float(selected["theta_lr"])
        applied_keys.append("generative_lr")

    tuned_metadata = {key: selected[key] for key in TUNED_METADATA_KEYS if selected.get(key) is not None}
    overrides.update(
        {
            "paper_tuned_config_applied": True,
            "paper_tuned_config_key": selected_key,
            "paper_tuned_config_source": "" if tuned_config_path is None else _repo_rel(tuned_config_path),
            "paper_tuned_config_candidate_id": str(selected.get("hybrid_tuning_candidate_id", "")),
            "paper_tuned_config_applied_keys": applied_keys,
            "paper_tuned_config_metadata": tuned_metadata,
        }
    )


def _energy_common_overrides(
    *,
    opts: BuildOptions,
    model_type: str,
    actor_family: str,
    energy_instance: str,
    dataset_split: str,
    tuning_split: str,
    validation_fraction: float,
    data_root: str,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        **_base_flags(),
        "benchmark": "energy",
        "solver_backend": "gurobi",
        "data_root": data_root,
        "energy_instance": energy_instance,
        "stream_mode": "chronological",
        "dataset_split": dataset_split,
        "energy_validation_fraction": float(validation_fraction),
        "T": ENERGY_QUICK_HORIZON if opts.quick else ENERGY_HORIZON,
        "p": 8,
        "d": 48,
        "feedback_mode": "bandit",
        "model_type": model_type,
        "torch_device": opts.device,
        "paper_problem": "energy",
        "paper_problem_label": "energy",
        "paper_point_model": _model_label(model_type),
        "paper_actor_family": actor_family,
        "paper_feedback": "scalar_bandit",
        "paper_energy_instance": energy_instance,
        "paper_energy_dataset_split": dataset_split,
        "paper_energy_tuning_split": tuning_split,
    }
    if opts.eval_every is not None:
        overrides["eval_every"] = int(opts.eval_every)
    _apply_energy_tuned_config(
        overrides,
        model_type=model_type,
        tuned_configs=opts.tuned_configs,
        tuned_config_path=opts.tuned_config_path,
        require_tuned_configs=opts.require_tuned_configs,
        energy_instance=energy_instance,
        tuning_split=tuning_split,
    )
    _apply_default_adaptive_alpha(overrides)
    return overrides


def _energy_experiments_for(
    *,
    block_id: str,
    opts: BuildOptions,
    output_root: Path,
    model_type: str,
    actor_family: str,
    algo_keys: Iterable[str],
    common: dict[str, Any],
    extra_slug: str,
    method_extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    experiments = []
    for algo_key in algo_keys:
        for seed in effective_seeds(opts.num_seeds, opts.quick):
            experiments.append(
                _experiment(
                    block_id=block_id,
                    problem="energy",
                    model_type=model_type,
                    algo_key=algo_key,
                    seed=seed,
                    output_root=output_root,
                    overrides=common,
                    extra_slug=extra_slug,
                    method_extra=method_extra,
                )
            )
    return experiments


def _build_energy_main(opts: BuildOptions, output_root: Path, args: Any) -> list[dict[str, Any]]:
    experiments = []
    algos = ("adaptive_hybrid", "surrogate_only", "score_only", "greedycb", "epsgreedycb", "tscb")
    # energy_main is now linear-only; the NN point model lives in
    # energy_distribution_family alongside the generative families.
    for model_type, actor_family in (("shared_linear", "gaussian_linear"),):
        common = _energy_common_overrides(
            opts=opts,
            model_type=model_type,
            actor_family=actor_family,
            energy_instance=args.energy_instance,
            dataset_split=args.dataset_split,
            tuning_split=args.tuning_split,
            validation_fraction=args.energy_validation_fraction,
            data_root=args.data_root,
        )
        experiments.extend(
            _energy_experiments_for(
                block_id="01_main_point_models",
                opts=opts,
                output_root=output_root,
                model_type=model_type,
                actor_family=actor_family,
                algo_keys=algos,
                common=common,
                extra_slug=f"{actor_family}_{args.energy_instance}_{args.dataset_split}",
            )
        )
    return experiments


def _build_energy_distribution_family(
    opts: BuildOptions,
    output_root: Path,
    args: Any,
) -> list[dict[str, Any]]:
    experiments = []
    algos = ("adaptive_hybrid", "surrogate_only")
    # Linear lives in energy_main; here we compare gaussian_nn vs the
    # generative families (cnf, diffusion).
    families = (
        ("gaussian_nn", "shared_nn"),
        ("cnf", "shared_cnf"),
        ("diffusion", "shared_diffusion"),
    )
    for actor_family, model_type in families:
        common = _energy_common_overrides(
            opts=opts,
            model_type=model_type,
            actor_family=actor_family,
            energy_instance=args.energy_instance,
            dataset_split=args.dataset_split,
            tuning_split=args.tuning_split,
            validation_fraction=args.energy_validation_fraction,
            data_root=args.data_root,
        )
        experiments.extend(
            _energy_experiments_for(
                block_id="02_generative_ablation",
                opts=opts,
                output_root=output_root,
                model_type=model_type,
                actor_family=actor_family,
                algo_keys=algos,
                common=common,
                extra_slug=f"{actor_family}_{args.energy_instance}_{args.dataset_split}",
            )
        )
    return experiments


def _build_energy_alpha_diagnostics(
    opts: BuildOptions,
    output_root: Path,
    args: Any,
) -> list[dict[str, Any]]:
    experiments = []
    model_type = "shared_linear"
    actor_family = "gaussian_linear"

    adaptive = _energy_common_overrides(
        opts=opts,
        model_type=model_type,
        actor_family=actor_family,
        energy_instance=args.energy_instance,
        dataset_split=args.dataset_split,
        tuning_split=args.tuning_split,
        validation_fraction=args.energy_validation_fraction,
        data_root=args.data_root,
    )
    experiments.extend(
        _energy_experiments_for(
            block_id="03_alpha_diagnostics",
            opts=opts,
            output_root=output_root,
            model_type=model_type,
            actor_family=actor_family,
            algo_keys=("adaptive_hybrid",),
            common=adaptive,
            extra_slug=f"adaptive_alpha_{actor_family}_{args.energy_instance}_{args.dataset_split}",
        )
    )

    for fixed_alpha in FIXED_ALPHA_DIAGNOSTIC_GRID:
        common = _energy_common_overrides(
            opts=opts,
            model_type=model_type,
            actor_family=actor_family,
            energy_instance=args.energy_instance,
            dataset_split=args.dataset_split,
            tuning_split=args.tuning_split,
            validation_fraction=args.energy_validation_fraction,
            data_root=args.data_root,
        )
        common.update(
            {
                "hybrid_alpha_schedule": "constant",
                "hybrid_alpha_init": fixed_alpha,
                "hybrid_alpha_final": fixed_alpha,
            }
        )
        experiments.extend(
            _energy_experiments_for(
                block_id="03_alpha_diagnostics",
                opts=opts,
                output_root=output_root,
                model_type=model_type,
                actor_family=actor_family,
                algo_keys=("adaptive_hybrid",),
                common=common,
                extra_slug=(
                    f"fixed_alpha_{fixed_alpha:g}_{actor_family}_{args.energy_instance}_{args.dataset_split}"
                ).replace(".", "p"),
                method_extra={"fixed_alpha": fixed_alpha},
            )
        )
    return experiments


def _build_energy_feedback_ablation(
    opts: BuildOptions,
    output_root: Path,
    args: Any,
) -> list[dict[str, Any]]:
    experiments = []
    model_type = "shared_linear"
    actor_family = "gaussian_linear"
    for regime in ("full_information", "semi_bandit", "scalar_bandit"):
        common = _energy_common_overrides(
            opts=opts,
            model_type=model_type,
            actor_family=actor_family,
            energy_instance=args.energy_instance,
            dataset_split=args.dataset_split,
            tuning_split=args.tuning_split,
            validation_fraction=args.energy_validation_fraction,
            data_root=args.data_root,
        )
        common.update(_feedback_override(regime))
        # Version the regimes whose feedback payload changed. Pure bandit is
        # identical to energy_main, so leave it unversioned to reuse those
        # completed scalar-feedback outputs.
        if regime == "semi_bandit":
            common["energy_feedback_protocol"] = "primitive_schedule_semibandit_v2"
        elif regime == "full_information":
            common["energy_feedback_protocol"] = "feedback_mode_respected_v1"
        experiments.extend(
            _energy_experiments_for(
                block_id="04_feedback_ablation",
                opts=opts,
                output_root=output_root,
                model_type=model_type,
                actor_family=actor_family,
                algo_keys=("adaptive_hybrid", "surrogate_only", "greedycb", "epsgreedycb", "tscb"),
                common=common,
                extra_slug=f"{regime}_{actor_family}_{args.energy_instance}_{args.dataset_split}",
            )
        )
    return experiments


def _build_energy_surrogate_loss_grid(
    opts: BuildOptions,
    output_root: Path,
    args: Any,
) -> list[dict[str, Any]]:
    experiments = []
    model_type = "shared_linear"
    actor_family = "gaussian_linear"
    for loss_name in PAPER_LOSSES:
        canonical = canonical_surrogate_loss_name(loss_name)
        display = surrogate_loss_display_name(loss_name)
        for algo_key in ("adaptive_hybrid", "surrogate_only"):
            common = _energy_common_overrides(
                opts=opts,
                model_type=model_type,
                actor_family=actor_family,
                energy_instance=args.energy_instance,
                dataset_split=args.dataset_split,
                tuning_split=args.tuning_split,
                validation_fraction=args.energy_validation_fraction,
                data_root=args.data_root,
            )
            common.update(
                {
                    "paper_loss": display,
                    "hybrid_loss_type": canonical,
                }
            )
            common.update(loss_mix_overrides(canonical, weight_key="hybrid_mse_weight"))
            experiments.extend(
                _energy_experiments_for(
                    block_id="05_surrogate_loss_grid",
                    opts=opts,
                    output_root=output_root,
                    model_type=model_type,
                    actor_family=actor_family,
                    algo_keys=(algo_key,),
                    common=common,
                    extra_slug=f"{actor_family}_{args.energy_instance}_{args.dataset_split}_{safe_name(display)}",
                )
            )
    return experiments


def _build_energy_experiments(
    *,
    opts: BuildOptions,
    output_root: Path,
    block_ids: Iterable[str],
    args: Any,
) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    for block_id in block_ids:
        canonical_block_id = _canonical_energy_block_id(block_id)
        if canonical_block_id == "01_main_point_models":
            experiments.extend(_build_energy_main(opts, output_root, args))
        elif canonical_block_id == "02_generative_ablation":
            experiments.extend(_build_energy_distribution_family(opts, output_root, args))
        elif canonical_block_id == "03_alpha_diagnostics":
            experiments.extend(_build_energy_alpha_diagnostics(opts, output_root, args))
        elif canonical_block_id == "04_feedback_ablation":
            experiments.extend(_build_energy_feedback_ablation(opts, output_root, args))
        elif canonical_block_id == "05_surrogate_loss_grid":
            experiments.extend(_build_energy_surrogate_loss_grid(opts, output_root, args))
        else:
            raise ValueError(f"Unknown energy block: {block_id}")
    return experiments


def _write_block_manifests(
    *,
    campaign_root: Path,
    opts: BuildOptions,
    block_ids: Iterable[str],
    args: Any,
    dry_run: bool,
) -> tuple[list[Path], list[dict[str, Any]]]:
    paths = campaign_paths(campaign_root)
    all_experiments: list[dict[str, Any]] = []
    manifest_paths: list[Path] = []
    if not dry_run:
        for key in ("manifests", "slurm", "outputs", "references", "results", "logs"):
            paths[key].mkdir(parents=True, exist_ok=True)
    canonical_block_ids = tuple(dict.fromkeys(_canonical_energy_block_id(block_id) for block_id in block_ids))
    if not dry_run:
        ensure_result_block_dirs(campaign_root, canonical_block_ids)
    for block_id in canonical_block_ids:
        block_experiments = _build_energy_experiments(
            opts=opts,
            output_root=paths["outputs"],
            block_ids=(block_id,),
            args=args,
        )
        manifest_path = paths["manifests"] / f"block_{block_id}.yaml"
        manifest_paths.append(manifest_path)
        all_experiments.extend(block_experiments)
        if not dry_run:
            write_yaml(
                manifest_path,
                {
                        "metadata": {
                            "paper_block_id": block_id,
                            "title": ENERGY_BLOCK_TITLES.get(block_id, block_id.replace("_", " ")),
                            "quick": opts.quick,
                            "num_seeds": 2 if opts.quick else opts.num_seeds,
                            "tuned_config_source": "" if opts.tuned_config_path is None else _repo_rel(opts.tuned_config_path),
                        "energy_instance": args.energy_instance,
                        "dataset_split": args.dataset_split,
                        "tuning_split": args.tuning_split,
                    },
                    "experiments": block_experiments,
                },
            )
    if not dry_run:
        existing_block_manifests = sorted(path.name for path in paths["manifests"].glob("block_*.yaml"))
        write_yaml(paths["manifests"] / "compare_energy_all.yaml", {"include": existing_block_manifests})
    return manifest_paths, all_experiments


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _replace_energy_alpha0_from_block01(campaign_root: Path) -> None:
    """Use complete block-01 DFHPG-0 rows for energy block-03 alpha=0."""

    block01 = campaign_root / "results/01_main_point_models"
    block03 = campaign_root / "results/03_alpha_diagnostics"
    raw01 = block01 / "raw/traces.csv"
    raw03 = block03 / "raw/traces.csv"
    summary01 = block01 / "summary/summary.csv"
    summary03 = block03 / "summary/summary.csv"
    if not (raw01.exists() and raw03.exists() and summary01.exists() and summary03.exists()):
        return

    alpha0_label = "DFHPG fixed alpha=0"

    raw_fields, raw03_rows = _read_csv_rows(raw03)
    _, raw01_rows = _read_csv_rows(raw01)
    alpha0_raw = []
    for row in raw01_rows:
        if row.get("problem") != "energy":
            continue
        if row.get("actor_family") != "gaussian_linear" or row.get("method") != "DFHPG-0":
            continue
        patched = dict(row)
        patched["method"] = alpha0_label
        if "algo_internal" in patched:
            patched["algo_internal"] = "DFHPG"
        alpha0_raw.append(patched)
    if alpha0_raw:
        kept = [row for row in raw03_rows if row.get("method") != alpha0_label]
        _write_csv_rows(raw03, raw_fields, kept + alpha0_raw)

    summary_fields, summary03_rows = _read_csv_rows(summary03)
    _, summary01_rows = _read_csv_rows(summary01)
    alpha0_summary = []
    for row in summary01_rows:
        if row.get("problem") != "energy":
            continue
        if row.get("actor_family") != "gaussian_linear" or row.get("method") != "DFHPG-0":
            continue
        patched = dict(row)
        patched["method"] = alpha0_label
        if "algo_internal" in patched:
            patched["algo_internal"] = "DFHPG"
        alpha0_summary.append(patched)
    if alpha0_summary:
        kept = [row for row in summary03_rows if row.get("method") != alpha0_label]
        _write_csv_rows(summary03, summary_fields, kept + alpha0_summary)


def main() -> None:
    raise SystemExit("Use scripts/paper_energy.py for public energy reproduction.")


if __name__ == "__main__":
    main()
