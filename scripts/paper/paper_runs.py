"""Isolated paper experiment manifest, execution, and aggregation helpers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lib.io import loss_mix_overrides, seed_list, write_yaml
from src.common.surrogate_loss_names import canonical_surrogate_loss_name, surrogate_loss_display_name
from src.experiment.config import load_config, safe_name


_BASE_CONFIG_CACHE: dict[str, dict[str, Any]] = {}


def _cached_base_config(base: str) -> dict[str, Any]:
    if base not in _BASE_CONFIG_CACHE:
        _BASE_CONFIG_CACHE[base] = load_config(str(REPO_ROOT / base))
    return dict(_BASE_CONFIG_CACHE[base])


EXECUTION_KEY_EXCLUDED_KEYS = {
    "experiment_name",
    "merge_output_dir",
    "output_dir",
    "output_root",
    "overwrite_output",
    "torch_device",
}


def _normalizable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _normalizable(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalizable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _execution_key_payload(base: str, config: dict[str, Any]) -> dict[str, Any]:
    behavior = {}
    for key in sorted(config):
        if key in EXECUTION_KEY_EXCLUDED_KEYS or key.startswith("paper_"):
            continue
        behavior[key] = _normalizable(config[key])
    return {"base": base, "config": behavior}


def _execution_key(base: str, config: dict[str, Any]) -> str:
    payload = _execution_key_payload(base, config)
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _execution_output_name(
    *,
    base: str,
    problem: str,
    model_type: str,
    algo_key: str,
    seed: int,
    config: dict[str, Any],
) -> tuple[str, str]:
    digest = _execution_key(base, config)
    degree = config.get("paper_degree", config.get("deg", config.get("pricing_context_degree", "")))
    degree_slug = "" if degree in {None, ""} else f"_deg{safe_name(degree)}"
    feedback = config.get("feedback_mode", config.get("paper_feedback", "bandit"))
    loss = config.get("hybrid_loss_type") if algo_key in {"adaptive_hybrid", "surrogate_only", "score_only"} else None
    loss_slug = "" if loss in {None, ""} else f"_{safe_name(loss)}"
    name = (
        f"paper_exec_{safe_name(problem)}_{safe_name(_model_label(model_type))}_{safe_name(algo_key)}"
        f"{degree_slug}_{safe_name(feedback)}{loss_slug}_seed{seed}_{digest}"
    )
    return digest, name


ALGORITHM_TOGGLES = (
    "run_greedy_contextual_bandit",
    "run_epsilon_greedy_contextual_bandit",
    "run_thompson_contextual_bandit",
    "run_hybrid_bandit",
    "run_random_oracle",
    "run_true_model",
)

PRIMARY_ALGO_DIR = {
    "adaptive_hybrid": "DFHPG",
    "surrogate_only": "DFHPG",
    "score_only": "DFHPG",
    "greedycb": "GreedyContextualBandit",
    "epsgreedycb": "EpsilonGreedyContextualBandit",
    "tscb": "ThompsonSamplingContextualBandit",
}

_LEGACY_GREEDY_CB_ALGO_DIR = "Greedy" + "Scalar" + "Bandit"
_LEGACY_EPS_GREEDY_CB_ALGO_DIR = "EpsilonGreedy" + "Scalar" + "Bandit"
_LEGACY_TS_CB_ALGO_DIR = "ThompsonSampling" + "Scalar" + "Bandit"

LEGACY_ALGO_DIRS = {
    "GreedyContextualBandit": (_LEGACY_GREEDY_CB_ALGO_DIR,),
    "EpsilonGreedyContextualBandit": (_LEGACY_EPS_GREEDY_CB_ALGO_DIR,),
    "ThompsonSamplingContextualBandit": (_LEGACY_TS_CB_ALGO_DIR,),
}

PRIMARY_METHOD_LABEL = {
    "adaptive_hybrid": "DFHPG",
    "surrogate_only": "DFHPG-0",
    "score_only": "DFHPG-1",
    "greedycb": "GreedyCB",
    "epsgreedycb": "$\\epsilon$-GreedyCB",
    "tscb": "TSCB",
}

_LEGACY_DFHPG_LABEL = "Hybrid" + "Bandit"
_LEGACY_DFHPG_ALGO_DIR = "Hybrid" + "Actor" + "Critic" + "Bandit"

PAPER_METHOD_RENAMES = {
    _LEGACY_DFHPG_LABEL: "DFHPG",
    f"{_LEGACY_DFHPG_LABEL}-ID": "DFHPG-ID",
    f"{_LEGACY_DFHPG_LABEL} fixed alpha=0": "DFHPG fixed alpha=0",
    f"{_LEGACY_DFHPG_LABEL} fixed alpha=0.25": "DFHPG fixed alpha=0.25",
    f"{_LEGACY_DFHPG_LABEL} fixed alpha=0.5": "DFHPG fixed alpha=0.5",
    f"{_LEGACY_DFHPG_LABEL} fixed alpha=0.75": "DFHPG fixed alpha=0.75",
    f"{_LEGACY_DFHPG_LABEL} fixed alpha=1": "DFHPG fixed alpha=1",
    "SurrogateOnly": "DFHPG-0",
    "ScoreOnly": "DFHPG-1",
    "EpsGreedyCB": "$\\epsilon$-GreedyCB",
}

BASELINE_METHOD_LABEL = {
    "TrueModel": "TrueModel",
}

BASE_BY_PROBLEM = {
    "topk": "configs/topk.yaml",
    "shortest_path": "configs/shortest_path.yaml",
    "pricing": "configs/pricing.yaml",
    "energy": "configs/energy.yaml",
}

BENCHMARK_BY_PROBLEM = {
    "topk": "topk",
    "shortest_path": "shortest_path",
    "pricing": "pricing",
    "energy": "energy",
}

PAPER_LOSSES = (
    "MSE+SPOPlus",
    "NID",
    "MAP_c",
    "SPOCaching",
    "listwiseLTR",
    "SPOPlus",
    "pairwiseDiff",
    "AIMLE",
    "PFYL",
    "NCE",
    "NCE_c",
    "pointwiseLTR",
    "IMLE",
    "pairwiseLTR",
    "contrastiveMAP",
    "PG",
    "DPO",
    "DBB",
)

ADAPTIVE_ALPHA_OVERRIDES = {
    "hybrid_alpha_schedule": "adaptive_nuisance_reliability",
    "hybrid_alpha_max": 0.5,
    "hybrid_alpha_min": 0.05,
    "hybrid_alpha_warmup_frac": 0.05,
    "hybrid_alpha_ema_decay": 0.98,
    "hybrid_alpha_smooth": 0.05,
}

DEFAULT_PAPER_DEGREE = 8
FIXED_ALPHA_DIAGNOSTIC_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
PAPER_DEGREES = (2, 4, 6, 8, 10)

GENERATIVE_MODEL_TYPES = {
    "cnf",
    "diffusion",
    "shared_cnf",
    "shared_diffusion",
}

TUNED_CONFIG_KEYS = (
    "theta_lr_schedule",
    "theta_lr",
    "theta_lr_offset",
    "nuisance_lr_schedule",
    "nuisance_lr",
    "nuisance_lr_offset",
    "model_update_batch_rounds",
    "nuisance_update_batch_rounds",
    "hybrid_loss_type",
    "hybrid_mse_weight",
    "policy_sampling_scale",
    "hybrid_alpha_schedule",
    "hybrid_alpha_init",
    "hybrid_alpha_final",
    "hybrid_alpha_warmup_init",
    "hybrid_alpha_warmup_steps",
    "hybrid_alpha_warmup_const_steps",
    "hybrid_alpha_warmdown_steps",
    "hybrid_alpha_max",
    "hybrid_alpha_min",
    "hybrid_alpha_gate",
    "hybrid_alpha_time_scale",
    "hybrid_alpha_warmup_frac",
    "hybrid_alpha_ema_decay",
    "hybrid_alpha_smooth",
    "hybrid_gradient_normalization",
    "policy_baseline_type",
    "generative_beta_dfl",
    "generative_regularizer_weight",
    "lambda_gen",
    "generative_num_dfl_samples",
    "generative_surrogate_num_iter",
    "generative_update_objective",
    "generative_aux_mode",
    "generative_risk_alpha",
    "diffusion_loss_weighting",
    "flow_num_coupling_layers",
    "flow_hidden_dim",
    "flow_log_scale_min",
    "flow_log_scale_max",
    "diffusion_num_steps",
    "diffusion_time_embed_dim",
    "diffusion_hidden_dim",
    "diffusion_inference_steps",
)

TUNED_METADATA_KEYS = (
    "hybrid_tuning_candidate_id",
    "hybrid_tuning_score_scale",
    "hybrid_tuning_actor_scale_multiplier",
    "hybrid_tuning_actor_scale_tuned",
    "hybrid_tuning_nuisance_lr_ratio",
    "hybrid_tuning_mixing_weight_type",
    "hybrid_tuning_generative_lambda_score",
    "hybrid_tuning_diffusion_surrogate_num_iter",
    "mean_avg_cum_expected_objective",
    "std_avg_cum_expected_objective",
    "runs",
)

REFERENCE_CONFIG_FIELDS = (
    "benchmark",
    "seed",
    "T",
    "solver_backend",
    "p",
    "d",
    "k",
    "grid_size",
    "deg",
    "eps_bar",
    "data_generator_family",
    "poly_offset",
    "poly_scale",
    "n_products",
    "num_price_levels",
    "pricing_context_dim",
    "pricing_context_degree",
    "pricing_price_degree",
    "promotion_budget",
    "pricing_num_low_levels",
    "pricing_common_shock_sigma",
    "data_root",
    "energy_instance",
    "stream_mode",
    "dataset_split",
    "energy_validation_fraction",
    "feedback_delay",
)

QUICK_HORIZON = {
    "topk": 50,
    "shortest_path": 50,
    "pricing": 50,
    "energy": 25,
}

FULL_HORIZON = {
    "topk": 2000,
    "shortest_path": 2000,
    "pricing": 2000,
    "energy": 395,
}

BLOCK02_FULL_HORIZON = {
    "topk": 15000,
    "shortest_path": 15000,
    "pricing": 15000,
}


@dataclass(frozen=True)
class BuildOptions:
    campaign_root: Path
    quick: bool
    num_seeds: int
    device: str
    eval_every: int | None = None
    tuned_configs: dict[str, Any] | None = None
    tuned_config_path: Path | None = None
    require_tuned_configs: bool = False


@dataclass(frozen=True)
class BlockSpec:
    block_id: str
    title: str
    default_num_seeds: int
    builder: Callable[[BuildOptions, Path], list[dict[str, Any]]]


def _repo_rel(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def timestamped_campaign_name() -> str:
    return datetime.now().strftime("paper_%Y%m%d_%H%M%S")


def load_tuned_configs(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    tuned_path = Path(path)
    with tuned_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Tuned config selection must be a mapping: {tuned_path}")
    return data


def infer_campaign_root_from_out(out: str | Path) -> Path:
    path = Path(out)
    if path.name and path.parent.name == "results":
        return path.parent.parent
    return path


def result_dir_from_out(out: str | Path, block_id: str) -> Path:
    """Normalize block output to ``<campaign>/results/<block_id>``."""

    path = Path(out)
    if path.name == block_id and path.parent.name == "results":
        return path
    return infer_campaign_root_from_out(path) / "results" / block_id


def ensure_result_block_dirs(campaign_root: Path, block_ids: Iterable[str]) -> None:
    for block_id in block_ids:
        for subdir in ("raw", "summary", "tables", "figures"):
            (campaign_root / "results" / block_id / subdir).mkdir(parents=True, exist_ok=True)


def effective_seeds(num_seeds: int, quick: bool) -> list[int]:
    return seed_list(2 if quick else num_seeds)


def _horizon(problem: str, quick: bool) -> int:
    return QUICK_HORIZON[problem] if quick else FULL_HORIZON[problem]


def _paper_degree(problem: str, overrides: dict[str, Any]) -> int | None:
    if problem == "pricing":
        value = overrides.get("pricing_context_degree", overrides.get("paper_degree"))
    elif problem in {"topk", "shortest_path"}:
        value = overrides.get("deg", overrides.get("paper_degree"))
    else:
        value = overrides.get("paper_degree")
    if value in {None, ""}:
        return None
    return int(value)


def _tuned_key(problem: str, model_type: str, degree: int | None) -> str | None:
    if degree is None:
        return None
    return f"{problem}/deg{int(degree)}/{model_type}"


def _apply_default_adaptive_alpha(overrides: dict[str, Any]) -> None:
    for key, value in ADAPTIVE_ALPHA_OVERRIDES.items():
        overrides.setdefault(key, value)


def _apply_tuned_config(problem: str, model_type: str, overrides: dict[str, Any], opts: BuildOptions) -> None:
    tuned_configs = opts.tuned_configs
    if tuned_configs is None:
        if opts.require_tuned_configs and problem in {"topk", "shortest_path", "pricing"}:
            raise ValueError("--require-tuned-configs was set but no --tuned-configs file was provided")
        overrides["paper_tuned_config_applied"] = False
        return

    degree = _paper_degree(problem, overrides)
    selected_key = _tuned_key(problem, model_type, degree)
    selected = tuned_configs.get(selected_key) if selected_key else None
    if selected is None:
        if opts.require_tuned_configs and problem in {"topk", "shortest_path", "pricing"}:
            raise KeyError(
                f"Missing tuned config for problem={problem!r}, model_type={model_type!r}, "
                f"degree={degree!r}; expected key {selected_key!r} in {opts.tuned_config_path}."
            )
        overrides.update(
            {
                "paper_tuned_config_applied": False,
                "paper_tuned_config_missing_key": selected_key or "",
            }
        )
        return

    applied_keys: list[str] = []
    for key in TUNED_CONFIG_KEYS:
        value = selected.get(key)
        if value is None:
            continue
        if key == "policy_sampling_scale" and model_type in GENERATIVE_MODEL_TYPES:
            continue
        overrides[key] = value
        applied_keys.append(key)

    if model_type in GENERATIVE_MODEL_TYPES and selected.get("theta_lr") is not None:
        # Generative models read generative_lr before theta_lr, so set both
        # when a tuned theta learning rate is selected for CNF/Diffusion.
        overrides["generative_lr"] = float(selected["theta_lr"])
        applied_keys.append("generative_lr")

    tuned_metadata = {key: selected[key] for key in TUNED_METADATA_KEYS if selected.get(key) is not None}
    overrides.update(
        {
            "paper_tuned_config_applied": True,
            "paper_tuned_config_key": str(selected_key),
            "paper_tuned_config_source": "" if opts.tuned_config_path is None else _repo_rel(opts.tuned_config_path),
            "paper_tuned_config_candidate_id": str(selected.get("hybrid_tuning_candidate_id", "")),
            "paper_tuned_config_applied_keys": applied_keys,
            "paper_tuned_config_metadata": tuned_metadata,
        }
    )


def _all_algos_off() -> dict[str, bool]:
    return {key: False for key in ALGORITHM_TOGGLES}


def _base_flags() -> dict[str, Any]:
    flags = _all_algos_off()
    flags.update(
        {
            "run_random_oracle": False,
            "run_true_model": False,
            "overwrite_output": False,
            "merge_output_dir": False,
        }
    )
    return flags


def _enable_algorithm(algo_key: str, overrides: dict[str, Any]) -> None:
    if algo_key == "adaptive_hybrid":
        overrides["run_hybrid_bandit"] = True
        return
    if algo_key == "surrogate_only":
        overrides["run_hybrid_bandit"] = True
        overrides["hybrid_alpha_schedule"] = "constant"
        overrides["hybrid_alpha_init"] = 0.0
        overrides["hybrid_alpha_final"] = 0.0
        return
    if algo_key == "score_only":
        overrides["run_hybrid_bandit"] = True
        overrides["hybrid_alpha_schedule"] = "constant"
        overrides["hybrid_alpha_init"] = 1.0
        overrides["hybrid_alpha_final"] = 1.0
        overrides["policy_baseline_type"] = "ema"
        return
    if algo_key == "greedycb":
        overrides["run_greedy_contextual_bandit"] = True
        return
    if algo_key == "epsgreedycb":
        overrides["run_epsilon_greedy_contextual_bandit"] = True
        overrides["epsilon_greedy_epsilon"] = 0.1
        return
    if algo_key == "tscb":
        overrides["run_thompson_contextual_bandit"] = True
        return
    raise ValueError(f"Unknown paper algorithm key: {algo_key}")


def _model_label(model_type: str) -> str:
    mapping = {
        "nn": "two_layer_nn",
        "shared_nn": "two_layer_nn",
        "linear": "linear",
        "shared_linear": "linear",
        "diffusion": "diffusion",
        "shared_diffusion": "diffusion",
        "cnf": "cnf",
        "shared_cnf": "cnf",
    }
    return mapping.get(str(model_type), str(model_type))


def _set_problem_degree(overrides: dict[str, Any], problem: str, degree: int) -> None:
    if problem in {"topk", "shortest_path"}:
        overrides.update({"deg": int(degree), "eps_bar": 0.5})
    elif problem == "pricing":
        overrides.update({"pricing_context_degree": int(degree), "pricing_price_degree": int(degree)})
    overrides["paper_degree"] = int(degree)


def _problem_overrides(
    problem: str,
    model_type: str,
    opts: BuildOptions,
    *,
    degree: int | None = DEFAULT_PAPER_DEGREE,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        **_base_flags(),
        "benchmark": BENCHMARK_BY_PROBLEM[problem],
        "T": _horizon(problem, opts.quick),
        "model_type": model_type,
        "torch_device": opts.device,
        "paper_problem": problem,
        "paper_problem_label": problem,
        "paper_point_model": _model_label(model_type),
    }
    if opts.eval_every is not None:
        overrides["eval_every"] = int(opts.eval_every)

    if problem == "topk":
        _set_problem_degree(overrides, problem, int(degree or DEFAULT_PAPER_DEGREE))
    elif problem == "shortest_path":
        _set_problem_degree(overrides, problem, int(degree or DEFAULT_PAPER_DEGREE))
    elif problem == "pricing":
        _set_problem_degree(overrides, problem, int(degree or DEFAULT_PAPER_DEGREE))
    _apply_tuned_config(problem, model_type, overrides, opts)
    return overrides


def _feedback_override(regime: str) -> dict[str, str]:
    mapping = {
        "scalar_bandit": "bandit",
        "semi_bandit": "semi_bandit",
        "full_information": "full_feedback",
    }
    return {"feedback_mode": mapping[regime], "paper_feedback": regime}


def _apply_feedback_lr_calibration(problem: str, regime: str, overrides: dict[str, Any]) -> None:
    if problem != "pricing":
        return

    if regime not in {"full_information", "semi_bandit"}:
        return

    overrides.update(
        {
            "theta_lr": 0.05,
            "theta_lr_schedule": "shifted_inverse_time",
            "theta_lr_offset": 100,
            "nuisance_lr": 0.05,
            "nuisance_lr_schedule": "shifted_inverse_time",
            "nuisance_lr_offset": 100,
            "hybrid_loss_type": "spo_plus",
            "hybrid_alpha_max": 0.5,
            "hybrid_alpha_min": 0.05,
            "hybrid_alpha_warmup_frac": 0.05,
            "hybrid_alpha_ema_decay": 0.98,
            "hybrid_alpha_smooth": 0.05,
            "policy_sampling_scale": 0.0003,
            "paper_feedback_config_calibration": "pricing_linear_old_deg6_feedback_recipe_nuisance_fix",
        }
    )


def _primary_method_label(algo_key: str, extra: dict[str, Any] | None = None) -> str:
    if algo_key == "adaptive_hybrid" and extra and extra.get("fixed_alpha") is not None:
        return f"DFHPG fixed alpha={float(extra['fixed_alpha']):g}"
    return PRIMARY_METHOD_LABEL[algo_key]


def _reference_payload(config: dict[str, Any]) -> dict[str, Any]:
    payload = {}
    for key in REFERENCE_CONFIG_FIELDS:
        value = config.get(key)
        if value not in {None, ""}:
            payload[key] = value
    return payload


def reference_key_from_config(config: dict[str, Any]) -> str:
    payload = _reference_payload(config)
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    problem = safe_name(config.get("paper_problem", config.get("benchmark", "ref")))
    seed = safe_name(f"seed{config.get('seed', 'unknown')}")
    degree = config.get("paper_degree", config.get("deg", config.get("pricing_context_degree", "")))
    degree_slug = "" if degree in {None, ""} else f"_deg{safe_name(degree)}"
    eps = config.get("eps_bar", "")
    eps_slug = "" if eps in {None, "", 0, 0.0} else f"_eps{safe_name(eps)}"
    return f"{problem}{degree_slug}{eps_slug}_{seed}_{digest}"


def reference_output_dir(campaign_root: Path, reference_key: str) -> Path:
    return campaign_root / "references" / "true_model" / reference_key


def _experiment(
    *,
    block_id: str,
    problem: str,
    model_type: str,
    algo_key: str,
    seed: int,
    output_root: Path,
    overrides: dict[str, Any],
    extra_slug: str = "",
    method_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primary_algo = PRIMARY_ALGO_DIR[algo_key]
    method = _primary_method_label(algo_key, method_extra)
    parts = [block_id, problem, _model_label(model_type), algo_key]
    if extra_slug:
        parts.append(extra_slug)
    parts.append(f"seed{seed}")
    run_name = "paper_" + "_".join(safe_name(part) for part in parts)
    merged = dict(overrides)
    _enable_algorithm(algo_key, merged)
    merged.update(
        {
            "seed": seed,
            "paper_block_id": block_id,
            "paper_algorithm_key": algo_key,
            "paper_method": method,
            "paper_algo_internal": primary_algo,
            "paper_actor_family": merged.get("paper_actor_family", "gaussian"),
        }
    )
    reference_key = reference_key_from_config(merged)
    merged.update(
        {
            "paper_reference_key": reference_key,
            "paper_reference_output_dir": _repo_rel(reference_output_dir(output_root.parent, reference_key)),
        }
    )
    execution_config = _cached_base_config(BASE_BY_PROBLEM[problem])
    execution_config.update(merged)
    execution_key, execution_output_name = _execution_output_name(
        base=BASE_BY_PROBLEM[problem],
        problem=problem,
        model_type=model_type,
        algo_key=algo_key,
        seed=seed,
        config=execution_config,
    )
    merged.update(
        {
            "output_dir": _repo_rel(output_root / execution_output_name),
            "paper_execution_key": execution_key,
            "paper_execution_output_name": execution_output_name,
            "paper_manifest_run_name": run_name,
        }
    )
    return {
        "name": run_name,
        "base": BASE_BY_PROBLEM[problem],
        "overrides": merged,
        "seeds": [seed],
        "metadata": {
            "paper_block_id": block_id,
            "method": method,
            "algo_internal": primary_algo,
            "problem": problem,
            "seed": seed,
        },
    }


def _experiments_for(
    *,
    block_id: str,
    opts: BuildOptions,
    output_root: Path,
    problem: str,
    model_type: str,
    algo_keys: Iterable[str],
    common: dict[str, Any],
    extra_slug: str = "",
    method_extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    experiments = []
    for algo_key in algo_keys:
        for seed in effective_seeds(opts.num_seeds, opts.quick):
            experiments.append(
                _experiment(
                    block_id=block_id,
                    problem=problem,
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


def _point_models(problem: str) -> tuple[str, str]:
    if problem == "pricing":
        return ("shared_linear", "shared_nn")
    return ("linear", "nn")


def _nn_model(problem: str) -> str:
    return "shared_nn" if problem == "pricing" else "nn"


def _linear_model(problem: str) -> str:
    return "shared_linear" if problem == "pricing" else "linear"


def build_block_01(opts: BuildOptions, output_root: Path) -> list[dict[str, Any]]:
    algos = ("adaptive_hybrid", "surrogate_only", "score_only", "greedycb", "epsgreedycb", "tscb")
    experiments = []
    for problem in ("topk", "shortest_path", "pricing"):
        for degree in PAPER_DEGREES:
            for model_type in _point_models(problem):
                common = _problem_overrides(problem, model_type, opts, degree=degree)
                _apply_default_adaptive_alpha(common)
                experiments.extend(
                    _experiments_for(
                        block_id="01_main_point_models",
                        opts=opts,
                        output_root=output_root,
                        problem=problem,
                        model_type=model_type,
                        algo_keys=algos,
                        common=common,
                        extra_slug=f"deg{degree}",
                    )
                )
    return experiments


def _generative_model(problem: str, actor_family: str) -> str:
    shared = problem == "pricing"
    if actor_family == "gaussian_linear":
        return "shared_linear" if shared else "linear"
    if actor_family == "gaussian_nn":
        return "shared_nn" if shared else "nn"
    if actor_family == "cnf":
        return "shared_cnf" if shared else "cnf"
    if actor_family == "diffusion":
        return "shared_diffusion" if shared else "diffusion"
    raise ValueError(actor_family)


def build_block_02(opts: BuildOptions, output_root: Path) -> list[dict[str, Any]]:
    experiments = []
    for problem in ("topk", "pricing", "shortest_path"):
        degree = DEFAULT_PAPER_DEGREE
        for actor_family in ("gaussian_linear", "gaussian_nn", "cnf", "diffusion"):
            model_type = _generative_model(problem, actor_family)
            common = _problem_overrides(problem, model_type, opts, degree=degree)
            if not opts.quick:
                common["T"] = BLOCK02_FULL_HORIZON[problem]
            common.update(
                {
                    "paper_actor_family": actor_family,
                    "paper_feedback": "scalar_bandit",
                }
            )
            _apply_default_adaptive_alpha(common)
            experiments.extend(
                _experiments_for(
                    block_id="02_generative_ablation",
                    opts=opts,
                    output_root=output_root,
                    problem=problem,
                    model_type=model_type,
                    algo_keys=("adaptive_hybrid",),
                    common=common,
                    extra_slug=f"{actor_family}_deg{degree}",
                )
            )
    return experiments


def build_block_03(opts: BuildOptions, output_root: Path) -> list[dict[str, Any]]:
    experiments = []
    for problem in ("topk", "pricing", "shortest_path"):
        model_type = _linear_model(problem)
        adaptive = _problem_overrides(problem, model_type, opts)
        _apply_default_adaptive_alpha(adaptive)
        adaptive["paper_feedback"] = "scalar_bandit"
        experiments.extend(
            _experiments_for(
                block_id="03_alpha_diagnostics",
                opts=opts,
                output_root=output_root,
                problem=problem,
                model_type=model_type,
                algo_keys=("adaptive_hybrid",),
                common=adaptive,
                extra_slug="adaptive_alpha",
            )
        )
        for fixed_alpha in FIXED_ALPHA_DIAGNOSTIC_GRID:
            common = _problem_overrides(problem, model_type, opts)
            common.update(
                {
                    "paper_feedback": "scalar_bandit",
                    "hybrid_alpha_schedule": "constant",
                    "hybrid_alpha_init": fixed_alpha,
                    "hybrid_alpha_final": fixed_alpha,
                }
            )
            experiments.extend(
                _experiments_for(
                    block_id="03_alpha_diagnostics",
                    opts=opts,
                    output_root=output_root,
                    problem=problem,
                    model_type=model_type,
                    algo_keys=("adaptive_hybrid",),
                    common=common,
                    extra_slug=f"fixed_alpha_{fixed_alpha:g}".replace(".", "p"),
                    method_extra={"fixed_alpha": fixed_alpha},
                )
            )
    return experiments


def build_block_04(opts: BuildOptions, output_root: Path) -> list[dict[str, Any]]:
    experiments = []
    for problem in ("topk", "shortest_path", "pricing"):
        model_type = _linear_model(problem)
        for regime in ("full_information", "semi_bandit", "scalar_bandit"):
            common = _problem_overrides(problem, model_type, opts)
            common.update(_feedback_override(regime))
            _apply_feedback_lr_calibration(problem, regime, common)
            _apply_default_adaptive_alpha(common)
            experiments.extend(
                _experiments_for(
                    block_id="04_feedback_ablation",
                    opts=opts,
                    output_root=output_root,
                    problem=problem,
                    model_type=model_type,
                    algo_keys=("adaptive_hybrid", "surrogate_only", "greedycb", "epsgreedycb", "tscb"),
                    common=common,
                    extra_slug=regime,
                )
            )
    return experiments


def build_block_05(opts: BuildOptions, output_root: Path) -> list[dict[str, Any]]:
    cases = (
        ("topk", DEFAULT_PAPER_DEGREE),
        ("shortest_path", DEFAULT_PAPER_DEGREE),
        ("pricing", DEFAULT_PAPER_DEGREE),
    )
    experiments = []
    for problem, degree in cases:
        model_type = _linear_model(problem)
        for loss_name in PAPER_LOSSES:
            canonical = canonical_surrogate_loss_name(loss_name)
            display = surrogate_loss_display_name(loss_name)
            for algo_key in ("adaptive_hybrid", "surrogate_only"):
                common = _problem_overrides(problem, model_type, opts)
                if problem == "pricing":
                    common.update({"pricing_context_degree": degree, "pricing_price_degree": degree})
                else:
                    common["deg"] = degree
                _apply_tuned_config(problem, model_type, common, opts)
                common.update(
                    {
                        "paper_degree": degree,
                        "paper_loss": display,
                        "hybrid_loss_type": canonical,
                    }
                )
                common.update(loss_mix_overrides(canonical, weight_key="hybrid_mse_weight"))
                _apply_default_adaptive_alpha(common)
                experiments.extend(
                    _experiments_for(
                        block_id="05_surrogate_loss_grid",
                        opts=opts,
                        output_root=output_root,
                        problem=problem,
                        model_type=model_type,
                        algo_keys=(algo_key,),
                        common=common,
                        extra_slug=f"deg{degree}_{safe_name(display)}",
                    )
                )
    return experiments


BLOCK_SPECS: dict[str, BlockSpec] = {
    "01_main_point_models": BlockSpec("01_main_point_models", "main point-model degree sweep", 30, build_block_01),
    "02_generative_ablation": BlockSpec("02_generative_ablation", "degree-8 generative actor comparison", 30, build_block_02),
    "03_alpha_diagnostics": BlockSpec("03_alpha_diagnostics", "adaptive-alpha diagnostics", 30, build_block_03),
    "04_feedback_ablation": BlockSpec("04_feedback_ablation", "feedback-regime comparison", 30, build_block_04),
    "05_surrogate_loss_grid": BlockSpec("05_surrogate_loss_grid", "surrogate-loss comparison", 30, build_block_05),
}

DEFAULT_BLOCK_IDS = tuple(BLOCK_SPECS)


def build_block_manifest(block_id: str, opts: BuildOptions, output_root: Path) -> dict[str, Any]:
    spec = BLOCK_SPECS[block_id]
    return {
        "metadata": {
            "paper_block_id": block_id,
            "title": spec.title,
            "quick": opts.quick,
            "num_seeds": 2 if opts.quick else opts.num_seeds,
            "git_hash": _git_hash(),
            "tuned_config_source": "" if opts.tuned_config_path is None else _repo_rel(opts.tuned_config_path),
            "require_tuned_configs": opts.require_tuned_configs,
        },
        "experiments": spec.builder(opts, output_root),
    }


def _reference_model_type(config: dict[str, Any]) -> str:
    return "shared_linear" if str(config.get("benchmark")) in {"pricing", "energy"} else "linear"


def _reference_experiment_from(exp: dict[str, Any]) -> dict[str, Any]:
    overrides = dict(exp.get("overrides", {}))
    reference_key = str(overrides["paper_reference_key"])
    reference_output = str(overrides["paper_reference_output_dir"])
    for key in ALGORITHM_TOGGLES:
        overrides[key] = False
    overrides.update(
        {
            "run_true_model": True,
            "run_random_oracle": False,
            "model_type": _reference_model_type(overrides),
            "output_dir": reference_output,
            "paper_reference_run": True,
            "paper_algorithm_key": "true_model_reference",
            "paper_method": "TrueModel",
            "paper_algo_internal": "TrueModel",
            "paper_reference_key": reference_key,
            "paper_reference_output_dir": reference_output,
        }
    )
    seed = int(overrides.get("seed", exp.get("metadata", {}).get("seed", exp.get("seeds", [42])[0])))
    name = f"paper_ref_true_model_{reference_key}"
    return {
        "name": name,
        "base": exp["base"],
        "overrides": overrides,
        "seeds": [seed],
        "metadata": {
            "paper_reference_run": True,
            "paper_reference_key": reference_key,
            "method": "TrueModel",
            "algo_internal": "TrueModel",
            "problem": overrides.get("paper_problem", overrides.get("benchmark", "")),
            "seed": seed,
        },
    }


def build_reference_manifest(experiments: Iterable[dict[str, Any]]) -> dict[str, Any]:
    references: dict[str, dict[str, Any]] = {}
    for exp in experiments:
        overrides = exp.get("overrides", {})
        reference_key = overrides.get("paper_reference_key")
        if not reference_key:
            continue
        references.setdefault(str(reference_key), _reference_experiment_from(exp))
    return {
        "metadata": {
            "paper_reference_manifest": True,
            "reference": "TrueModel",
            "git_hash": _git_hash(),
        },
        "experiments": [references[key] for key in sorted(references)],
    }


def campaign_paths(campaign_root: Path) -> dict[str, Path]:
    return {
        "root": campaign_root,
        "manifests": campaign_root / "manifests",
        "slurm": campaign_root / "slurm",
        "outputs": campaign_root / "outputs",
        "references": campaign_root / "references",
        "results": campaign_root / "results",
        "logs": campaign_root / "logs",
    }


def write_campaign_manifests(
    *,
    campaign_root: Path,
    quick: bool,
    num_seeds: int | None,
    device: str,
    eval_every: int | None = None,
    tuned_config_path: Path | None = None,
    allow_missing_tuned_configs: bool = False,
    block_ids: Iterable[str] | None = None,
    dry_run: bool = False,
) -> tuple[list[Path], dict[str, Any]]:
    paths = campaign_paths(campaign_root)
    selected = list(block_ids or DEFAULT_BLOCK_IDS)
    if not dry_run:
        for key in ("manifests", "slurm", "outputs", "results", "logs"):
            paths[key].mkdir(parents=True, exist_ok=True)
        ensure_result_block_dirs(campaign_root, selected)
    opts = BuildOptions(
        campaign_root=campaign_root,
        quick=quick,
        num_seeds=int(num_seeds or max(BLOCK_SPECS[block_id].default_num_seeds for block_id in selected)),
        device=device,
        eval_every=eval_every,
        tuned_configs=load_tuned_configs(tuned_config_path),
        tuned_config_path=tuned_config_path,
        require_tuned_configs=tuned_config_path is not None and not allow_missing_tuned_configs,
    )
    manifests: list[Path] = []
    summary: dict[str, Any] = {
        "blocks": {},
        "total_experiment_groups": 0,
        "tuned_config_source": "" if tuned_config_path is None else _repo_rel(tuned_config_path),
        "require_tuned_configs": tuned_config_path is not None and not allow_missing_tuned_configs,
    }
    for block_id in selected:
        block_opts = BuildOptions(
            campaign_root=campaign_root,
            quick=quick,
            num_seeds=int(num_seeds or BLOCK_SPECS[block_id].default_num_seeds),
            device=device,
            eval_every=eval_every,
            tuned_configs=opts.tuned_configs,
            tuned_config_path=tuned_config_path,
            require_tuned_configs=opts.require_tuned_configs,
        )
        manifest = build_block_manifest(block_id, block_opts, paths["outputs"])
        manifest_path = paths["manifests"] / f"block_{block_id}.yaml"
        manifests.append(manifest_path)
        count = len(manifest["experiments"])
        summary["blocks"][block_id] = count
        summary["total_experiment_groups"] += count
        if not dry_run:
            write_yaml(manifest_path, manifest)

    compare_path = paths["manifests"] / "compare_all.yaml"
    if not dry_run:
        write_yaml(compare_path, {"include": [path.name for path in manifests]})
    return manifests, summary


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_compare_experiments(compare_path: Path) -> list[dict[str, Any]]:
    data = _load_yaml(compare_path)
    experiments = []
    for item in data.get("include", []):
        manifest_path = (compare_path.parent / str(item)).resolve()
        experiments.extend(_load_yaml(manifest_path).get("experiments", []))
    return experiments


def dedupe_experiments_for_execution(experiments: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    skipped = 0
    for exp in experiments:
        cfg = dict(exp.get("overrides", {}))
        if "seed" not in cfg:
            cfg["seed"] = exp.get("seeds", [42])[0]
        algo_dirs = tuple(_expected_algo_dirs(cfg))
        key = (str(cfg.get("output_dir", "")), ",".join(algo_dirs))
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        unique.append(exp)
    return unique, skipped


def filter_completed_experiments(
    experiments: Iterable[dict[str, Any]], *, resume: bool
) -> tuple[list[dict[str, Any]], int]:
    if not resume:
        return list(experiments), 0
    remaining: list[dict[str, Any]] = []
    skipped = 0
    for exp in experiments:
        cfg = dict(exp.get("overrides", {}))
        if "seed" not in cfg:
            cfg["seed"] = exp.get("seeds", [42])[0]
        output_dir = REPO_ROOT / str(cfg["output_dir"])
        if _metrics_path_complete(output_dir, _expected_algo_dirs(cfg)):
            skipped += 1
            continue
        remaining.append(exp)
    return remaining, skipped


def write_slurm_runner(
    *,
    campaign_root: Path,
    compare_path: Path,
    max_array_jobs: int,
    partition: str,
    qos: str,
    gpu_type: str,
    account: str,
    time_limit: str,
    job_name: str,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    paths = campaign_paths(campaign_root)
    manifest_experiments = load_compare_experiments(compare_path)
    execution_experiments, deduped_experiments = dedupe_experiments_for_execution(manifest_experiments)
    experiments, skipped_completed = filter_completed_experiments(execution_experiments, resume=resume)
    if not manifest_experiments:
        raise ValueError(f"No experiments found in {compare_path}")
    experiments_per_task = 0 if not experiments else max(1, math.ceil(len(experiments) / max_array_jobs))
    array_jobs = 0 if not experiments else math.ceil(len(experiments) / experiments_per_task)
    meta = {
        "manifest": _repo_rel(compare_path),
        "manifest_experiment_groups": len(manifest_experiments),
        "unique_execution_groups": len(execution_experiments),
        "experiment_groups": len(experiments),
        "deduped_experiment_groups": deduped_experiments,
        "skipped_completed": skipped_completed,
        "experiments_per_task": experiments_per_task,
        "array_jobs": array_jobs,
        "array_spec": "" if array_jobs == 0 else f"1-{array_jobs}%{array_jobs}",
        "partition": partition,
        "qos": qos,
        "gpu_type": gpu_type,
        "job_name": job_name,
        "submit": _repo_rel(paths["slurm"] / "submit.sbatch"),
    }
    if dry_run:
        return meta

    shard_dir = paths["slurm"] / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    paths["logs"].mkdir(parents=True, exist_ok=True)
    for stale_shard in shard_dir.glob("shard_*.yaml"):
        stale_shard.unlink()
    if not experiments:
        stale_submit = paths["slurm"] / "submit.sbatch"
        if stale_submit.exists():
            stale_submit.unlink()
        meta["submit"] = ""
        (paths["slurm"] / "meta.yaml").write_text(
            "\n".join(f"{key}: {value}" for key, value in meta.items()) + "\n",
            encoding="utf-8",
        )
        return meta
    for index in range(array_jobs):
        chunk = experiments[index * experiments_per_task : (index + 1) * experiments_per_task]
        write_yaml(shard_dir / f"shard_{index + 1:03d}.yaml", {"experiments": chunk})

    (paths["slurm"] / "meta.yaml").write_text(
        "\n".join(f"{key}: {value}" for key, value in meta.items()) + "\n",
        encoding="utf-8",
    )
    submit_path = paths["slurm"] / "submit.sbatch"
    submit_text = "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --account={account}",
            "#SBATCH -N1",
            f"#SBATCH --gres=gpu:{gpu_type}:1",
            "#SBATCH --gres-flags=enforce-binding",
            f"#SBATCH -p {partition}",
            f"#SBATCH -q {qos}",
            f"#SBATCH -t {time_limit}",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output={paths['logs'].resolve()}/{job_name}_%A_%a.out",
            f"#SBATCH --array=1-{array_jobs}%{array_jobs}",
            "",
            "set -euo pipefail",
            f'REPO_ROOT="{REPO_ROOT}"',
            f'CAMPAIGN_ROOT="{campaign_root.resolve()}"',
            'SHARD_DIR="$CAMPAIGN_ROOT/slurm/shards"',
            'LOG_DIR="$CAMPAIGN_ROOT/logs"',
            'mkdir -p "$LOG_DIR"',
            'cd "$REPO_ROOT"',
            "set +u",
            "module load gurobi >/dev/null 2>&1",
            "module load mamba >/dev/null 2>&1",
            "mamba activate dfl",
            "set -u",
            "export PYTHONUNBUFFERED=1",
            "export OMP_NUM_THREADS=1",
            "export CUDA_VISIBLE_DEVICES=0",
            'shard_path=$(printf "%s/shard_%03d.yaml" "$SHARD_DIR" "$SLURM_ARRAY_TASK_ID")',
            f'task_log=$(printf "%s/{job_name}_A%s_a%s.log" "$LOG_DIR" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID")',
            'python scripts/utils/run_experiments.py --config "$shard_path" --no-dedup-baselines > "$task_log" 2>&1',
            "",
        ]
    )
    submit_path.write_text(submit_text, encoding="utf-8")
    submit_path.chmod(0o755)
    return meta


def _metrics_path_complete(output_dir: Path, algo_dirs: Iterable[str]) -> bool:
    for algo in algo_dirs:
        if algo == "DFHPG":
            if not any((output_dir / name / "metrics.json").exists() for name in ("DFHPG", _LEGACY_DFHPG_ALGO_DIR)):
                return False
            continue
        aliases = (algo, *LEGACY_ALGO_DIRS.get(algo, ()))
        if not any((output_dir / name / "metrics.json").exists() for name in aliases):
            return False
    return True


def _expected_algo_dirs(overrides: dict[str, Any]) -> list[str]:
    if bool(overrides.get("paper_reference_run", False)):
        return ["TrueModel"]
    dirs = []
    primary = overrides.get("paper_algo_internal")
    if primary:
        dirs.append(str(primary))
    return dirs


def _run_config_from_experiment(exp: dict[str, Any]) -> dict[str, Any]:
    base_path = REPO_ROOT / str(exp["base"])
    cfg = load_config(str(base_path))
    cfg.update(exp.get("overrides", {}))
    cfg["seed"] = exp.get("seeds", [cfg.get("seed", 42)])[0]
    cfg["experiment_name"] = exp["name"]
    return cfg


def execute_manifest(manifest: dict[str, Any], *, resume: bool, dry_run: bool) -> tuple[int, int, int]:
    completed = 0
    skipped = 0
    failed = 0
    for exp in manifest.get("experiments", []):
        cfg = _run_config_from_experiment(exp)
        output_dir = REPO_ROOT / str(cfg["output_dir"])
        if resume and _metrics_path_complete(output_dir, _expected_algo_dirs(cfg)):
            skipped += 1
            continue
        if dry_run:
            print(f"[DRY RUN] {exp['name']} -> {cfg['output_dir']}")
            completed += 1
            continue
        from src.experiment.engine import ExperimentRunner

        try:
            ExperimentRunner(cfg, verbose=True).run()
            completed += 1
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {exp['name']}: {exc}", file=sys.stderr)
    return completed, skipped, failed


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _series(metrics: dict[str, Any]) -> np.ndarray | None:
    values = metrics.get("cum_expected_objective") or metrics.get("cum_objective")
    if not isinstance(values, list) or not values:
        return None
    return np.asarray(values, dtype=float)


def _regret_series(metrics: dict[str, Any], true_metrics: dict[str, Any]) -> np.ndarray | None:
    values = _series(metrics)
    true_values = _series(true_metrics)
    if values is None or true_values is None:
        return None
    n = min(len(values), len(true_values))
    if n == 0:
        return None
    values = values[:n]
    true_values = true_values[:n]
    sense = str(metrics.get("objective_sense", true_metrics.get("objective_sense", "min"))).lower()
    if sense == "max":
        return true_values - values
    return values - true_values


def _relative_regret_series(metrics: dict[str, Any], true_metrics: dict[str, Any]) -> np.ndarray | None:
    regret = _regret_series(metrics, true_metrics)
    true_values = _series(true_metrics)
    if regret is None or true_values is None:
        return None
    n = min(len(regret), len(true_values))
    if n == 0:
        return None
    denominator = np.maximum(np.abs(true_values[:n]), 1.0e-12)
    return regret[:n] / denominator


def _true_reference_metrics_path(run_dir: Path, cfg: dict[str, Any]) -> Path:
    reference_output = cfg.get("paper_reference_output_dir")
    if reference_output:
        return REPO_ROOT / str(reference_output) / "TrueModel" / "metrics.json"
    return run_dir / "TrueModel" / "metrics.json"


def _method_for_algo(algo_name: str, cfg: dict[str, Any]) -> tuple[str, str]:
    legacy_method_by_algo = {
        _LEGACY_GREEDY_CB_ALGO_DIR: "GreedyCB",
        "GreedyContextualBandit": "GreedyCB",
        _LEGACY_EPS_GREEDY_CB_ALGO_DIR: "$\\epsilon$-GreedyCB",
        "EpsilonGreedyContextualBandit": "$\\epsilon$-GreedyCB",
        _LEGACY_TS_CB_ALGO_DIR: "TSCB",
        "ThompsonSamplingContextualBandit": "TSCB",
    }
    if algo_name in legacy_method_by_algo:
        return legacy_method_by_algo[algo_name], algo_name
    if algo_name in BASELINE_METHOD_LABEL:
        return BASELINE_METHOD_LABEL[algo_name], algo_name
    if algo_name == cfg.get("paper_algo_internal"):
        method = str(cfg.get("paper_method", algo_name))
        return PAPER_METHOD_RENAMES.get(method, method), algo_name
    if algo_name in {"DFHPG", _LEGACY_DFHPG_ALGO_DIR}:
        primary = str(cfg.get("paper_algo_internal", ""))
        if primary in {"DFHPG", _LEGACY_DFHPG_ALGO_DIR}:
            method = str(cfg.get("paper_method", algo_name))
            return PAPER_METHOD_RENAMES.get(method, method), algo_name
        return "DFHPG", algo_name
    return algo_name, algo_name


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _mean_sem(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    sem = 0.0 if len(arr) < 2 else float(np.std(arr, ddof=1) / math.sqrt(len(arr)))
    return mean, sem


def _aggregate_sources(block_id: str, output_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    manifest_path = output_root.parent / "manifests" / f"block_{block_id}.yaml"
    if manifest_path.exists():
        data = _load_yaml(manifest_path)
        sources: list[tuple[Path, dict[str, Any]]] = []
        for exp in data.get("experiments", []):
            cfg = _run_config_from_experiment(exp)
            sources.append((REPO_ROOT / str(cfg["output_dir"]), cfg))
        return sources

    sources = []
    for run_dir in sorted(output_root.iterdir() if output_root.exists() else []):
        cfg_path = run_dir / "config.yaml"
        if not cfg_path.exists():
            continue
        with cfg_path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        if str(cfg.get("paper_block_id", "")) != block_id:
            continue
        sources.append((run_dir, cfg))
    return sources


def aggregate_block_results(block_id: str, *, result_dir: Path, output_root: Path) -> dict[str, Any]:
    trace_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for run_dir, cfg in _aggregate_sources(block_id, output_root):
        if not run_dir.exists():
            continue
        true_path = _true_reference_metrics_path(run_dir, cfg)
        if not true_path.exists():
            continue
        true_metrics = _load_json(true_path)
        for metrics_path in sorted(run_dir.glob("*/metrics.json")):
            algo_name = metrics_path.parent.name
            metrics = _load_json(metrics_path)
            values = _series(metrics)
            regret = _regret_series(metrics, true_metrics)
            relative_regret = _relative_regret_series(metrics, true_metrics)
            if values is None:
                continue
            method, algo_internal = _method_for_algo(algo_name, cfg)
            seed = int(cfg.get("seed", 0))
            diagnostics = metrics.get("diagnostics", [])
            if not isinstance(diagnostics, list):
                diagnostics = []
            rows = min(len(values), 0 if regret is None else len(regret))
            if regret is None:
                rows = len(values)
            for idx in range(rows):
                diag = diagnostics[idx] if idx < len(diagnostics) and isinstance(diagnostics[idx], dict) else {}
                trace_rows.append(
                    {
                        "seed": seed,
                        "problem": cfg.get("paper_problem_label", cfg.get("benchmark", "")),
                        "benchmark": cfg.get("benchmark", ""),
                        "point_model": cfg.get("paper_point_model", cfg.get("model_type", "")),
                        "actor_family": cfg.get("paper_actor_family", ""),
                        "feedback": cfg.get("paper_feedback", "scalar_bandit"),
                        "degree": cfg.get("paper_degree", cfg.get("deg", cfg.get("pricing_context_degree", ""))),
                        "round": idx + 1,
                        "method": method,
                        "algo_internal": algo_internal,
                        "cum_cost": float(values[idx]),
                        "cum_regret": "" if regret is None else float(regret[idx]),
                        "relative_regret": "" if relative_regret is None else float(relative_regret[idx]),
                        "alpha_t": diag.get("alpha_t", ""),
                        "score_grad_norm": diag.get("score_grad_norm", ""),
                        "plugin_grad_norm": diag.get("plugin_grad_norm", ""),
                        "combined_grad_norm": diag.get("combined_grad_norm", ""),
                        "nuisance_loss": diag.get("nuisance_loss", diag.get("critic_loss", "")),
                        "scalar_loss": diag.get("scalar_loss", ""),
                        "scalar_feedback": diag.get("scalar_feedback", ""),
                        "adaptive_residual_ema": diag.get("adaptive_residual_ema", ""),
                        "adaptive_scale_ema": diag.get("adaptive_scale_ema", ""),
                        "adaptive_residual_ratio_ref": diag.get("adaptive_residual_ratio_ref", ""),
                    }
                )
            final_regret = "" if regret is None or len(regret) == 0 else float(regret[-1])
            final_relative_regret = (
                "" if relative_regret is None or len(relative_regret) == 0 else float(relative_regret[-1])
            )
            final_rows.append(
                {
                    "seed": seed,
                    "problem": cfg.get("paper_problem_label", cfg.get("benchmark", "")),
                    "benchmark": cfg.get("benchmark", ""),
                    "point_model": cfg.get("paper_point_model", cfg.get("model_type", "")),
                    "actor_family": cfg.get("paper_actor_family", ""),
                    "feedback": cfg.get("paper_feedback", "scalar_bandit"),
                    "degree": cfg.get("paper_degree", cfg.get("deg", cfg.get("pricing_context_degree", ""))),
                    "loss": cfg.get("paper_loss", ""),
                    "method": method,
                    "algo_internal": algo_internal,
                    "final_cost": float(values[-1]),
                    "cum_regret": final_regret,
                    "relative_regret": final_relative_regret,
                }
            )

    raw_fields = [
        "seed",
        "problem",
        "benchmark",
        "point_model",
        "actor_family",
        "feedback",
        "degree",
        "round",
        "method",
        "algo_internal",
        "cum_cost",
        "cum_regret",
        "relative_regret",
        "alpha_t",
        "score_grad_norm",
        "plugin_grad_norm",
        "combined_grad_norm",
        "nuisance_loss",
        "scalar_loss",
        "scalar_feedback",
        "adaptive_residual_ema",
        "adaptive_scale_ema",
        "adaptive_residual_ratio_ref",
    ]
    final_fields = [
        "seed",
        "problem",
        "benchmark",
        "point_model",
        "actor_family",
        "feedback",
        "degree",
        "loss",
        "method",
        "algo_internal",
        "final_cost",
        "cum_regret",
        "relative_regret",
    ]
    _write_csv(result_dir / "raw" / "traces.csv", trace_rows, raw_fields)
    _write_csv(result_dir / "raw" / "per_seed_final.csv", final_rows, final_fields)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in final_rows:
        key = (
            row["problem"],
            row["point_model"],
            row["actor_family"],
            row["feedback"],
            row["degree"],
            row["loss"],
            row["method"],
            row["algo_internal"],
        )
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for key, rows in grouped.items():
        regrets = [float(row["cum_regret"]) for row in rows if row["cum_regret"] != ""]
        relative_regrets = [float(row["relative_regret"]) for row in rows if row["relative_regret"] != ""]
        costs = [float(row["final_cost"]) for row in rows]
        regret_mean, regret_sem = _mean_sem(regrets)
        relative_regret_mean, relative_regret_sem = _mean_sem(relative_regrets)
        cost_mean, cost_sem = _mean_sem(costs)
        summary_rows.append(
            {
                "problem": key[0],
                "point_model": key[1],
                "actor_family": key[2],
                "feedback": key[3],
                "degree": key[4],
                "loss": key[5],
                "method": key[6],
                "algo_internal": key[7],
                "n": len(rows),
                "mean_cum_regret": regret_mean,
                "sem_cum_regret": regret_sem,
                "mean_relative_regret": relative_regret_mean,
                "sem_relative_regret": relative_regret_sem,
                "mean_final_cost": cost_mean,
                "sem_final_cost": cost_sem,
            }
        )
    summary_rows.sort(
        key=lambda row: (
            str(row["problem"]),
            str(row["point_model"]),
            _sort_value(row["degree"]),
            str(row["loss"]),
            str(row["method"]),
        )
    )
    summary_fields = [
        "problem",
        "point_model",
        "actor_family",
        "feedback",
        "degree",
        "loss",
        "method",
        "algo_internal",
        "n",
        "mean_cum_regret",
        "sem_cum_regret",
        "mean_relative_regret",
        "sem_relative_regret",
        "mean_final_cost",
        "sem_final_cost",
    ]
    _write_csv(result_dir / "summary" / "summary.csv", summary_rows, summary_fields)
    (result_dir / "summary").mkdir(parents=True, exist_ok=True)
    (result_dir / "summary" / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    _write_latex_table(result_dir / "tables" / f"{block_id}_summary.tex", summary_rows)
    return {"trace_rows": len(trace_rows), "final_rows": len(final_rows), "summary_rows": len(summary_rows)}


def _sort_value(value: Any) -> tuple[int, Any]:
    if value in {None, ""}:
        return (1, "")
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (0, str(value))


def _has_varying_nonempty(rows: list[dict[str, Any]], key: str) -> bool:
    values = {str(row.get(key, "")) for row in rows if str(row.get(key, "")) != ""}
    return len(values) > 1


def _latex_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _write_latex_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    include_degree = _has_varying_nonempty(rows, "degree")
    include_feedback = _has_varying_nonempty(rows, "feedback")
    include_loss = _has_varying_nonempty(rows, "loss")
    columns = ["Problem", "Model"]
    if include_degree:
        columns.append("Degree")
    if include_feedback:
        columns.append("Feedback")
    if include_loss:
        columns.append("Loss")
    columns.extend(["Method", "Regret", "N"])
    align = "l" * (len(columns) - 1) + "r"
    lines = [f"\\begin{{tabular}}{{{align}}}", " & ".join(columns) + " \\\\", "\\hline"]
    table_rows = rows if (include_degree or include_loss) else rows[:80]
    for row in table_rows:
        regret = row["mean_cum_regret"]
        sem = row["sem_cum_regret"]
        regret_text = "NA" if not np.isfinite(float(regret)) else f"{float(regret):.4f} $\\pm$ {float(sem):.4f}"
        values = [_latex_cell(row["problem"]), _latex_cell(row["point_model"])]
        if include_degree:
            values.append(_latex_cell(row.get("degree", "")))
        if include_feedback:
            values.append(_latex_cell(row.get("feedback", "")))
        if include_loss:
            values.append(_latex_cell(row.get("loss", "")))
        values.extend([_latex_cell(row["method"]), regret_text, _latex_cell(row["n"])])
        lines.append(" & ".join(values) + " \\\\")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_block_manifest_json(
    *,
    result_dir: Path,
    block_id: str,
    command: list[str],
    manifest_path: Path,
    opts: BuildOptions,
    runtime_sec: float,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "paper_block_id": block_id,
        "command": command,
        "manifest": _repo_rel(manifest_path),
        "git_hash": _git_hash(),
        "runtime_sec": runtime_sec,
        "device": opts.device,
        "quick": opts.quick,
        "num_seeds": 2 if opts.quick else opts.num_seeds,
        "tuned_config_source": "" if opts.tuned_config_path is None else _repo_rel(opts.tuned_config_path),
        "require_tuned_configs": opts.require_tuned_configs,
    }
    (result_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def block_arg_parser(block_id: str) -> argparse.ArgumentParser:
    spec = BLOCK_SPECS[block_id]
    parser = argparse.ArgumentParser(description=f"Run paper block {block_id}: {spec.title}.")
    parser.add_argument("--out", default=f"paper_runs/manual/results/{block_id}")
    parser.add_argument("--num-seeds", type=int, default=spec.default_num_seeds)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-jobs", default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Skip completed canonical execution outputs. This is the default for paper blocks.",
    )
    parser.add_argument(
        "--force-rerun",
        dest="resume",
        action="store_false",
        help="Run even when the canonical output already has the expected metrics.",
    )
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--tuned-configs", type=Path, default=None)
    parser.add_argument(
        "--allow-missing-tuned-configs",
        action="store_true",
        help="Use tuned configs where available but allow untuned problem/model/degree settings.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser


def main_block_cli(block_id: str) -> None:
    parser = block_arg_parser(block_id)
    args = parser.parse_args()
    del args.n_jobs
    result_dir = result_dir_from_out(args.out, block_id)
    campaign_root = infer_campaign_root_from_out(result_dir)
    paths = campaign_paths(campaign_root)
    if not args.dry_run:
        for key in ("manifests", "outputs", "results", "logs"):
            paths[key].mkdir(parents=True, exist_ok=True)
        ensure_result_block_dirs(campaign_root, (block_id,))
    opts = BuildOptions(
        campaign_root=campaign_root,
        quick=bool(args.quick),
        num_seeds=int(args.num_seeds),
        device=str(args.device),
        eval_every=args.eval_every,
        tuned_configs=load_tuned_configs(args.tuned_configs),
        tuned_config_path=args.tuned_configs,
        require_tuned_configs=args.tuned_configs is not None and not args.allow_missing_tuned_configs,
    )
    manifest_path = paths["manifests"] / f"block_{block_id}.yaml"
    start = time.time()
    if not args.aggregate_only:
        manifest = build_block_manifest(block_id, opts, paths["outputs"])
        if args.dry_run:
            print(f"[DRY RUN] {block_id}: {len(manifest['experiments'])} experiment groups")
            return
        write_yaml(manifest_path, manifest)
        completed, skipped, failed = execute_manifest(manifest, resume=bool(args.resume), dry_run=False)
        print(f"{block_id}: completed={completed} skipped={skipped} failed={failed}")
        if failed:
            raise SystemExit(1)
    aggregate = aggregate_block_results(block_id, result_dir=result_dir, output_root=paths["outputs"])
    write_block_manifest_json(
        result_dir=result_dir,
        block_id=block_id,
        command=sys.argv,
        manifest_path=manifest_path,
        opts=opts,
        runtime_sec=time.time() - start,
    )
    print(f"{block_id}: aggregate={aggregate}")


def prepare_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare isolated paper experiment manifests and Slurm runner.")
    parser.add_argument("--artifact-root", default="paper_runs")
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--block", action="append", choices=tuple(BLOCK_SPECS), default=[])
    parser.add_argument("--num-seeds", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-jobs", default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--tuned-configs", type=Path, default=None)
    parser.add_argument(
        "--allow-missing-tuned-configs",
        action="store_true",
        help="Use tuned configs where available but allow untuned problem/model/degree settings.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-array-jobs", type=int, default=50)
    parser.add_argument("--partition", default="gpu-rtx6000")
    parser.add_argument("--qos", default="embers")
    parser.add_argument("--gpu-type", default="rtx_6000")
    parser.add_argument(
        "--account",
        default=os.environ.get("SLURM_ACCOUNT", "your-slurm-account"),
        help="Slurm account. Defaults to $SLURM_ACCOUNT if set.",
    )
    parser.add_argument("--time-limit", default="08:00:00")
    parser.add_argument("--job-name", default="paperexp")
    return parser


def main_prepare_cli() -> None:
    parser = prepare_arg_parser()
    args = parser.parse_args()
    del args.n_jobs
    campaign = args.campaign or timestamped_campaign_name()
    campaign_root = Path(args.artifact_root) / campaign
    block_ids = list(args.block or DEFAULT_BLOCK_IDS)
    manifests, summary = write_campaign_manifests(
        campaign_root=campaign_root,
        quick=bool(args.quick),
        num_seeds=args.num_seeds,
        device=str(args.device),
        eval_every=args.eval_every,
        tuned_config_path=args.tuned_configs,
        allow_missing_tuned_configs=bool(args.allow_missing_tuned_configs),
        block_ids=block_ids,
        dry_run=bool(args.dry_run),
    )
    compare_path = campaign_paths(campaign_root)["manifests"] / "compare_all.yaml"
    print(f"Campaign: {_repo_rel(campaign_root)}")
    print(f"Experiment groups: {summary['total_experiment_groups']}")
    for block_id, count in summary["blocks"].items():
        print(f"  {block_id}: {count}")
    if args.dry_run:
        experiments = []
        for block_id in block_ids:
            opts = BuildOptions(
                campaign_root=campaign_root,
                quick=bool(args.quick),
                num_seeds=int(args.num_seeds or BLOCK_SPECS[block_id].default_num_seeds),
                device=str(args.device),
                eval_every=args.eval_every,
                tuned_configs=load_tuned_configs(args.tuned_configs),
                tuned_config_path=args.tuned_configs,
                require_tuned_configs=args.tuned_configs is not None and not args.allow_missing_tuned_configs,
            )
            experiments.extend(build_block_manifest(block_id, opts, campaign_paths(campaign_root)["outputs"])["experiments"])
        execution_experiments, deduped_experiments = dedupe_experiments_for_execution(experiments)
        remaining_experiments, skipped_completed = filter_completed_experiments(
            execution_experiments, resume=bool(args.resume)
        )
        execution_experiments_per_task = (
            0
            if not remaining_experiments
            else max(1, math.ceil(len(remaining_experiments) / int(args.max_array_jobs)))
        )
        array_jobs = (
            0
            if not remaining_experiments
            else math.ceil(len(remaining_experiments) / execution_experiments_per_task)
        )
        print(f"[DRY RUN] Would write {len(manifests)} block manifests and {array_jobs} Slurm array jobs")
        print(f"[DRY RUN] Execution groups after de-duplication: {len(execution_experiments)}")
        print(f"[DRY RUN] Reused manifest groups: {deduped_experiments}")
        print(f"[DRY RUN] Skipped completed groups: {skipped_completed}")
        print(f"[DRY RUN] Remaining execution groups: {len(remaining_experiments)}")
        return
    meta = write_slurm_runner(
        campaign_root=campaign_root,
        compare_path=compare_path,
        max_array_jobs=int(args.max_array_jobs),
        partition=str(args.partition),
        qos=str(args.qos),
        gpu_type=str(args.gpu_type),
        account=str(args.account),
        time_limit=str(args.time_limit),
        job_name=str(args.job_name),
        resume=bool(args.resume),
    )
    print(f"Wrote compare manifest: {_repo_rel(compare_path)}")
    if meta["array_jobs"] == 0:
        print("No Slurm runner needed: all unique execution groups are already complete.")
    else:
        print(f"Wrote Slurm runner: {meta['submit']}")
        print(
            f"Array jobs: {meta['array_jobs']} "
            f"({meta['experiment_groups']} groups, {meta['experiments_per_task']} per task)"
        )
