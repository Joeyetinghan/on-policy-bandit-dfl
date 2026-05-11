"""Generate DFHPG tuning manifests for point-model sweeps."""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lib.io import load_yaml, seed_list, write_yaml

DEFAULT_PROBLEMS = ("topk", "shortest_path", "pricing")
DEFAULT_DEGREES = (2, 4, 6, 8, 10)
THETA_LR_GRID = (3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2)
NUISANCE_LR_RATIOS = (3.0, 10.0)
ACTOR_SCALE_MULTIPLIERS = (0.003, 0.01, 0.03, 0.1)
ALPHA_MAX_GRID = (0.3, 0.5, 0.7)
ALPHA_MIN_GRID = (0.02, 0.05)
ALPHA_DIAG_MAX_GRID = (0.5, 0.7, 0.9)
ALPHA_DIAG_MIN_GRID = (0.05, 0.1, 0.2)
ALPHA_DIAG_WARMUP_FRACS = (0.05, 0.1, 0.2)

DEFAULT_ALPHA_MAX = 0.5
DEFAULT_ALPHA_MIN = 0.05
DEFAULT_PILOT_THETA_LR = 3.0e-3
DEFAULT_PILOT_NUISANCE_RATIO = 10.0
DEFAULT_PILOT_ACTOR_MULTIPLIER = 0.03
ALPHA_DIAG_THETA_LR = 1.0e-2
ALPHA_DIAG_NUISANCE_RATIO = 10.0
ALPHA_DIAG_ACTOR_MULTIPLIER = 0.01
ADAPTIVE_ALPHA_FIXED = {
    "hybrid_alpha_schedule": "adaptive_nuisance_reliability",
    "hybrid_alpha_warmup_frac": 0.05,
    "hybrid_alpha_ema_decay": 0.98,
    "hybrid_alpha_smooth": 0.05,
}
POINT_MODEL_TYPES = {"linear", "nn", "shared_linear", "shared_nn"}
GENERATIVE_MODEL_TYPES = {"cnf", "diffusion", "shared_cnf", "shared_diffusion"}
MODEL_FAMILIES = ("point", "generative", "all")


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    base: str
    model_types: tuple[str, ...]
    hard_overrides: dict
    hybrid_loss_type: str = "spo_plus"


PROBLEM_SPECS: dict[str, ProblemSpec] = {
    "topk": ProblemSpec(
        name="topk",
        base="configs/topk.yaml",
        model_types=("linear", "nn", "cnf", "diffusion"),
        hard_overrides={"T": 1000, "eps_bar": 0.5, "feedback_mode": "bandit"},
    ),
    "shortest_path": ProblemSpec(
        name="shortest_path",
        base="configs/shortest_path.yaml",
        model_types=("linear", "nn", "cnf", "diffusion"),
        hard_overrides={
            "T": 1000,
            "p": 10,
            "grid_size": 5,
            "eps_bar": 0.5,
            "data_generator_family": "shortest_path_calibrated_poly",
            "poly_offset": 1.0,
            "poly_scale": 1.0,
        },
    ),
    "pricing": ProblemSpec(
        name="pricing",
        base="configs/pricing.yaml",
        model_types=("shared_linear", "shared_nn", "shared_cnf", "shared_diffusion"),
        hard_overrides={
            "T": 2000,
            "pricing_common_shock_sigma": 0.1,
        },
    ),
}

SIMPLE_GENERATIVE_OVERRIDES = {
    "nn_hidden_dim": 128,
    "flow_num_coupling_layers": 2,
    "flow_hidden_dim": 128,
    "diffusion_num_steps": 20,
    "diffusion_time_embed_dim": 8,
    "diffusion_hidden_dim": 128,
    "diffusion_inference_steps": 8,
    "diffusion_ddim_eta": 0.0,
    "generative_surrogate_num_iter": 16,
    "generative_weight_decay": 0.0,
    "generative_grad_clip_norm": 10.0,
}


def _load_yaml(path: Path) -> dict:
    return load_yaml(path) or {}


def _base_config(spec: ProblemSpec) -> dict:
    return _load_yaml(REPO_ROOT / spec.base)


def _problem_config(spec: ProblemSpec, model_type: str) -> dict:
    cfg = _base_config(spec)
    cfg.update(spec.hard_overrides)
    cfg["benchmark"] = spec.name
    cfg["model_type"] = model_type
    return cfg


def _setting_overrides(spec: ProblemSpec, degree: int) -> tuple[str, dict]:
    degree = int(degree)
    if degree < 1:
        raise ValueError("degrees must be positive")
    setting_id = f"deg{degree}"
    overrides = {"hybrid_tuning_setting_id": setting_id, "hybrid_tuning_degree": degree}
    if spec.name in {"topk", "shortest_path"}:
        overrides["deg"] = degree
    elif spec.name == "pricing":
        overrides["pricing_context_degree"] = degree
        overrides["pricing_price_degree"] = degree
    else:
        raise ValueError(f"Unsupported tuning problem: {spec.name}")
    return setting_id, overrides


def _score_scale_from_matrix(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        return 1.0
    coord_std = np.std(values, axis=0)
    nonzero_std = coord_std[np.isfinite(coord_std) & (coord_std > 1.0e-12)]
    if nonzero_std.size:
        return float(max(np.median(nonzero_std), 1.0e-8))
    centered = values - np.mean(values, axis=0, keepdims=True)
    mad = np.median(np.abs(centered))
    if np.isfinite(mad) and mad > 1.0e-12:
        return float(max(mad, 1.0e-8))
    magnitude = np.median(np.abs(values))
    if np.isfinite(magnitude) and magnitude > 1.0e-12:
        return float(max(magnitude, 1.0e-8))
    return 1.0


def estimate_score_scale(config: dict, *, sample_count: int = 128, seed: int = 314159) -> float:
    """Estimate a robust score-unit scale from independent calibration samples."""

    from src.experiment.registry import create_env_and_oracle

    env, _ = create_env_and_oracle(config, seed=seed)
    scores: list[np.ndarray] = []
    for _ in range(max(1, int(sample_count))):
        sample = env.sample_eval_instance()
        scores.append(np.asarray(sample["score_vector"], dtype=float).reshape(-1))
    return _score_scale_from_matrix(np.vstack(scores))


def safe_estimate_score_scale(config: dict, *, sample_count: int, seed: int) -> float:
    try:
        return estimate_score_scale(config, sample_count=sample_count, seed=seed)
    except Exception as exc:
        fallback = float(config.get("policy_sampling_scale", 1.0))
        print(
            f"[warn] score-scale calibration failed for {config.get('benchmark')} "
            f"model={config.get('model_type')}: {exc}; using fallback scale {fallback:g}",
            file=sys.stderr,
        )
        return max(fallback, 1.0e-8)


def float_slug(value: float) -> str:
    return f"{float(value):.0e}".replace("+", "").replace("-", "m").replace(".", "p")


def alpha_slug(alpha_max: float, alpha_min: float) -> str:
    return f"amax{float_slug(alpha_max)}_amin{float_slug(alpha_min)}"


def warmup_slug(warmup_frac: float) -> str:
    return f"wfrac{float_slug(warmup_frac)}"


def lr_actor_candidate_id(
    *,
    theta_lr: float,
    nuisance_ratio: float,
    actor_multiplier: float | None,
    alpha_max: float,
    alpha_min: float,
    alpha_warmup_frac: float | None = None,
    generative_lambda_score: float | None = None,
) -> str:
    parts = [
        f"tlr{float_slug(theta_lr)}",
        f"nrat{float_slug(nuisance_ratio)}",
    ]
    if actor_multiplier is not None:
        parts.append(f"amul{float_slug(actor_multiplier)}")
    if generative_lambda_score is not None:
        parts.append(f"glam{float_slug(generative_lambda_score)}")
    else:
        parts.append(alpha_slug(alpha_max, alpha_min))
    if alpha_warmup_frac is not None and generative_lambda_score is None:
        parts.append(warmup_slug(alpha_warmup_frac))
    return "_".join(parts)


def _algo_off_overrides() -> dict:
    return {
        "run_true_model": False,
        "run_random_oracle": True,
        "run_greedy_contextual_bandit": False,
        "run_epsilon_greedy_contextual_bandit": False,
        "run_thompson_contextual_bandit": False,
        "run_hybrid_bandit": True,
    }


def _common_tuning_overrides(spec: ProblemSpec) -> dict:
    return {
        **_algo_off_overrides(),
        **SIMPLE_GENERATIVE_OVERRIDES,
        "model_update_batch_rounds": 1,
        "nuisance_update_batch_rounds": 1,
        "theta_lr_schedule": "shifted_inverse_time",
        "nuisance_lr_schedule": "shifted_inverse_time",
        "theta_lr_offset": 100,
        "nuisance_lr_offset": 100,
        "hybrid_loss_type": spec.hybrid_loss_type,
        "hybrid_mse_weight": 0.0,
        "policy_baseline_type": "nuisance_induced",
        "perturbation_distribution": "normal",
        "exploration_epsilon": 0.0,
        **ADAPTIVE_ALPHA_FIXED,
    }


def _candidate_overrides(
    spec: ProblemSpec,
    *,
    theta_lr: float,
    nuisance_ratio: float,
    actor_multiplier: float | None,
    score_scale: float,
    alpha_max: float,
    alpha_min: float,
    alpha_warmup_frac: float,
    stage: str,
    actor_scale_tuned: bool = True,
    generative_lambda_score: float | None = None,
) -> tuple[str, dict]:
    nuisance_lr = float(theta_lr) * float(nuisance_ratio)
    scale_multiplier = None if actor_multiplier is None else float(actor_multiplier)
    actor_sampling_scale = None if scale_multiplier is None else float(score_scale) * float(scale_multiplier)
    candidate_id = lr_actor_candidate_id(
        theta_lr=theta_lr,
        nuisance_ratio=nuisance_ratio,
        actor_multiplier=actor_multiplier,
        alpha_max=alpha_max,
        alpha_min=alpha_min,
        alpha_warmup_frac=alpha_warmup_frac if stage == "alpha_diag" else None,
        generative_lambda_score=generative_lambda_score,
    )
    if generative_lambda_score is None:
        mixing_label = f"alpha=[{alpha_min:g},{alpha_max:g}] warmup_frac={alpha_warmup_frac:g}"
        mixing_overrides = {
            "hybrid_tuning_mixing_weight_type": "adaptive_hybrid_alpha",
        }
    else:
        mixing_label = f"generative_lambda_score={generative_lambda_score:g}"
        mixing_overrides = {
            "generative_lambda_score": float(generative_lambda_score),
            "hybrid_tuning_generative_lambda_score": float(generative_lambda_score),
            "hybrid_tuning_mixing_weight_type": "generative_lambda_score",
        }
    overrides = {
        "theta_lr": float(theta_lr),
        "nuisance_lr": nuisance_lr,
        "hybrid_alpha_max": float(alpha_max),
        "hybrid_alpha_min": float(alpha_min),
        "hybrid_alpha_warmup_frac": float(alpha_warmup_frac),
        "hybrid_tuning_stage": stage,
        "hybrid_tuning_problem": spec.name,
        "hybrid_tuning_candidate_id": candidate_id,
        "hybrid_tuning_candidate_label": f"tlr={theta_lr:g} nlr={nuisance_lr:g} ratio={nuisance_ratio:g} {mixing_label}",
        "hybrid_tuning_nuisance_lr_ratio": float(nuisance_ratio),
        **mixing_overrides,
    }
    if actor_sampling_scale is not None and scale_multiplier is not None:
        overrides.update(
            {
                "policy_sampling_scale": actor_sampling_scale,
                "hybrid_tuning_score_scale": float(score_scale),
                "hybrid_tuning_actor_scale_multiplier": float(scale_multiplier),
                "hybrid_tuning_actor_scale_tuned": bool(actor_scale_tuned),
                "hybrid_tuning_candidate_label": (
                    f"tlr={theta_lr:g} nlr={nuisance_lr:g} "
                    f"ratio={nuisance_ratio:g} actor={actor_sampling_scale:g} "
                    f"(scale={score_scale:g} x {scale_multiplier:g}"
                    f"{' tuned' if actor_scale_tuned else ' fixed'}) "
                    f"{mixing_label}"
                ),
            }
        )
    return candidate_id, overrides


def _stage_candidate_grid(
    stage: str,
    *,
    alpha_max: float,
    alpha_min: float,
    pilot_theta_lr: float,
    pilot_nuisance_ratio: float,
    pilot_actor_multiplier: float,
) -> Iterable[tuple[float, float, float, float, float, float]]:
    fixed_warmup_frac = float(ADAPTIVE_ALPHA_FIXED["hybrid_alpha_warmup_frac"])
    if stage == "alpha_pilot":
        for amax in ALPHA_MAX_GRID:
            for amin in ALPHA_MIN_GRID:
                yield (pilot_theta_lr, pilot_nuisance_ratio, pilot_actor_multiplier, amax, amin, fixed_warmup_frac)
        return
    if stage == "alpha_diag":
        for amax in ALPHA_DIAG_MAX_GRID:
            for amin in ALPHA_DIAG_MIN_GRID:
                for warmup_frac in ALPHA_DIAG_WARMUP_FRACS:
                    yield (
                        pilot_theta_lr,
                        pilot_nuisance_ratio,
                        pilot_actor_multiplier,
                        amax,
                        amin,
                        warmup_frac,
                    )
        return
    if stage == "problem_grid":
        for theta_lr in THETA_LR_GRID:
            for ratio in NUISANCE_LR_RATIOS:
                for actor_multiplier in ACTOR_SCALE_MULTIPLIERS:
                    yield (theta_lr, ratio, actor_multiplier, alpha_max, alpha_min, fixed_warmup_frac)
        return
    if stage == "full_grid":
        for theta_lr in THETA_LR_GRID:
            for ratio in NUISANCE_LR_RATIOS:
                for actor_multiplier in ACTOR_SCALE_MULTIPLIERS:
                    for amax in ALPHA_MAX_GRID:
                        for amin in ALPHA_MIN_GRID:
                            yield (theta_lr, ratio, actor_multiplier, amax, amin, fixed_warmup_frac)
        return
    raise ValueError(f"Unknown stage: {stage}")


def _stage_candidate_grid_for_model(
    stage: str,
    model_type: str,
    *,
    alpha_max: float,
    alpha_min: float,
    pilot_theta_lr: float,
    pilot_nuisance_ratio: float,
    pilot_actor_multiplier: float,
) -> Iterable[dict]:
    fixed_warmup_frac = float(ADAPTIVE_ALPHA_FIXED["hybrid_alpha_warmup_frac"])
    if model_type in GENERATIVE_MODEL_TYPES:
        if stage in {"alpha_pilot", "alpha_diag"}:
            for _, _, _, amax, amin, warmup_frac in _stage_candidate_grid(
                stage,
                alpha_max=alpha_max,
                alpha_min=alpha_min,
                pilot_theta_lr=pilot_theta_lr,
                pilot_nuisance_ratio=pilot_nuisance_ratio,
                pilot_actor_multiplier=pilot_actor_multiplier,
            ):
                yield {
                    "theta_lr": pilot_theta_lr,
                    "nuisance_ratio": pilot_nuisance_ratio,
                    "actor_multiplier": None,
                    "alpha_max": amax,
                    "alpha_min": amin,
                    "alpha_warmup_frac": warmup_frac,
                    "actor_scale_tuned": False,
                    "generative_lambda_score": None,
                }
            return

        if stage == "full_grid":
            alpha_grid = ((amax, amin) for amax in ALPHA_MAX_GRID for amin in ALPHA_MIN_GRID)
        elif stage == "problem_grid":
            alpha_grid = ((alpha_max, alpha_min),)
        else:
            raise ValueError(f"Unknown stage: {stage}")

        for theta_lr in THETA_LR_GRID:
            for ratio in NUISANCE_LR_RATIOS:
                for amax, amin in alpha_grid:
                    yield {
                        "theta_lr": theta_lr,
                        "nuisance_ratio": ratio,
                        "actor_multiplier": None,
                        "alpha_max": amax,
                        "alpha_min": amin,
                        "alpha_warmup_frac": fixed_warmup_frac,
                        "actor_scale_tuned": False,
                        "generative_lambda_score": None,
                    }
        return

    if model_type not in POINT_MODEL_TYPES:
        raise ValueError(f"Unknown tuning model_type: {model_type}")

    if stage in {"alpha_pilot", "alpha_diag"}:
        for theta_lr, ratio, actor_multiplier, amax, amin, warmup_frac in _stage_candidate_grid(
            stage,
            alpha_max=alpha_max,
            alpha_min=alpha_min,
            pilot_theta_lr=pilot_theta_lr,
            pilot_nuisance_ratio=pilot_nuisance_ratio,
            pilot_actor_multiplier=pilot_actor_multiplier,
        ):
            yield {
                "theta_lr": theta_lr,
                "nuisance_ratio": ratio,
                "actor_multiplier": actor_multiplier,
                "alpha_max": amax,
                "alpha_min": amin,
                "alpha_warmup_frac": warmup_frac,
                "actor_scale_tuned": True,
                "generative_lambda_score": None,
            }
        return

    if stage == "full_grid":
        raw_grid = _stage_candidate_grid(
            stage,
            alpha_max=alpha_max,
            alpha_min=alpha_min,
            pilot_theta_lr=pilot_theta_lr,
            pilot_nuisance_ratio=pilot_nuisance_ratio,
            pilot_actor_multiplier=pilot_actor_multiplier,
        )
    elif stage == "problem_grid":
        raw_values = []
        raw_values.extend(
            _stage_candidate_grid(
                "problem_grid",
                alpha_max=alpha_max,
                alpha_min=alpha_min,
                pilot_theta_lr=pilot_theta_lr,
                pilot_nuisance_ratio=pilot_nuisance_ratio,
                pilot_actor_multiplier=pilot_actor_multiplier,
            )
        )
        raw_values.extend(
            _stage_candidate_grid(
                "alpha_pilot",
                alpha_max=alpha_max,
                alpha_min=alpha_min,
                pilot_theta_lr=pilot_theta_lr,
                pilot_nuisance_ratio=pilot_nuisance_ratio,
                pilot_actor_multiplier=pilot_actor_multiplier,
            )
        )
        raw_grid = raw_values
    else:
        raise ValueError(f"Unknown stage: {stage}")

    seen = set()
    for theta_lr, ratio, actor_multiplier, amax, amin, warmup_frac in raw_grid:
        key = (theta_lr, ratio, actor_multiplier, amax, amin, warmup_frac)
        if key in seen:
            continue
        seen.add(key)
        yield {
            "theta_lr": theta_lr,
            "nuisance_ratio": ratio,
            "actor_multiplier": actor_multiplier,
            "alpha_max": amax,
            "alpha_min": amin,
            "alpha_warmup_frac": warmup_frac,
            "actor_scale_tuned": True,
            "generative_lambda_score": None,
        }


def _default_pilot_values(stage: str) -> tuple[float, float, float]:
    if stage == "alpha_diag":
        return ALPHA_DIAG_THETA_LR, ALPHA_DIAG_NUISANCE_RATIO, ALPHA_DIAG_ACTOR_MULTIPLIER
    return DEFAULT_PILOT_THETA_LR, DEFAULT_PILOT_NUISANCE_RATIO, DEFAULT_PILOT_ACTOR_MULTIPLIER


def build_manifest(
    *,
    stage: str,
    problems: Iterable[str],
    seeds: list[int],
    degrees: Iterable[int] = DEFAULT_DEGREES,
    model_family: str = "point",
    output_root: str | None = None,
    campaign: str | None = None,
    alpha_max: float = DEFAULT_ALPHA_MAX,
    alpha_min: float = DEFAULT_ALPHA_MIN,
    pilot_theta_lr: float | None = None,
    pilot_nuisance_ratio: float | None = None,
    pilot_actor_multiplier: float | None = None,
    score_scale_sample_count: int = 128,
    score_scale_seed: int = 314159,
) -> dict:
    if model_family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model_family={model_family!r}; expected one of {MODEL_FAMILIES}")

    default_theta_lr, default_ratio, default_actor_multiplier = _default_pilot_values(stage)
    if pilot_theta_lr is None:
        pilot_theta_lr = default_theta_lr
    if pilot_nuisance_ratio is None:
        pilot_nuisance_ratio = default_ratio
    if pilot_actor_multiplier is None:
        pilot_actor_multiplier = default_actor_multiplier

    experiments: list[dict] = []
    degrees = tuple(int(degree) for degree in degrees)
    for problem in problems:
        spec = PROBLEM_SPECS[problem]
        for degree in degrees:
            setting_id, setting_overrides = _setting_overrides(spec, degree)
            scale_cfg = _problem_config(spec, spec.model_types[0])
            scale_cfg.update(setting_overrides)
            score_scale = safe_estimate_score_scale(
                scale_cfg,
                sample_count=score_scale_sample_count,
                seed=score_scale_seed,
            )
            if model_family == "point":
                model_types = tuple(model_type for model_type in spec.model_types if model_type in POINT_MODEL_TYPES)
            elif model_family == "generative":
                model_types = tuple(model_type for model_type in spec.model_types if model_type in GENERATIVE_MODEL_TYPES)
            else:
                model_types = spec.model_types

            for model_type in model_types:
                common_overrides = {
                    **_common_tuning_overrides(spec),
                    "benchmark": spec.name,
                    "model_type": model_type,
                    **spec.hard_overrides,
                    **setting_overrides,
                }
                for candidate in _stage_candidate_grid_for_model(
                    stage,
                    model_type,
                    alpha_max=alpha_max,
                    alpha_min=alpha_min,
                    pilot_theta_lr=pilot_theta_lr,
                    pilot_nuisance_ratio=pilot_nuisance_ratio,
                    pilot_actor_multiplier=pilot_actor_multiplier,
                ):
                    candidate_id, candidate_overrides = _candidate_overrides(
                        spec,
                        theta_lr=candidate["theta_lr"],
                        nuisance_ratio=candidate["nuisance_ratio"],
                        actor_multiplier=candidate["actor_multiplier"],
                        score_scale=score_scale,
                        alpha_max=candidate["alpha_max"],
                        alpha_min=candidate["alpha_min"],
                        alpha_warmup_frac=candidate["alpha_warmup_frac"],
                        stage=stage,
                        actor_scale_tuned=candidate["actor_scale_tuned"],
                        generative_lambda_score=candidate["generative_lambda_score"],
                    )
                    overrides = copy.deepcopy(common_overrides)
                    overrides.update(candidate_overrides)
                    if output_root is not None:
                        overrides["output_root"] = str(output_root)
                    if campaign is not None:
                        overrides["hybrid_tuning_campaign"] = str(campaign)
                    experiments.append(
                        {
                            "name": (
                                f"problem_hybrid_tuning_{stage}_{spec.name}_"
                                f"{setting_id}_{model_type}_{candidate_id}"
                            ),
                            "base": spec.base,
                            "overrides": overrides,
                            "seeds": seeds,
                        }
                    )
    return {"experiments": experiments}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("alpha_pilot", "alpha_diag", "problem_grid", "full_grid"),
        default="problem_grid",
    )
    parser.add_argument(
        "--model-family",
        choices=MODEL_FAMILIES,
        default="point",
        help="Model family to tune. Defaults to Gaussian point models.",
    )
    parser.add_argument("--problem", action="append", choices=tuple(PROBLEM_SPECS), default=[])
    parser.add_argument(
        "--degree",
        action="append",
        type=int,
        default=[],
        help="Problem degree to tune. May be repeated. Defaults to 2, 4, 6, 8, and 10.",
    )
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument(
        "--seed-start-index",
        type=int,
        default=30,
        help="Start index into the fixed seed pool. Default keeps tuning seeds disjoint from main 30-seed runs.",
    )
    parser.add_argument("--alpha-max", type=float, default=DEFAULT_ALPHA_MAX)
    parser.add_argument("--alpha-min", type=float, default=DEFAULT_ALPHA_MIN)
    parser.add_argument("--pilot-theta-lr", type=float, default=None)
    parser.add_argument("--pilot-nuisance-ratio", type=float, default=None)
    parser.add_argument("--pilot-actor-multiplier", type=float, default=None)
    parser.add_argument("--score-scale-samples", type=int, default=128)
    parser.add_argument("--score-scale-seed", type=int, default=314159)
    parser.add_argument(
        "--artifact-root",
        default="tuning_runs",
        help="Artifact root for generated tuning manifests and outputs.",
    )
    parser.add_argument(
        "--campaign",
        default=None,
        help="Campaign directory name under --artifact-root. Defaults to a timestamped tuning campaign.",
    )
    return parser.parse_args()


def selected_problems(args: argparse.Namespace) -> tuple[str, ...]:
    if args.stage == "alpha_diag" and not args.problem:
        return ("shortest_path",)
    return tuple(args.problem) if args.problem else DEFAULT_PROBLEMS


def main() -> None:
    args = _parse_args()
    problems = selected_problems(args)
    degrees = tuple(args.degree) if args.degree else DEFAULT_DEGREES
    seeds = seed_list(args.num_seeds, start_index=args.seed_start_index)

    campaign = args.campaign or "hybrid_tuning_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign_root = Path(args.artifact_root) / campaign
    output_root = campaign_root / "outputs"

    manifest = build_manifest(
        stage=args.stage,
        problems=problems,
        degrees=degrees,
        seeds=seeds,
        model_family=str(args.model_family),
        output_root=str(output_root) if output_root is not None else None,
        campaign=campaign,
        alpha_max=args.alpha_max,
        alpha_min=args.alpha_min,
        pilot_theta_lr=args.pilot_theta_lr,
        pilot_nuisance_ratio=args.pilot_nuisance_ratio,
        pilot_actor_multiplier=args.pilot_actor_multiplier,
        score_scale_sample_count=args.score_scale_samples,
        score_scale_seed=args.score_scale_seed,
    )

    problem_slug = "main" if tuple(problems) == DEFAULT_PROBLEMS else "_".join(problems)
    degree_slug = "deg" + "_".join(str(degree) for degree in degrees)
    sweep_name = f"problem_hybrid_tuning_{args.stage}_{problem_slug}_{degree_slug}_{args.num_seeds}seed.yaml"
    compare_name = f"compare_problem_hybrid_tuning_{args.stage}_{problem_slug}_{degree_slug}_{args.num_seeds}seed.yaml"
    manifest_dir = campaign_root / "manifests"
    write_yaml(manifest_dir / sweep_name, manifest)
    write_yaml(manifest_dir / compare_name, {"include": [sweep_name]})
    print(f"Generated {len(manifest['experiments'])} experiments in {manifest_dir / sweep_name}")
    print(f"Outputs will route to {output_root}")


if __name__ == "__main__":
    main()
