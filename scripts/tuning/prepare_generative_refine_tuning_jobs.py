"""Generate reproducible DFHPG tuning manifests for CNF/diffusion sweeps.

This one-shot campaign tunes the three main synthetic problems with a compact
artifact layout under ``tuning_runs/<campaign>/``:

- top-k and shortest-path generative models over degrees 2, 4, 6, 8, and 10;
- pricing generative models over degrees 2, 4, 6, 8, and 10;

The generated runs keep offline opt disabled and keep RandomOracle enabled for
per-run sanity checks. To resume a tuning campaign, pass
``--skip-completed-from`` one or more times to filter out seed-level
configurations that already have both DFHPG and RandomOracle metrics in
previous campaign outputs.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Iterable, Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiment.algorithm_names import dfhpg_metrics_path  # noqa: E402

from scripts.tuning.generate_problem_hybrid_tuning_manifests import (  # noqa: E402
    ADAPTIVE_ALPHA_FIXED,
    DEFAULT_ALPHA_MAX,
    DEFAULT_ALPHA_MIN,
    PROBLEM_SPECS,
    _candidate_overrides,
    _common_tuning_overrides,
    _problem_config,
    _setting_overrides,
    alpha_slug,
    safe_estimate_score_scale,
)
from scripts.lib.io import seed_list, write_yaml  # noqa: E402


DEFAULT_CAMPAIGN_PREFIX = "generative_tuning"

GENERATIVE_DEGREES = (2, 4, 6, 8, 10)
PRICING_DEGREES = (2, 4, 6, 8, 10)

TOPK_SP_GENERATIVE_THETA_LR_GRID = (3.0e-3, 1.0e-2, 2.0e-2, 3.0e-2)
PRICING_GENERATIVE_THETA_LR_GRID = (1.0e-3, 3.0e-3, 1.0e-2, 2.0e-2)
DIFFUSION_SURROGATE_NUM_ITER_GRID = (16,)

NUISANCE_RATIO_GRID = {
    "topk": (1.0, 2.0, 3.0, 5.0, 10.0),
    "shortest_path": (1.0, 2.0, 3.0, 5.0, 10.0),
    "pricing": (1.0, 2.0, 3.0, 5.0, 10.0),
}
FIXED_ALPHA_PAIR = (DEFAULT_ALPHA_MAX, DEFAULT_ALPHA_MIN)
REFINED_ALPHA_PAIR_GRID = (
    (0.3, 0.02),
    (0.5, 0.02),
    (0.5, 0.05),
    (0.7, 0.05),
    (0.7, 0.10),
    (0.9, 0.05),
)
GENERATIVE_ALPHA_ANCHORS = {
    "topk": {"theta_lr": 2.0e-2, "nuisance_ratio": 3.0, "diffusion_surrogate_num_iter": 16},
    "shortest_path": {
        "theta_lr": 3.0e-2,
        "nuisance_ratio": 3.0,
        "diffusion_surrogate_num_iter": 16,
    },
    "pricing": {
        "theta_lr": 1.0e-3,
        "nuisance_ratio": 3.0,
        "diffusion_surrogate_num_iter": 16,
    },
}

GENERATIVE_MODEL_TYPES = {
    "topk": ("cnf", "diffusion"),
    "shortest_path": ("cnf", "diffusion"),
    "pricing": ("shared_cnf", "shared_diffusion"),
}
SCORE_SCALE_MODEL_TYPES = {"topk": "linear", "shortest_path": "linear", "pricing": "shared_linear"}

ALL_GENERATIVE_MODEL_TYPES = {
    model_type
    for model_types in GENERATIVE_MODEL_TYPES.values()
    for model_type in model_types
}


def _parse_float_csv(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _parse_alpha_pairs(raw: str) -> tuple[tuple[float, float], ...]:
    pairs: list[tuple[float, float]] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise argparse.ArgumentTypeError("alpha pairs must use AMAX:AMIN entries")
        raw_max, raw_min = part.split(":", 1)
        alpha_max = float(raw_max.strip())
        alpha_min = float(raw_min.strip())
        if not 0.0 <= alpha_min <= alpha_max <= 1.0:
            raise argparse.ArgumentTypeError("alpha pairs require 0 <= AMIN <= AMAX <= 1")
        pairs.append((alpha_max, alpha_min))
    if not pairs:
        raise argparse.ArgumentTypeError("expected at least one alpha pair")
    return tuple(dict.fromkeys(pairs))


def _parse_top_level_scalar(value: str) -> Any:
    value = value.strip()
    if not value or value in {"null", "None", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


COMPLETION_CONFIG_KEYS = {
    "seed",
    "benchmark",
    "feedback_mode",
    "hybrid_tuning_problem",
    "hybrid_tuning_degree",
    "deg",
    "pricing_context_degree",
    "model_type",
    "theta_lr",
    "nuisance_lr",
    "hybrid_tuning_nuisance_lr_ratio",
    "policy_sampling_scale",
    "hybrid_tuning_actor_scale_multiplier",
    "hybrid_alpha_max",
    "hybrid_alpha_min",
    "hybrid_alpha_warmup_frac",
    # Legacy fields are kept only for completed-run signatures, so old
    # lambda-mixed generative runs are not treated as equivalent to alpha-mixed
    # DFHPG runs.
    "generative_lambda_score",
    "hybrid_tuning_generative_lambda_score",
    "generative_surrogate_num_iter",
    "hybrid_tuning_diffusion_surrogate_num_iter",
}


def _load_completion_config(path: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw_line or raw_line[0].isspace() or ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        if key in COMPLETION_CONFIG_KEYS:
            config[key] = _parse_top_level_scalar(value)
    return config


def _float_key(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return f"{float(value):.12g}"


def _int_key(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _config_signature(config: dict[str, Any], *, seed: int | None) -> tuple[Any, ...] | None:
    problem = config.get("hybrid_tuning_problem") or config.get("benchmark")
    model_type = config.get("model_type")
    degree = config.get("hybrid_tuning_degree")
    if degree is None:
        degree = config.get("pricing_context_degree") if problem == "pricing" else config.get("deg")
    if problem is None or model_type is None or degree is None or seed is None:
        return None

    theta_lr = config.get("theta_lr")
    ratio = config.get("hybrid_tuning_nuisance_lr_ratio")
    if ratio is None and theta_lr not in {None, 0, ""} and config.get("nuisance_lr") not in {None, ""}:
        ratio = float(config["nuisance_lr"]) / float(theta_lr)

    legacy_lambda_score = config.get(
        "hybrid_tuning_generative_lambda_score",
        config.get("generative_lambda_score"),
    )
    model_type_str = str(model_type)
    diffusion_iter = None
    if "diffusion" in model_type_str:
        diffusion_iter = config.get(
            "hybrid_tuning_diffusion_surrogate_num_iter",
            config.get("generative_surrogate_num_iter"),
        )

    return (
        str(problem),
        int(float(degree)),
        model_type_str,
        int(seed),
        str(config.get("feedback_mode", "")),
        _float_key(theta_lr),
        _float_key(ratio),
        _float_key(config.get("policy_sampling_scale")),
        _float_key(config.get("hybrid_tuning_actor_scale_multiplier")),
        _float_key(config.get("hybrid_alpha_max")),
        _float_key(config.get("hybrid_alpha_min")),
        _float_key(config.get("hybrid_alpha_warmup_frac")),
        _float_key(legacy_lambda_score),
        _int_key(diffusion_iter),
    )


def _experiment_signature(experiment: dict) -> tuple[Any, ...] | None:
    seeds = experiment.get("seeds") or []
    if len(seeds) != 1:
        return None
    return _config_signature(experiment["overrides"], seed=int(seeds[0]))


def _outputs_root(root: Path) -> Path:
    return root / "outputs" if (root / "outputs").is_dir() else root


def _completed_signatures(roots: Iterable[Path]) -> set[tuple[Any, ...]]:
    signatures: set[tuple[Any, ...]] = set()
    for raw_root in roots:
        output_root = _outputs_root(Path(raw_root))
        if not output_root.is_dir():
            print(f"[warn] completed root does not exist or is not a directory: {raw_root}", file=sys.stderr)
            continue
        for config_path in sorted(output_root.glob("*/config.yaml")):
            run_dir = config_path.parent
            if not dfhpg_metrics_path(run_dir).exists():
                continue
            if not (run_dir / "RandomOracle" / "metrics.json").exists():
                continue
            config = _load_completion_config(config_path)
            signature = _config_signature(config, seed=_int_key(config.get("seed")))
            if signature is not None:
                signatures.add(signature)
    return signatures


def _filter_completed(experiments: list[dict], roots: Iterable[Path]) -> tuple[list[dict], int]:
    completed = _completed_signatures(roots)
    if not completed:
        return experiments, 0
    remaining: list[dict] = []
    skipped = 0
    for experiment in experiments:
        signature = _experiment_signature(experiment)
        if signature is not None and signature in completed:
            skipped += 1
            continue
        remaining.append(experiment)
    return remaining, skipped


def _score_scale(
    *,
    problem: str,
    degree: int,
    cache: dict[tuple[str, int], float],
    sample_count: int,
    seed: int,
) -> float:
    key = (problem, int(degree))
    if key in cache:
        return cache[key]

    spec = PROBLEM_SPECS[problem]
    model_type = SCORE_SCALE_MODEL_TYPES[problem]
    cfg = _problem_config(spec, model_type)
    _, setting_overrides = _setting_overrides(spec, int(degree))
    cfg.update(setting_overrides)

    if problem == "pricing":
        scale = 1.0
    else:
        scale = safe_estimate_score_scale(cfg, sample_count=sample_count, seed=seed)
    cache[key] = float(scale)
    return float(scale)


def _base_overrides(*, problem: str, model_type: str, degree: int, output_root: Path, campaign: str) -> dict:
    spec = PROBLEM_SPECS[problem]
    _, setting_overrides = _setting_overrides(spec, int(degree))
    return {
        **_common_tuning_overrides(spec),
        "benchmark": spec.name,
        "model_type": model_type,
        **spec.hard_overrides,
        **setting_overrides,
        "output_root": str(output_root),
        "hybrid_tuning_campaign": campaign,
    }


def _candidate(
    *,
    problem: str,
    model_type: str,
    degree: int,
    stage: str,
    theta_lr: float,
    nuisance_ratio: float,
    actor_multiplier: float | None,
    alpha_max: float,
    alpha_min: float,
    diffusion_surrogate_num_iter: int | None,
    output_root: Path,
    campaign: str,
    score_scale_cache: dict[tuple[str, int], float],
    score_scale_samples: int,
    score_scale_seed: int,
) -> dict:
    spec = PROBLEM_SPECS[problem]
    is_generative = str(model_type) in ALL_GENERATIVE_MODEL_TYPES
    candidate_id, candidate_overrides = _candidate_overrides(
        spec,
        theta_lr=float(theta_lr),
        nuisance_ratio=float(nuisance_ratio),
        actor_multiplier=None if actor_multiplier is None else float(actor_multiplier),
        score_scale=_score_scale(
            problem=problem,
            degree=int(degree),
            cache=score_scale_cache,
            sample_count=score_scale_samples,
            seed=score_scale_seed,
        ),
        alpha_max=float(alpha_max),
        alpha_min=float(alpha_min),
        alpha_warmup_frac=float(ADAPTIVE_ALPHA_FIXED["hybrid_alpha_warmup_frac"]),
        stage=stage,
        actor_scale_tuned=actor_multiplier is not None,
        generative_lambda_score=None,
    )
    overrides = _base_overrides(
        problem=problem,
        model_type=model_type,
        degree=int(degree),
        output_root=output_root,
        campaign=campaign,
    )
    overrides.update(candidate_overrides)
    overrides.pop("generative_lr", None)
    if is_generative:
        for key in (
            "policy_sampling_scale",
            "hybrid_tuning_score_scale",
            "hybrid_tuning_actor_scale_multiplier",
            "hybrid_tuning_actor_scale_tuned",
        ):
            overrides.pop(key, None)
        overrides["hybrid_tuning_candidate_label"] = (
            f"tlr={theta_lr:g} nlr={overrides['nuisance_lr']:g} "
            f"ratio={nuisance_ratio:g} "
            f"alpha=[{alpha_min:g},{alpha_max:g}] "
            f"warmup_frac={ADAPTIVE_ALPHA_FIXED['hybrid_alpha_warmup_frac']:g}"
        )
    is_diffusion = "diffusion" in str(model_type)
    if not is_diffusion:
        overrides.pop("generative_surrogate_num_iter", None)
    if diffusion_surrogate_num_iter is not None and is_diffusion:
        overrides["generative_surrogate_num_iter"] = int(diffusion_surrogate_num_iter)
        overrides["hybrid_tuning_diffusion_surrogate_num_iter"] = int(diffusion_surrogate_num_iter)
    elif diffusion_surrogate_num_iter is not None:
        overrides["hybrid_tuning_diffusion_surrogate_num_iter"] = None

    if is_generative and diffusion_surrogate_num_iter is not None and is_diffusion:
        extra_parts = [f"dsiter{int(diffusion_surrogate_num_iter)}"]
        candidate_id = f"{candidate_id}_{'_'.join(extra_parts)}"
        overrides["hybrid_tuning_candidate_id"] = candidate_id
        overrides["hybrid_tuning_candidate_label"] = (
            f"{overrides['hybrid_tuning_candidate_label']} "
            f"diffusion_surrogate_num_iter={int(diffusion_surrogate_num_iter)}"
        )

    if not is_generative:
        overrides["hybrid_tuning_stage"] = stage
        overrides["hybrid_tuning_grid_family"] = "point_lr_nuisance_actor_refine"
        name_prefix = f"problem_hybrid_tuning_{stage}"
    else:
        overrides["hybrid_tuning_stage"] = stage
        overrides["hybrid_tuning_grid_family"] = "generative_theta_lr_nuisance_alpha_refine"
        name_prefix = f"problem_hybrid_tuning_{stage}"

    name = f"{name_prefix}_{problem}_deg{int(degree)}_{model_type}_{candidate_id}"
    return {"name": name, "base": spec.base, "overrides": overrides}


def _round_robin(groups: dict[tuple[str, int, str, str], list[dict]]) -> list[dict]:
    queues = [deque(values) for _, values in sorted(groups.items())]
    ordered: list[dict] = []
    while queues:
        next_queues = []
        for queue in queues:
            if queue:
                ordered.append(queue.popleft())
            if queue:
                next_queues.append(queue)
        queues = next_queues
    return ordered


def _single_seed_experiments(candidate_groups: Iterable[dict], seeds: Iterable[int]) -> list[dict]:
    experiments: list[dict] = []
    for candidate in candidate_groups:
        name = str(candidate["name"])
        for seed in seeds:
            experiments.append(
                {
                    "name": f"{name}_seed{int(seed)}",
                    "base": candidate["base"],
                    "overrides": candidate["overrides"],
                    "seeds": [int(seed)],
                }
            )
    return experiments


def build_experiments(
    *,
    campaign: str,
    output_root: Path,
    problems: tuple[str, ...],
    seeds: list[int],
    generative_degrees: tuple[int, ...],
    pricing_degrees: tuple[int, ...],
    topk_sp_generative_theta_lr_grid: tuple[float, ...],
    pricing_generative_theta_lr_grid: tuple[float, ...],
    diffusion_surrogate_num_iter_grid: tuple[int, ...],
    nuisance_ratio_grid: dict[str, tuple[float, ...]],
    alpha_pairs: tuple[tuple[float, float], ...],
    alpha_refine_mode: str,
    tune_generative_alpha: bool,
    include_topk_sp_generative: bool,
    include_pricing_generative: bool,
    score_scale_samples: int,
    score_scale_seed: int,
) -> list[dict]:
    score_scale_cache: dict[tuple[str, int], float] = {}
    groups: dict[tuple[str, int, str, str], list[dict]] = defaultdict(list)
    core_alpha_pairs = alpha_pairs if alpha_refine_mode == "full" else (FIXED_ALPHA_PAIR,)
    anchor_alpha_pairs = tuple(pair for pair in alpha_pairs if pair != FIXED_ALPHA_PAIR)

    if include_topk_sp_generative:
        for problem in tuple(p for p in ("topk", "shortest_path") if p in problems):
            for degree in generative_degrees:
                for model_type in GENERATIVE_MODEL_TYPES[problem]:
                    iter_grid = diffusion_surrogate_num_iter_grid if "diffusion" in model_type else (None,)
                    for theta_lr, ratio, diffusion_iter, alpha_pair in itertools.product(
                        topk_sp_generative_theta_lr_grid,
                        nuisance_ratio_grid[problem],
                        iter_grid,
                        core_alpha_pairs,
                    ):
                        exp = _candidate(
                            problem=problem,
                            model_type=model_type,
                            degree=degree,
                            stage="generative_refine",
                            theta_lr=theta_lr,
                            nuisance_ratio=ratio,
                            actor_multiplier=None,
                            alpha_max=alpha_pair[0],
                            alpha_min=alpha_pair[1],
                            diffusion_surrogate_num_iter=(
                                None if diffusion_iter is None else int(diffusion_iter)
                            ),
                            output_root=output_root,
                            campaign=campaign,
                            score_scale_cache=score_scale_cache,
                            score_scale_samples=score_scale_samples,
                            score_scale_seed=score_scale_seed,
                        )
                        groups[(problem, int(degree), model_type, "generative")].append(exp)
                    if alpha_refine_mode == "anchor" and tune_generative_alpha:
                        anchor = GENERATIVE_ALPHA_ANCHORS[problem]
                        for alpha_pair in anchor_alpha_pairs:
                            exp = _candidate(
                                problem=problem,
                                model_type=model_type,
                                degree=degree,
                                stage="generative_alpha_refine",
                                theta_lr=float(anchor["theta_lr"]),
                                nuisance_ratio=float(anchor["nuisance_ratio"]),
                                actor_multiplier=None,
                                alpha_max=alpha_pair[0],
                                alpha_min=alpha_pair[1],
                                diffusion_surrogate_num_iter=(
                                    int(anchor["diffusion_surrogate_num_iter"]) if "diffusion" in model_type else None
                                ),
                                output_root=output_root,
                                campaign=campaign,
                                score_scale_cache=score_scale_cache,
                                score_scale_samples=score_scale_samples,
                                score_scale_seed=score_scale_seed,
                            )
                            groups[(problem, int(degree), model_type, "generative_alpha")].append(exp)

    if include_pricing_generative and "pricing" in problems:
        for degree in pricing_degrees:
            for model_type in GENERATIVE_MODEL_TYPES["pricing"]:
                iter_grid = diffusion_surrogate_num_iter_grid if "diffusion" in model_type else (None,)
                for theta_lr, ratio, diffusion_iter, alpha_pair in itertools.product(
                    pricing_generative_theta_lr_grid,
                    nuisance_ratio_grid["pricing"],
                    iter_grid,
                    core_alpha_pairs,
                ):
                    exp = _candidate(
                        problem="pricing",
                        model_type=model_type,
                        degree=degree,
                        stage="generative_refine",
                        theta_lr=theta_lr,
                        nuisance_ratio=ratio,
                        actor_multiplier=None,
                        alpha_max=alpha_pair[0],
                        alpha_min=alpha_pair[1],
                        diffusion_surrogate_num_iter=(
                            None if diffusion_iter is None else int(diffusion_iter)
                        ),
                        output_root=output_root,
                        campaign=campaign,
                        score_scale_cache=score_scale_cache,
                        score_scale_samples=score_scale_samples,
                        score_scale_seed=score_scale_seed,
                    )
                    groups[("pricing", int(degree), model_type, "generative")].append(exp)
                if alpha_refine_mode == "anchor" and tune_generative_alpha:
                    anchor = GENERATIVE_ALPHA_ANCHORS["pricing"]
                    for alpha_pair in anchor_alpha_pairs:
                        exp = _candidate(
                            problem="pricing",
                            model_type=model_type,
                            degree=degree,
                            stage="generative_alpha_refine",
                            theta_lr=float(anchor["theta_lr"]),
                            nuisance_ratio=float(anchor["nuisance_ratio"]),
                            actor_multiplier=None,
                            alpha_max=alpha_pair[0],
                            alpha_min=alpha_pair[1],
                            diffusion_surrogate_num_iter=(
                                int(anchor["diffusion_surrogate_num_iter"]) if "diffusion" in model_type else None
                            ),
                            output_root=output_root,
                            campaign=campaign,
                            score_scale_cache=score_scale_cache,
                            score_scale_samples=score_scale_samples,
                            score_scale_seed=score_scale_seed,
                        )
                        groups[("pricing", int(degree), model_type, "generative_alpha")].append(exp)

    return _single_seed_experiments(_round_robin(groups), seeds)


def _summarize(experiments: list[dict]) -> dict:
    counts: dict[str, int] = defaultdict(int)
    for exp in experiments:
        overrides = exp["overrides"]
        key = "/".join(
            [
                str(overrides["hybrid_tuning_grid_family"]),
                str(overrides["benchmark"]),
                f"deg{int(overrides['hybrid_tuning_degree'])}",
                str(overrides["model_type"]),
            ]
        )
        counts[key] += 1
    return dict(sorted(counts.items()))


def _validate_unique_names(experiments: list[dict]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for exp in experiments:
        name = str(exp["name"])
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise SystemExit(f"Duplicate experiment names generated: {preview}")


def _write_manifests(
    *,
    manifest_dir: Path,
    experiments: list[dict],
    problems: tuple[str, ...],
    num_seeds: int,
) -> tuple[Path, Path]:
    problem_slug = "all_models" if problems == ("topk", "shortest_path", "pricing") else "_".join(problems)
    manifest_name = f"problem_hybrid_tuning_{problem_slug}_refined_{int(num_seeds)}seed.yaml"
    compare_name = f"compare_problem_hybrid_tuning_{problem_slug}_refined_{int(num_seeds)}seed.yaml"
    manifest_path = manifest_dir / manifest_name
    compare_path = manifest_dir / compare_name
    write_yaml(manifest_path, {"experiments": experiments})
    write_yaml(compare_path, {"include": [manifest_name]})
    return manifest_path, compare_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default="tuning_runs")
    parser.add_argument(
        "--problem",
        action="append",
        choices=("topk", "shortest_path", "pricing"),
        default=[],
        help="Problem(s) to tune. May be repeated; defaults to all three main problems.",
    )
    parser.add_argument(
        "--campaign",
        default=None,
        help="Defaults to a timestamped generative refinement campaign.",
    )
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument(
        "--seed-start-index",
        type=int,
        default=30,
        help="Default matches the recent tuning seed slice.",
    )
    parser.add_argument("--generative-degrees", type=_parse_int_csv, default=GENERATIVE_DEGREES)
    parser.add_argument("--pricing-degrees", type=_parse_int_csv, default=PRICING_DEGREES)
    parser.add_argument(
        "--topk-sp-generative-theta-lrs",
        "--topk-sp-generative-lrs",
        dest="topk_sp_generative_theta_lrs",
        type=_parse_float_csv,
        default=TOPK_SP_GENERATIVE_THETA_LR_GRID,
    )
    parser.add_argument(
        "--pricing-generative-theta-lrs",
        "--pricing-generative-lrs",
        dest="pricing_generative_theta_lrs",
        type=_parse_float_csv,
        default=PRICING_GENERATIVE_THETA_LR_GRID,
    )
    parser.add_argument(
        "--diffusion-surrogate-num-iters",
        type=_parse_int_csv,
        default=DIFFUSION_SURROGATE_NUM_ITER_GRID,
        help="Diffusion-only Monte Carlo repeats used in the denoising likelihood surrogate.",
    )
    parser.add_argument(
        "--nuisance-ratios",
        dest="nuisance_ratios",
        type=_parse_float_csv,
        default=NUISANCE_RATIO_GRID["topk"],
        help="Nuisance/theta LR ratios used for all model families.",
    )
    parser.add_argument(
        "--alpha-pairs",
        type=_parse_alpha_pairs,
        default=REFINED_ALPHA_PAIR_GRID,
        help="Comma-separated adaptive alpha pairs as AMAX:AMIN.",
    )
    parser.add_argument(
        "--alpha-refine-mode",
        choices=("anchor", "full", "off"),
        default="anchor",
        help=(
            "anchor tunes alpha only at representative anchor configs; full crosses alpha "
            "with every candidate; off uses only alpha=(0.5,0.05)."
        ),
    )
    parser.set_defaults(tune_generative_alpha=True)
    parser.add_argument(
        "--tune-generative-alpha",
        dest="tune_generative_alpha",
        action="store_true",
        help="Run alpha anchor candidates for CNF/diffusion. This is the default.",
    )
    parser.add_argument(
        "--skip-generative-alpha",
        dest="tune_generative_alpha",
        action="store_false",
        help="Disable generative alpha anchor tuning for a smaller generative grid.",
    )
    parser.add_argument("--skip-topk-sp-generative", action="store_true")
    parser.add_argument("--skip-pricing-generative", action="store_true")
    parser.add_argument(
        "--skip-completed-from",
        action="append",
        type=Path,
        default=[],
        help=(
            "Prior campaign root or outputs directory to use for delta submissions. "
            "A generated seed-level run is skipped only when both DFHPG "
            "and RandomOracle metrics exist for the same tuning config and seed. "
            "May be passed multiple times."
        ),
    )
    parser.add_argument("--score-scale-samples", type=int, default=128)
    parser.add_argument("--score-scale-seed", type=int, default=314159)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    campaign = args.campaign or f"{DEFAULT_CAMPAIGN_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    problems = tuple(args.problem) if args.problem else ("topk", "shortest_path", "pricing")
    campaign_root = Path(args.artifact_root) / campaign
    manifest_dir = campaign_root / "manifests"
    output_root = campaign_root / "outputs"
    seeds = seed_list(args.num_seeds, start_index=args.seed_start_index)
    nuisance_ratio_grid = {
        "topk": tuple(args.nuisance_ratios),
        "shortest_path": tuple(args.nuisance_ratios),
        "pricing": tuple(args.nuisance_ratios),
    }

    experiments = build_experiments(
        campaign=campaign,
        output_root=output_root,
        problems=problems,
        seeds=seeds,
        generative_degrees=tuple(args.generative_degrees),
        pricing_degrees=tuple(args.pricing_degrees),
        topk_sp_generative_theta_lr_grid=tuple(args.topk_sp_generative_theta_lrs),
        pricing_generative_theta_lr_grid=tuple(args.pricing_generative_theta_lrs),
        diffusion_surrogate_num_iter_grid=tuple(args.diffusion_surrogate_num_iters),
        nuisance_ratio_grid=nuisance_ratio_grid,
        alpha_pairs=tuple(args.alpha_pairs),
        alpha_refine_mode=str(args.alpha_refine_mode),
        tune_generative_alpha=bool(args.tune_generative_alpha),
        include_topk_sp_generative=not args.skip_topk_sp_generative,
        include_pricing_generative=not args.skip_pricing_generative,
        score_scale_samples=args.score_scale_samples,
        score_scale_seed=args.score_scale_seed,
    )

    if not experiments:
        raise SystemExit("No experiments selected; all blocks were skipped.")
    _validate_unique_names(experiments)
    full_experiment_count = len(experiments)
    skipped_completed = 0
    if args.skip_completed_from:
        experiments, skipped_completed = _filter_completed(experiments, args.skip_completed_from)
        if not experiments:
            print(f"Campaign: {campaign}")
            print(f"Full grid experiments: {full_experiment_count} single-seed runs")
            print(f"Skipped completed: {skipped_completed}")
            print("Experiments: 0 single-seed runs")
            return
        _validate_unique_names(experiments)

    summary = _summarize(experiments)
    print(f"Campaign: {campaign}")
    if args.skip_completed_from:
        print(f"Full grid experiments: {full_experiment_count} single-seed runs")
        print(f"Skipped completed: {skipped_completed}")
    print(f"Experiments: {len(experiments)} single-seed runs")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if args.dry_run:
        return

    manifest_path, compare_path = _write_manifests(
        manifest_dir=manifest_dir,
        experiments=experiments,
        problems=problems,
        num_seeds=int(args.num_seeds),
    )
    print(f"Manifest: {manifest_path}")
    print(f"Compare: {compare_path}")
    print(f"Run locally: {sys.executable} scripts/utils/run_experiments.py --config {compare_path}")


if __name__ == "__main__":
    main()
