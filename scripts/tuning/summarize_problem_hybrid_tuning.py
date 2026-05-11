"""Summarize DFHPG tuning outputs and selected configs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiment.algorithm_names import dfhpg_metrics_path

from scripts.lib.io import load_yaml, write_yaml


pd: Any = None


def _require_pandas() -> Any:
    global pd
    if pd is None:
        import pandas as _pd

        pd = _pd
    return pd


def _load_yaml(path: Path) -> dict:
    return load_yaml(path) or {}


def _avg_cum_expected_objective(metrics_path: Path) -> float | None:
    for key in ("cum_expected_objective", "cum_objective"):
        result = _avg_last_json_array_value(metrics_path, key)
        if result is not None:
            return result
    return None


def _avg_last_json_array_value(path: Path, key: str) -> float | None:
    data = path.read_bytes()
    start = data.find(f'  "{key}": [\n'.encode("utf-8"))
    if start < 0:
        return None
    start += len(f'  "{key}": [\n'.encode("utf-8"))
    end = data.find(b"\n  ]", start)
    if end < 0:
        return None
    block = data[start:end].strip()
    if not block:
        return None
    count = block.count(b"\n") + 1
    last_line = block.rsplit(b"\n", 1)[-1].strip().rstrip(b",")
    try:
        last_value = float(last_line)
    except ValueError:
        return None
    return last_value / float(count)


def collect_rows(outputs: Path, prefix: str | None = None) -> pd.DataFrame:
    pandas = _require_pandas()
    rows: list[dict] = []
    for run_dir in sorted(outputs.iterdir()):
        if not run_dir.is_dir():
            continue
        if prefix and not run_dir.name.startswith(prefix):
            continue
        cfg_path = run_dir / "config.yaml"
        metrics_path = dfhpg_metrics_path(run_dir)
        if not cfg_path.exists() or not metrics_path.exists():
            continue
        cfg = _load_yaml(cfg_path)
        if "hybrid_tuning_problem" not in cfg:
            continue
        avg_cost = _avg_cum_expected_objective(metrics_path)
        if avg_cost is None:
            continue
        rows.append(
            {
                "run_name": run_dir.name,
                "seed": int(cfg.get("seed", 0)),
                "problem": str(cfg["hybrid_tuning_problem"]),
                "benchmark": str(cfg.get("benchmark", "")),
                "setting_id": str(cfg.get("hybrid_tuning_setting_id", "")),
                "degree": int(cfg.get("hybrid_tuning_degree", cfg.get("deg", 0) or cfg.get("pricing_context_degree", 0))),
                "model_type": str(cfg.get("model_type", "")),
                "candidate_id": str(cfg.get("hybrid_tuning_candidate_id", "")),
                "candidate_label": str(cfg.get("hybrid_tuning_candidate_label", "")),
                "stage": str(cfg.get("hybrid_tuning_stage", "")),
                "theta_lr_schedule": str(cfg.get("theta_lr_schedule", "constant")),
                "theta_lr": float(cfg.get("theta_lr", 0.01)),
                "theta_lr_offset": float(cfg.get("theta_lr_offset", 0.0)),
                "nuisance_lr_schedule": str(cfg.get("nuisance_lr_schedule", "constant")),
                "nuisance_lr": float(cfg.get("nuisance_lr", cfg.get("theta_lr", 0.01))),
                "nuisance_lr_offset": float(cfg.get("nuisance_lr_offset", 0.0)),
                "model_update_batch_rounds": int(cfg.get("model_update_batch_rounds", 1)),
                "nuisance_update_batch_rounds": int(cfg.get("nuisance_update_batch_rounds", 1)),
                "hybrid_loss_type": str(cfg.get("hybrid_loss_type", "")),
                "hybrid_mse_weight": float(cfg.get("hybrid_mse_weight", 0.0)),
                "policy_sampling_scale": float(cfg.get("policy_sampling_scale", 0.01)),
                "hybrid_alpha_schedule": str(cfg.get("hybrid_alpha_schedule", "")),
                "hybrid_alpha_max": float(cfg.get("hybrid_alpha_max", cfg.get("hybrid_alpha_init", 0.7))),
                "hybrid_alpha_min": float(cfg.get("hybrid_alpha_min", cfg.get("hybrid_alpha_final", 0.1))),
                "hybrid_alpha_gate": str(cfg.get("hybrid_alpha_gate", "warmup_frac")),
                "hybrid_alpha_time_scale": float(cfg.get("hybrid_alpha_time_scale", 100.0)),
                "hybrid_alpha_warmup_frac": float(cfg.get("hybrid_alpha_warmup_frac", 0.05)),
                "hybrid_alpha_ema_decay": float(cfg.get("hybrid_alpha_ema_decay", 0.98)),
                "hybrid_alpha_smooth": float(cfg.get("hybrid_alpha_smooth", 0.05)),
                "hybrid_gradient_normalization": bool(cfg.get("hybrid_gradient_normalization", True)),
                "policy_baseline_type": str(cfg.get("policy_baseline_type", "")),
                "hybrid_tuning_score_scale": float(cfg.get("hybrid_tuning_score_scale", 1.0)),
                "hybrid_tuning_actor_scale_multiplier": float(
                    cfg.get("hybrid_tuning_actor_scale_multiplier", 1.0)
                ),
                "hybrid_tuning_actor_scale_tuned": bool(cfg.get("hybrid_tuning_actor_scale_tuned", True)),
                "hybrid_tuning_nuisance_lr_ratio": float(cfg.get("hybrid_tuning_nuisance_lr_ratio", 1.0)),
                "hybrid_tuning_mixing_weight_type": str(
                    cfg.get("hybrid_tuning_mixing_weight_type", "adaptive_hybrid_alpha")
                ),
                "hybrid_tuning_generative_lambda_score": float(
                    cfg.get("hybrid_tuning_generative_lambda_score", cfg.get("generative_lambda_score", -1.0))
                ),
                "generative_surrogate_num_iter": int(cfg.get("generative_surrogate_num_iter") or -1),
                "hybrid_tuning_diffusion_surrogate_num_iter": int(
                    cfg.get("hybrid_tuning_diffusion_surrogate_num_iter") or -1
                ),
                "avg_cum_expected_objective": avg_cost,
            }
        )
    return pandas.DataFrame(rows)


GROUP_COLS = [
    "problem",
    "setting_id",
    "degree",
    "candidate_id",
    "candidate_label",
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
    "hybrid_alpha_max",
    "hybrid_alpha_min",
    "hybrid_alpha_gate",
    "hybrid_alpha_time_scale",
    "hybrid_alpha_warmup_frac",
    "hybrid_alpha_ema_decay",
    "hybrid_alpha_smooth",
    "hybrid_gradient_normalization",
    "policy_baseline_type",
    "hybrid_tuning_score_scale",
    "hybrid_tuning_actor_scale_multiplier",
    "hybrid_tuning_actor_scale_tuned",
    "hybrid_tuning_nuisance_lr_ratio",
    "hybrid_tuning_mixing_weight_type",
    "hybrid_tuning_generative_lambda_score",
    "generative_surrogate_num_iter",
    "hybrid_tuning_diffusion_surrogate_num_iter",
]


def summarize_by_model(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model_type", *GROUP_COLS], as_index=False)
        .agg(
            mean_avg_cum_expected_objective=("avg_cum_expected_objective", "mean"),
            std_avg_cum_expected_objective=("avg_cum_expected_objective", "std"),
            runs=("seed", "count"),
        )
        .sort_values(["problem", "model_type", "mean_avg_cum_expected_objective"])
    )


def summarize_by_problem(model_summary: pd.DataFrame) -> pd.DataFrame:
    expected = model_summary.groupby(["problem", "setting_id"], as_index=False).agg(
        expected_model_types=("model_type", "nunique")
    )
    problem_summary = (
        model_summary.groupby(GROUP_COLS, as_index=False)
        .agg(
            mean_avg_cum_expected_objective=("mean_avg_cum_expected_objective", "mean"),
            max_avg_cum_expected_objective=("mean_avg_cum_expected_objective", "max"),
            model_types_covered=("model_type", "nunique"),
            total_runs=("runs", "sum"),
        )
        .merge(expected, on=["problem", "setting_id"], how="left")
    )
    problem_summary = problem_summary[
        problem_summary["model_types_covered"] == problem_summary["expected_model_types"]
    ].sort_values(["problem", "mean_avg_cum_expected_objective", "max_avg_cum_expected_objective"])
    return problem_summary


def selected_configs(model_summary: pd.DataFrame) -> dict:
    selected: dict[str, dict] = {}
    if model_summary.empty:
        return selected
    for (problem, setting_id, model_type), sub in model_summary.groupby(["problem", "setting_id", "model_type"]):
        row = sub.sort_values(["mean_avg_cum_expected_objective", "std_avg_cum_expected_objective"]).iloc[0]
        key = f"{problem}/{setting_id}/{model_type}"
        selected[key] = {
            "problem": str(problem),
            "setting_id": str(setting_id),
            "degree": int(row["degree"]),
            "model_type": str(model_type),
            "theta_lr_schedule": str(row["theta_lr_schedule"]),
            "theta_lr": float(row["theta_lr"]),
            "theta_lr_offset": float(row["theta_lr_offset"]),
            "nuisance_lr_schedule": str(row["nuisance_lr_schedule"]),
            "nuisance_lr": float(row["nuisance_lr"]),
            "nuisance_lr_offset": float(row["nuisance_lr_offset"]),
            "model_update_batch_rounds": int(row["model_update_batch_rounds"]),
            "nuisance_update_batch_rounds": int(row["nuisance_update_batch_rounds"]),
            "hybrid_loss_type": str(row["hybrid_loss_type"]),
            "hybrid_mse_weight": float(row["hybrid_mse_weight"]),
            "policy_sampling_scale": float(row["policy_sampling_scale"]),
            "hybrid_alpha_schedule": str(row["hybrid_alpha_schedule"]),
            "hybrid_alpha_max": float(row["hybrid_alpha_max"]),
            "hybrid_alpha_min": float(row["hybrid_alpha_min"]),
            "hybrid_alpha_gate": str(row["hybrid_alpha_gate"]),
            "hybrid_alpha_time_scale": float(row["hybrid_alpha_time_scale"]),
            "hybrid_alpha_warmup_frac": float(row["hybrid_alpha_warmup_frac"]),
            "hybrid_alpha_ema_decay": float(row["hybrid_alpha_ema_decay"]),
            "hybrid_alpha_smooth": float(row["hybrid_alpha_smooth"]),
            "hybrid_gradient_normalization": bool(row["hybrid_gradient_normalization"]),
            "policy_baseline_type": str(row["policy_baseline_type"]),
            "hybrid_tuning_candidate_id": str(row["candidate_id"]),
            "hybrid_tuning_score_scale": float(row["hybrid_tuning_score_scale"]),
            "hybrid_tuning_actor_scale_multiplier": float(row["hybrid_tuning_actor_scale_multiplier"]),
            "hybrid_tuning_actor_scale_tuned": bool(row["hybrid_tuning_actor_scale_tuned"]),
            "hybrid_tuning_nuisance_lr_ratio": float(row["hybrid_tuning_nuisance_lr_ratio"]),
            "hybrid_tuning_mixing_weight_type": str(row["hybrid_tuning_mixing_weight_type"]),
            "hybrid_tuning_generative_lambda_score": float(row["hybrid_tuning_generative_lambda_score"]),
            "generative_surrogate_num_iter": (
                None if int(row["generative_surrogate_num_iter"]) < 0 else int(row["generative_surrogate_num_iter"])
            ),
            "hybrid_tuning_diffusion_surrogate_num_iter": (
                None
                if int(row["hybrid_tuning_diffusion_surrogate_num_iter"]) < 0
                else int(row["hybrid_tuning_diffusion_surrogate_num_iter"])
            ),
            "mean_avg_cum_expected_objective": float(row["mean_avg_cum_expected_objective"]),
            "std_avg_cum_expected_objective": (
                None
                if pd.isna(row["std_avg_cum_expected_objective"])
                else float(row["std_avg_cum_expected_objective"])
            ),
            "runs": int(row["runs"]),
        }
    return selected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--selected-out",
        type=Path,
        default=None,
        help="Optional extra YAML path for the selected tuned configs.",
    )
    parser.add_argument(
        "--skip-selected",
        action="store_true",
        help="Write CSV summaries only, without selected-config YAML output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = collect_rows(Path(args.outputs), prefix=args.prefix)
    if df.empty:
        raise SystemExit("No problem-level hybrid tuning outputs found.")

    model_summary = summarize_by_model(df)
    problem_summary = summarize_by_problem(model_summary)
    selected = selected_configs(model_summary)

    df.to_csv(outdir / "problem_hybrid_tuning_runs.csv", index=False)
    model_summary.to_csv(outdir / "problem_hybrid_tuning_by_model.csv", index=False)
    problem_summary.to_csv(outdir / "problem_hybrid_tuning_by_problem.csv", index=False)
    selected_paths: list[Path] = []
    if not args.skip_selected:
        selected_paths.append(outdir / "selected_problem_hybrid_configs.yaml")
    if args.selected_out is not None:
        selected_paths.append(args.selected_out)
    for path in selected_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_yaml(path, selected)
    print(f"Saved summaries to {outdir}")
    for path in selected_paths:
        print(f"Saved selected configs to {path}")


if __name__ == "__main__":
    main()
