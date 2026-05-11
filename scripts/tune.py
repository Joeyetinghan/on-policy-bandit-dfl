"""Single entrypoint for tuning-manifest generation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POINT_PROBLEMS = ("topk", "shortest_path", "pricing")
DEFAULT_POINT_DEGREES = (2, 4, 6, 8, 10)


def _run(cmd: list[str], *, dry_run: bool = False) -> None:
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _point_compare_manifest(
    *,
    artifact_root: str,
    campaign: str,
    stage: str,
    problems: tuple[str, ...],
    degrees: tuple[int, ...],
    num_seeds: int,
) -> Path:
    problem_slug = "main" if problems == DEFAULT_POINT_PROBLEMS else "_".join(problems)
    degree_slug = "deg" + "_".join(str(degree) for degree in degrees)
    name = f"compare_problem_hybrid_tuning_{stage}_{problem_slug}_{degree_slug}_{num_seeds}seed.yaml"
    return REPO_ROOT / artifact_root / campaign / "manifests" / name


def _point_cmd(args: argparse.Namespace) -> None:
    campaign = args.campaign or "point_tuning_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    problems = tuple(args.problem) if args.problem else DEFAULT_POINT_PROBLEMS
    degrees = tuple(args.degree) if args.degree else DEFAULT_POINT_DEGREES

    generator_cmd = [
        sys.executable,
        "scripts/tuning/generate_problem_hybrid_tuning_manifests.py",
        "--stage",
        args.stage,
        "--model-family",
        "point",
        "--artifact-root",
        args.artifact_root,
        "--campaign",
        campaign,
        "--num-seeds",
        str(args.num_seeds),
        "--seed-start-index",
        str(args.seed_start_index),
    ]
    for problem in problems:
        generator_cmd.extend(["--problem", problem])
    for degree in degrees:
        generator_cmd.extend(["--degree", str(degree)])
    _run(generator_cmd, dry_run=args.dry_run)

    compare_path = _point_compare_manifest(
        artifact_root=args.artifact_root,
        campaign=campaign,
        stage=args.stage,
        problems=problems,
        degrees=degrees,
        num_seeds=args.num_seeds,
    )
    print(f"Campaign: {campaign}")
    print(f"Compare manifest: {compare_path}")
    print(f"Run locally: {sys.executable} scripts/utils/run_experiments.py --config {compare_path}")


def _generative_cmd(args: argparse.Namespace) -> None:
    campaign = args.campaign or "generative_tuning_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    problems = tuple(args.problem) if args.problem else DEFAULT_POINT_PROBLEMS
    cmd = [
        sys.executable,
        "scripts/tuning/prepare_generative_refine_tuning_jobs.py",
        "--artifact-root",
        args.artifact_root,
        "--campaign",
        campaign,
        "--num-seeds",
        str(args.num_seeds),
        "--seed-start-index",
        str(args.seed_start_index),
    ]
    for problem in problems:
        cmd.extend(["--problem", problem])
    if args.dry_run:
        cmd.append("--dry-run")
    _run(cmd, dry_run=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    point = subparsers.add_parser("point", help="Generate point-model DFHPG tuning manifests by problem.")
    point.add_argument(
        "--stage",
        choices=("alpha_pilot", "alpha_diag", "problem_grid", "full_grid"),
        default="problem_grid",
    )
    point.add_argument("--problem", action="append", choices=DEFAULT_POINT_PROBLEMS, default=[])
    point.add_argument("--degree", action="append", type=int, default=[])
    point.add_argument("--artifact-root", default="tuning_runs")
    point.add_argument("--campaign", default=None)
    point.add_argument("--num-seeds", type=int, default=10)
    point.add_argument("--seed-start-index", type=int, default=30)
    point.add_argument("--dry-run", action="store_true")
    point.set_defaults(func=_point_cmd)

    generative = subparsers.add_parser(
        "generative",
        help="Generate the maintained CNF/diffusion refinement manifests.",
    )
    generative.add_argument("--problem", action="append", choices=DEFAULT_POINT_PROBLEMS, default=[])
    generative.add_argument("--artifact-root", default="tuning_runs")
    generative.add_argument("--campaign", default=None)
    generative.add_argument("--num-seeds", type=int, default=10)
    generative.add_argument("--seed-start-index", type=int, default=30)
    generative.add_argument("--dry-run", action="store_true")
    generative.set_defaults(func=_generative_cmd)

    summarize = subparsers.add_parser(
        "summarize-point",
        help="Summarize point-model DFHPG tuning outputs and write selected configs.",
    )
    summarize.add_argument("--outputs", default="outputs")
    summarize.add_argument("--prefix", default=None)
    summarize.add_argument("--outdir", required=True)
    summarize.add_argument(
        "--selected-out",
        type=Path,
        default=None,
        help="Optional extra YAML path for selected tuned configs, e.g. configs/tuned/main.yaml.",
    )
    summarize.add_argument("--skip-selected", action="store_true")

    args = parser.parse_args()
    if args.command == "summarize-point":
        cmd = [
            sys.executable,
            "scripts/tuning/summarize_problem_hybrid_tuning.py",
            "--outputs",
            str(args.outputs),
            "--outdir",
            str(args.outdir),
        ]
        if args.prefix is not None:
            cmd.extend(["--prefix", str(args.prefix)])
        if args.selected_out is not None:
            cmd.extend(["--selected-out", str(args.selected_out)])
        if args.skip_selected:
            cmd.append("--skip-selected")
        _run(cmd)
        return args
    args.func(args)
    return args


if __name__ == "__main__":
    _parse_args()
