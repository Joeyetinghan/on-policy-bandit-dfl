#!/usr/bin/env python3
"""Single-entry CLI for reproducing the energy paper campaign.

Local subcommands run and write the standard
``results/<block>/{raw,summary,tables,figures}`` layout.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.paper.paper_runs import (  # noqa: E402
    BuildOptions,
    aggregate_block_results,
    build_reference_manifest,
    campaign_paths,
    dedupe_experiments_for_execution,
    execute_manifest,
    load_tuned_configs,
    write_block_manifest_json,
)
from scripts.paper.prepare_energy_experiment_runs import (  # noqa: E402
    DEFAULT_TUNED_CONFIGS,
    ENERGY_BLOCK_CHOICES,
    ENERGY_BLOCK_IDS,
    _canonical_energy_block_id,
    _replace_energy_alpha0_from_block01,
    _timestamped_campaign_name,
    _write_block_manifests,
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifact-root", default="paper_runs")
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--block", action="append", choices=ENERGY_BLOCK_CHOICES, default=[])
    parser.add_argument("--num-seeds", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--tuned-configs", type=Path, required=True)
    parser.add_argument("--allow-missing-tuned-configs", action="store_true")
    parser.add_argument("--energy-instance", default="load3")
    parser.add_argument("--dataset-split", default="experiment")
    parser.add_argument("--tuning-split", default="validation")
    parser.add_argument("--energy-validation-fraction", type=float, default=0.5)
    parser.add_argument("--data-root", default="data/Energy/SchedulingInstances")


def _campaign_root(args: argparse.Namespace) -> Path:
    campaign = args.campaign or _timestamped_campaign_name()
    return Path(args.artifact_root) / campaign


def _block_ids(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_canonical_energy_block_id(block_id) for block_id in (args.block or ENERGY_BLOCK_IDS)))


def _opts(args: argparse.Namespace, campaign_root: Path) -> BuildOptions:
    return BuildOptions(
        campaign_root=campaign_root,
        quick=bool(args.quick),
        num_seeds=int(args.num_seeds),
        device=str(args.device),
        eval_every=args.eval_every,
        tuned_configs=load_tuned_configs(args.tuned_configs),
        tuned_config_path=args.tuned_configs,
        require_tuned_configs=not bool(args.allow_missing_tuned_configs),
    )


def _prepare(args: argparse.Namespace) -> tuple[Path, list[dict]]:
    campaign_root = _campaign_root(args)
    opts = _opts(args, campaign_root)
    manifest_paths, experiments = _write_block_manifests(
        campaign_root=campaign_root,
        opts=opts,
        block_ids=_block_ids(args),
        args=args,
        dry_run=bool(getattr(args, "dry_run", False)),
    )
    print(f"Campaign: {campaign_root}")
    print(f"Energy blocks: {', '.join(_block_ids(args))}")
    print(f"Manifest experiment groups: {len(experiments)}")
    if getattr(args, "dry_run", False):
        print(f"[DRY RUN] Would write {len(manifest_paths)} block manifests")
    return campaign_root, experiments


def _run(args: argparse.Namespace) -> Path:
    campaign_root, experiments = _prepare(args)
    references = build_reference_manifest(experiments)
    completed, skipped, failed = execute_manifest(
        references,
        resume=bool(getattr(args, "resume", True)),
        dry_run=bool(getattr(args, "dry_run", False)),
    )
    print(f"TrueModel references: completed={completed} skipped={skipped} failed={failed}")
    if failed:
        raise SystemExit(1)
    execution_experiments, deduped = dedupe_experiments_for_execution(experiments)
    manifest = {"experiments": execution_experiments}
    completed, skipped, failed = execute_manifest(
        manifest,
        resume=bool(getattr(args, "resume", True)),
        dry_run=bool(getattr(args, "dry_run", False)),
    )
    print(f"Energy experiments: completed={completed} skipped={skipped} failed={failed} deduped={deduped}")
    if failed:
        raise SystemExit(1)
    return campaign_root


def _aggregate(args: argparse.Namespace) -> Path:
    campaign_root = _campaign_root(args)
    paths = campaign_paths(campaign_root)
    opts = _opts(args, campaign_root)
    start = time.time()
    for block_id in _block_ids(args):
        result_dir = paths["results"] / block_id
        aggregate = aggregate_block_results(block_id, result_dir=result_dir, output_root=paths["outputs"])
        if block_id == "03_alpha_diagnostics":
            _replace_energy_alpha0_from_block01(campaign_root)
        write_block_manifest_json(
            result_dir=result_dir,
            block_id=block_id,
            command=sys.argv,
            manifest_path=paths["manifests"] / f"block_{block_id}.yaml",
            opts=opts,
            runtime_sec=time.time() - start,
        )
        print(f"{block_id}: aggregate={aggregate}")
    return campaign_root


def _plot(args: argparse.Namespace) -> None:
    campaign_root = _campaign_root(args)
    blocks = ",".join(_block_ids(args))
    subprocess.run(
        [
            sys.executable,
            "scripts/paper/plot_paper_custom_figures.py",
            "--campaign",
            str(campaign_root),
            "--blocks",
            blocks,
            "--width",
            str(getattr(args, "width", 5.2)),
            "--height",
            str(getattr(args, "height", 3.8)),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    if "02_generative_ablation" in _block_ids(args):
        subprocess.run(
            [sys.executable, "scripts/paper/plot_energy_block02_with_linear.py", "--campaign", str(campaign_root)],
            cwd=REPO_ROOT,
            check=True,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "aggregate", "plot", "all"):
        sub = subparsers.add_parser(name)
        _common(sub)
        if name in {"prepare", "run"}:
            sub.add_argument("--dry-run", action="store_true")
        if name == "run":
            sub.add_argument("--resume", action="store_true", default=True)
            sub.add_argument("--force-rerun", dest="resume", action="store_false")
        if name == "plot":
            sub.add_argument("--width", type=float, default=5.2)
            sub.add_argument("--height", type=float, default=3.8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "prepare":
        _prepare(args)
    elif args.command == "run":
        _run(args)
    elif args.command == "aggregate":
        _aggregate(args)
    elif args.command == "plot":
        _plot(args)
    elif args.command == "all":
        _run(args)
        _aggregate(args)
        _plot(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
