"""Build a compact Slurm array runner for an experiment manifest."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lib.io import load_yaml, write_yaml


GPU_DEFAULT_QOS = "embers"
CPU_DEFAULT_PARTITION = "cpu-small"
CPU_DEFAULT_QOS = "embers"


def _load_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return json.loads(text) or {}
    return load_yaml(path) or {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact Slurm runner for a compare manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--experiments-per-task", type=int, default=8)
    parser.add_argument("--time-limit", default="08:00:00")
    parser.add_argument("--partition", default=None)
    parser.add_argument("--qos", default=None)
    parser.add_argument(
        "--account",
        default=os.environ.get("SLURM_ACCOUNT", "your-slurm-account"),
        help="Slurm account. Defaults to $SLURM_ACCOUNT if set.",
    )
    parser.add_argument("--gpu-type", default="rtx_6000")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--mem", default="4G")
    parser.add_argument("--max-parallel-jobs", type=int, default=None)
    parser.add_argument("--log-dir", default=None, help="Directory for Slurm stdout and per-task runner logs.")
    parser.add_argument("--split-seeds", action="store_true", help="Shard each seed as its own experiment group.")
    parser.add_argument(
        "--runtime-slice",
        action="store_true",
        help="Avoid writing per-array-task shard YAMLs; each task slices the manifest at runtime.",
    )
    parser.add_argument(
        "--no-dedup-baselines",
        action="store_true",
        help="Pass through to run_experiments.py so each shard runs baseline sanity checks for every config.",
    )
    return parser.parse_args()


def _resolve_scheduler_args(args: argparse.Namespace) -> tuple[str, str | None]:
    partition = args.partition
    qos = args.qos
    if args.cpu_only:
        if partition in {None, GPU_DEFAULT_QOS}:
            partition = CPU_DEFAULT_PARTITION
        if qos is None:
            qos = CPU_DEFAULT_QOS
        return partition, qos

    if qos is not None:
        if partition is None:
            raise ValueError("--partition is required when --qos is set")
        return partition, qos

    if partition is None:
        partition = GPU_DEFAULT_QOS
    return partition, None


def main() -> None:
    args = _parse_args()
    partition, qos = _resolve_scheduler_args(args)
    manifest = Path(args.manifest).resolve()
    run_dir = Path(args.run_dir).resolve()
    log_dir = Path(args.log_dir).resolve() if args.log_dir else REPO_ROOT / "logs" / "tune"
    shard_dir = run_dir / "shards"
    if not args.runtime_slice:
        shard_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    compare = _load_json(manifest)
    experiments = []
    for inc in compare["include"]:
        path = (manifest.parent / inc).resolve()
        experiments.extend(_load_json(path)["experiments"])
    if args.split_seeds:
        split_experiments = []
        for exp in experiments:
            seeds = exp.get("seeds", [None])
            for seed in seeds:
                split_exp = dict(exp)
                split_exp["seeds"] = [seed]
                if seed is not None:
                    split_exp["name"] = f"{exp.get('name', 'exp')}_seed{seed}"
                split_experiments.append(split_exp)
        experiments = split_experiments

    num_tasks = math.ceil(len(experiments) / args.experiments_per_task)
    max_parallel_jobs = num_tasks if args.max_parallel_jobs is None else max(1, min(args.max_parallel_jobs, num_tasks))
    if not args.runtime_slice:
        for i in range(num_tasks):
            chunk = experiments[i * args.experiments_per_task : (i + 1) * args.experiments_per_task]
            shard_path = shard_dir / f"shard_{i + 1:03d}.yaml"
            write_yaml(shard_path, {"experiments": chunk})

    submit_path = run_dir / "submit.sbatch"
    gpu_count = 0 if args.cpu_only else 1
    meta_text = "\n".join(
        [
            f"manifest: {manifest}",
            f"tag: {run_dir.name}",
            f"experiment_groups: {len(experiments)}",
            f"gpus_per_job: {gpu_count}",
            f"array_jobs: {num_tasks}",
            f"array_spec: 1-{num_tasks}%{max_parallel_jobs}",
            f"max_parallel_jobs: {max_parallel_jobs}",
            f"runtime_slice: {args.runtime_slice}",
            f"experiments_per_task: {args.experiments_per_task}",
            f"split_seeds: {args.split_seeds}",
            f"account: {args.account}",
            f"partition: {partition}",
            f"qos: {qos}",
            f"time_limit: {args.time_limit}",
            f"job_name: {args.job_name}",
            f"log_dir: {log_dir}",
            f"sbatch: {submit_path}",
        ]
    )
    (run_dir / "meta.yaml").write_text(meta_text + "\n", encoding="utf-8")

    sbatch_lines = [
        "#!/bin/bash",
        f"#SBATCH --account={args.account}",
        "#SBATCH -N1",
    ]
    if args.cpu_only:
        sbatch_lines.extend(
            [
                f"#SBATCH --cpus-per-task={args.cpus_per_task}",
                f"#SBATCH --mem={args.mem}",
                f"#SBATCH -p {partition}",
            ]
        )
    else:
        gpu_type = str(args.gpu_type or "").strip().lower()
        gres = "gpu:1" if gpu_type in {"", "any", "generic"} else f"gpu:{args.gpu_type}:1"
        sbatch_lines.extend(
            [
                f"#SBATCH --cpus-per-task={args.cpus_per_task}",
                f"#SBATCH --gres={gres}",
                "#SBATCH --gres-flags=enforce-binding",
            ]
        )
        if qos:
            sbatch_lines.append(f"#SBATCH -p {partition}")
        else:
            # Backward compatibility for older GPU runner invocations that passed
            # the QoS name via --partition.
            sbatch_lines.append(f"#SBATCH -q {partition}")
    if qos:
        sbatch_lines.append(f"#SBATCH -q {qos}")
    sbatch_lines.extend(
        [
            f"#SBATCH -t {args.time_limit}",
            f"#SBATCH --job-name={args.job_name}",
            f"#SBATCH --output={log_dir}/{run_dir.name}_%A_%a.out",
            f"#SBATCH --array=1-{num_tasks}%{max_parallel_jobs}",
            "",
            "set -euo pipefail",
            "",
            f'REPO_ROOT="{REPO_ROOT}"',
            f'RUN_DIR="{run_dir}"',
            f'LOG_DIR="{log_dir}"',
            f'MANIFEST="{manifest}"',
            f"EXPERIMENTS_PER_TASK={args.experiments_per_task}",
            "",
            'mkdir -p "$LOG_DIR"',
            'cd "$REPO_ROOT"',
            "",
            "set +u",
            "module load gurobi >/dev/null 2>&1",
            "module load mamba >/dev/null 2>&1",
            "mamba activate dfl",
            "set -u",
            "export PYTHONUNBUFFERED=1",
            "export OMP_NUM_THREADS=1",
            'export GUROBI_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
        ]
    )
    if args.cpu_only:
        sbatch_lines.append('export CUDA_VISIBLE_DEVICES=""')
    else:
        sbatch_lines.append("export CUDA_VISIBLE_DEVICES=0")
    run_experiments_args = ' --no-dedup-baselines' if args.no_dedup_baselines else ''
    if args.split_seeds:
        run_experiments_args += " --split-seeds"
    sbatch_lines.append("")
    if args.runtime_slice:
        sbatch_lines.extend(
            [
                f'task_log=$(printf "%s/%s_A%s_a%s.log" "$LOG_DIR" "{args.job_name}" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID")',
                (
                    'python scripts/utils/run_experiments.py --config "$MANIFEST" '
                    '--shard-index "$SLURM_ARRAY_TASK_ID" '
                    f'--shard-size "$EXPERIMENTS_PER_TASK"{run_experiments_args} > "$task_log" 2>&1'
                ),
                "",
            ]
        )
    else:
        sbatch_lines.extend(
            [
                'SHARD_DIR="$RUN_DIR/shards"',
                'shard_path=$(printf "%s/shard_%03d.yaml" "$SHARD_DIR" "$SLURM_ARRAY_TASK_ID")',
                f'task_log=$(printf "%s/%s_A%s_a%s.log" "$LOG_DIR" "{args.job_name}" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID")',
                f'python scripts/utils/run_experiments.py --config "$shard_path"{run_experiments_args} > "$task_log" 2>&1',
                "",
            ]
        )
    submit_text = "\n".join(sbatch_lines)
    submit_path.write_text(submit_text, encoding="utf-8")
    submit_path.chmod(0o755)

    print(f"Wrote runner at {run_dir}")
    print(f"Experiment groups: {len(experiments)}")
    print(f"Array jobs: {num_tasks}")


if __name__ == "__main__":
    main()
