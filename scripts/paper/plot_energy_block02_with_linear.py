#!/usr/bin/env python3
"""Plot energy block-02 distribution-family comparison including linear.

Energy block 02 evaluates NN/CNF/diffusion directly, while Gaussian Linear is
already available from block 01.  This paper-facing helper combines those
completed outputs into one block-02 style trajectory plot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiment.algorithm_names import dfhpg_metrics_path


ACTORS = ("gaussian_linear", "gaussian_nn", "cnf", "diffusion")
ACTOR_LABELS = {
    "gaussian_linear": "Gaussian Linear",
    "gaussian_nn": "Gaussian NN",
    "cnf": "CNF",
    "diffusion": "Diffusion",
}
ACTOR_STYLES = {
    "gaussian_linear": {"color": "#4D4D4D", "linestyle": "-", "marker": "o"},
    "gaussian_nn": {"color": "#0072B2", "linestyle": (0, (5.0, 2.0)), "marker": "s"},
    "cnf": {"color": "#009E73", "linestyle": (0, (1.2, 1.6)), "marker": "^"},
    "diffusion": {"color": "#D55E00", "linestyle": (0, (3.0, 1.4, 1.0, 1.4)), "marker": "D"},
}
METHODS = ("DFHPG", "DFHPG-0")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _cum_series(metrics: dict[str, Any]) -> np.ndarray:
    values = metrics.get("cum_expected_objective") or metrics.get("cum_objective")
    return np.asarray(values, dtype=float)


def _regret_series(metrics: dict[str, Any], true_metrics: dict[str, Any]) -> np.ndarray:
    values = _cum_series(metrics)
    true_values = _cum_series(true_metrics)
    n = min(len(values), len(true_values))
    sense = str(metrics.get("objective_sense", true_metrics.get("objective_sense", "min"))).lower()
    return true_values[:n] - values[:n] if sense == "max" else values[:n] - true_values[:n]


def _manifest_experiments(campaign: Path, block_id: str) -> list[dict[str, Any]]:
    path = campaign / "manifests" / f"block_{block_id}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")).get("experiments", [])


def _collect_actor(
    *,
    campaign: Path,
    block_id: str,
    actor_family: str,
) -> list[dict[str, Any]]:
    rows = []
    for experiment in _manifest_experiments(campaign, block_id):
        cfg = experiment.get("overrides", {})
        if cfg.get("paper_actor_family") != actor_family:
            continue
        method = str(cfg.get("paper_method", ""))
        if method not in METHODS:
            continue
        metrics_path = dfhpg_metrics_path(cfg["output_dir"])
        true_path = Path(cfg["paper_reference_output_dir"]) / "TrueModel" / "metrics.json"
        if not metrics_path.exists() or not true_path.exists():
            continue
        metrics = _load_json(metrics_path)
        true_metrics = _load_json(true_path)
        regret = _regret_series(metrics, true_metrics)
        rounds = np.arange(1, len(regret) + 1, dtype=float)
        rows.append(
            {
                "actor": actor_family,
                "method": method,
                "seed": int(cfg.get("seed", 0)),
                "avg_regret": regret / rounds,
                "final_avg_regret_per_round": float(regret[-1] / len(regret)),
                "final_cum_regret": float(regret[-1]),
                "candidate_id": cfg.get("paper_tuned_config_candidate_id", ""),
                "metrics_path": str(metrics_path),
            }
        )
    return rows


def _mean_sem(series: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    min_len = min(len(item) for item in series)
    arr = np.vstack([item[:min_len] for item in series])
    mean = arr.mean(axis=0)
    sem = np.zeros_like(mean) if arr.shape[0] < 2 else arr.std(axis=0, ddof=1) / math.sqrt(arr.shape[0])
    return mean, sem


def _write_tables(rows: list[dict[str, Any]], result_dir: Path) -> None:
    raw_dir = result_dir / "raw"
    summary_dir = result_dir / "summary"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / "energy_block02_with_linear_traces.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["actor", "method", "seed", "round", "avg_regret_per_round", "candidate_id", "metrics_path"],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["method"], r["actor"], r["seed"])):
            for idx, value in enumerate(row["avg_regret"], start=1):
                writer.writerow(
                    {
                        "actor": row["actor"],
                        "method": row["method"],
                        "seed": row["seed"],
                        "round": idx,
                        "avg_regret_per_round": float(value),
                        "candidate_id": row["candidate_id"],
                        "metrics_path": row["metrics_path"],
                    }
                )

    summary_rows = []
    for method in METHODS:
        for actor in ACTORS:
            vals = [row for row in rows if row["method"] == method and row["actor"] == actor]
            finals = np.asarray([row["final_avg_regret_per_round"] for row in vals], dtype=float)
            candidates = sorted({str(row["candidate_id"]) for row in vals if str(row["candidate_id"])})
            summary_rows.append(
                {
                    "method": method,
                    "actor": actor,
                    "label": ACTOR_LABELS[actor],
                    "n": len(vals),
                    "mean_final_avg_regret_per_round": float(finals.mean()) if len(finals) else math.nan,
                    "sem_final_avg_regret_per_round": (
                        0.0 if len(finals) < 2 else float(finals.std(ddof=1) / math.sqrt(len(finals)))
                    ),
                    "candidate_ids": ";".join(candidates),
                }
            )
    summary_path = summary_dir / "energy_block02_with_linear_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    canonical_rows = [row for row in rows if row["method"] == "DFHPG"]
    canonical_trace_path = raw_dir / "traces.csv"
    with canonical_trace_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["problem", "actor", "seed", "round", "avg_regret_per_round"])
        writer.writeheader()
        for row in sorted(canonical_rows, key=lambda r: (r["actor"], r["seed"])):
            actor = {"gaussian_linear": "linear", "gaussian_nn": "nn"}.get(row["actor"], row["actor"])
            for idx, value in enumerate(row["avg_regret"], start=1):
                writer.writerow(
                    {
                        "problem": "energy",
                        "actor": actor,
                        "seed": row["seed"],
                        "round": idx,
                        "avg_regret_per_round": float(value),
                    }
                )

    canonical_summary_rows = []
    for actor in ACTORS:
        vals = [row for row in canonical_rows if row["actor"] == actor]
        finals = np.asarray([row["final_avg_regret_per_round"] for row in vals], dtype=float)
        rounds = min((len(row["avg_regret"]) for row in vals), default=0)
        canonical_summary_rows.append(
            {
                "problem": "energy",
                "actor": {"gaussian_linear": "linear", "gaussian_nn": "nn"}.get(actor, actor),
                "label": ACTOR_LABELS[actor],
                "n": len(vals),
                "rounds": rounds,
                "mean_final_avg_regret_per_round": float(finals.mean()) if len(finals) else math.nan,
                "sem_final_avg_regret_per_round": (
                    0.0 if len(finals) < 2 else float(finals.std(ddof=1) / math.sqrt(len(finals)))
                ),
            }
        )
    canonical_summary_path = summary_dir / "summary.csv"
    with canonical_summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(canonical_summary_rows[0]))
        writer.writeheader()
        writer.writerows(canonical_summary_rows)


def _load_old_gen_fallback(result_dir: Path) -> list[dict[str, Any]]:
    path = result_dir / "raw" / "energy_block02_with_linear_old_gen_traces.csv"
    if not path.exists():
        return []

    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            actor = row.get("actor_family", "")
            if actor not in {"cnf", "diffusion"}:
                continue
            grouped[(actor, int(row.get("seed", 0)))].append(row)

    rows: list[dict[str, Any]] = []
    for (actor, seed), values in grouped.items():
        values.sort(key=lambda row: int(row["round"]))
        avg_regret = np.asarray([float(row["avg_regret_per_round"]) for row in values], dtype=float)
        cum_regret = np.asarray([float(row["cum_regret"]) for row in values], dtype=float)
        rows.append(
            {
                "actor": actor,
                "method": "DFHPG",
                "seed": seed,
                "avg_regret": avg_regret,
                "final_avg_regret_per_round": float(avg_regret[-1]),
                "final_cum_regret": float(cum_regret[-1]) if len(cum_regret) else math.nan,
                "candidate_id": values[-1].get("candidate_id", ""),
                "metrics_path": values[-1].get("run_dir", ""),
            }
        )
    return rows


def _plot(rows: list[dict[str, Any]], outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    grouped: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["actor"])].append(row["avg_regret"])

    for method in METHODS:
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        for actor in ACTORS:
            vals = grouped.get((method, actor), [])
            if not vals:
                continue
            mean, sem = _mean_sem(vals)
            rounds = np.arange(1, len(mean) + 1, dtype=float)
            stride = max(1, len(mean) // 180)
            xs = rounds[::stride]
            ys = mean[::stride]
            band = sem[::stride]
            style = ACTOR_STYLES[actor]
            label = ACTOR_LABELS[actor] if len(vals) == 30 else f"{ACTOR_LABELS[actor]} (n={len(vals)})"
            ax.plot(
                xs,
                ys,
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=4.0,
                markevery=max(1, len(xs) // 7),
                linewidth=1.8,
            )
            if np.nanmax(band) > 0.0:
                ax.fill_between(xs, ys - band, ys + band, color=style["color"], alpha=0.08, linewidth=0.0)
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Avg. regret per iter.", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=True, framealpha=0.94, facecolor="white", edgecolor="#DDDDDD", fontsize=9.5)
        fig.tight_layout()
        method_slug = method.lower().replace("-", "_")
        stem = f"energy_block02_with_linear_{method_slug}_avg_regret_per_round"
        for suffix in (".png", ".pdf"):
            path = outdir / f"{stem}{suffix}"
            fig.savefig(path, dpi=300 if suffix == ".png" else None)
            paths.append(path)
        plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=Path("paper_runs/paper_energy_load3_v3linear_20260505"))
    args = parser.parse_args()

    result_dir = args.campaign / "results/02_generative_ablation"
    rows = []
    rows.extend(_collect_actor(campaign=args.campaign, block_id="01_main_point_models", actor_family="gaussian_linear"))
    for actor in ("gaussian_nn", "cnf", "diffusion"):
        rows.extend(_collect_actor(campaign=args.campaign, block_id="02_generative_ablation", actor_family=actor))
    fallback = _load_old_gen_fallback(result_dir)
    for actor in ("cnf", "diffusion"):
        if not any(row["method"] == "DFHPG" and row["actor"] == actor for row in rows):
            rows.extend(row for row in fallback if row["actor"] == actor)
    _write_tables(rows, result_dir)
    paths = _plot(rows, result_dir / "figures/generative_comparison_with_linear")

    counts = defaultdict(int)
    for row in rows:
        counts[(row["method"], row["actor"])] += 1
    for method in METHODS:
        for actor in ACTORS:
            print(f"{method} {actor}: {counts[(method, actor)]}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
