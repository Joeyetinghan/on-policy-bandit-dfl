"""Generate custom paper figures from aggregated paper-run CSVs.

Figures written here are higher-level paper figures, distinct from the generic
per-block aggregate convergence plot.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd


# Okabe-Ito / colorblind-friendly palette plus neutral gray.
PALETTE = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
    "gray": "#7F7F7F",
}

METHOD_ORDER = [
    "GreedyCB",
    "$\\epsilon$-GreedyCB",
    "TSCB",
    "DFHPG-1",
    "DFHPG fixed alpha=0",
    "DFHPG fixed alpha=0.25",
    "DFHPG fixed alpha=0.5",
    "DFHPG fixed alpha=0.75",
    "DFHPG fixed alpha=1",
    "DFHPG-0",
    "DFHPG",
]

METHOD_STYLES = {
    "GreedyCB": {"label": "GreedyCB", "color": PALETTE["gray"], "marker": "o", "linestyle": (0, (1.2, 1.2))},
    "$\\epsilon$-GreedyCB": {"label": "$\\epsilon$-GreedyCB", "color": PALETTE["vermillion"], "marker": "P", "linestyle": (0, (4.5, 1.7))},
    "TSCB": {"label": "TSCB", "color": PALETTE["reddish_purple"], "marker": "X", "linestyle": (0, (6.0, 1.6, 1.2, 1.6))},
    "DFHPG-1": {"label": "DFHPG-1", "color": PALETTE["bluish_green"], "marker": "s", "linestyle": (0, (2.2, 1.4))},
    "DFHPG": {"label": "DFHPG", "color": PALETTE["black"], "marker": "^", "linestyle": "-", "zorder": 5},
    "DFHPG-0": {"label": "DFHPG-0", "color": PALETTE["blue"], "marker": "H", "linestyle": (0, (7.0, 2.0))},
    "DFHPG fixed alpha=0": {
        "label": "DFHPG alpha=0",
        "color": PALETTE["sky_blue"],
        "marker": "o",
        "linestyle": (0, (3.0, 1.5)),
    },
    "DFHPG fixed alpha=0.25": {
        "label": "DFHPG alpha=0.25",
        "color": PALETTE["orange"],
        "marker": "v",
        "linestyle": (0, (4.0, 1.4)),
    },
    "DFHPG fixed alpha=0.5": {
        "label": "DFHPG alpha=0.5",
        "color": PALETTE["bluish_green"],
        "marker": "D",
        "linestyle": (0, (5.0, 1.5)),
    },
    "DFHPG fixed alpha=0.75": {
        "label": "DFHPG alpha=0.75",
        "color": PALETTE["yellow"],
        "marker": "P",
        "linestyle": (0, (6.0, 1.5)),
    },
    "DFHPG fixed alpha=1": {
        "label": "DFHPG alpha=1",
        "color": PALETTE["reddish_purple"],
        "marker": "X",
        "linestyle": (0, (1.2, 1.2)),
    },
}

# Bar-plot-specific colors. Line plots keep DFHPG black/solid as the visual hero,
# but on bar charts a black bar next to a saturated blue looks heavy and "ugly"
# (per user feedback). Use a softer Okabe-Ito blue+orange pair for bars.
BAR_METHOD_COLORS = {
    "DFHPG": "#E69F00",  # orange (warm accent for the hero method)
    "DFHPG-0": "#0072B2",  # blue
    "DFHPG-1": "#CC79A7",  # reddish purple
    "GreedyCB": "#7F7F7F",  # gray
    "TSCB": "#56B4E9",  # sky blue
    "$\\epsilon$-GreedyCB": "#D55E00",  # vermillion
}


def _bar_color(method: str) -> str:
    """Color used for ``method`` in bar plots (decoupled from line-plot styles)."""

    if method in BAR_METHOD_COLORS:
        return BAR_METHOD_COLORS[method]
    style = METHOD_STYLES.get(method, {})
    return style.get("color", PALETTE["gray"])


_LEGACY_DFHPG_LABEL = "Hybrid" + "Bandit"
_LEGACY_GREEDY_CB_ALGO_DIR = "Greedy" + "Scalar" + "Bandit"
_LEGACY_EPS_GREEDY_CB_ALGO_DIR = "EpsilonGreedy" + "Scalar" + "Bandit"
_LEGACY_TS_CB_ALGO_DIR = "ThompsonSampling" + "Scalar" + "Bandit"

METHOD_RENAMES = {
    _LEGACY_DFHPG_LABEL: "DFHPG",
    f"{_LEGACY_DFHPG_LABEL} fixed alpha=0": "DFHPG fixed alpha=0",
    f"{_LEGACY_DFHPG_LABEL} fixed alpha=0.25": "DFHPG fixed alpha=0.25",
    f"{_LEGACY_DFHPG_LABEL} fixed alpha=0.5": "DFHPG fixed alpha=0.5",
    f"{_LEGACY_DFHPG_LABEL} fixed alpha=0.75": "DFHPG fixed alpha=0.75",
    f"{_LEGACY_DFHPG_LABEL} fixed alpha=1": "DFHPG fixed alpha=1",
    "SurrogateOnly": "DFHPG-0",
    "ScoreOnly": "DFHPG-1",
    "EpsGreedyCB": "$\\epsilon$-GreedyCB",
    "GreedyContextualBandit": "GreedyCB",
    "EpsilonGreedyContextualBandit": "$\\epsilon$-GreedyCB",
    "ThompsonSamplingContextualBandit": "TSCB",
    _LEGACY_GREEDY_CB_ALGO_DIR: "GreedyCB",
    _LEGACY_EPS_GREEDY_CB_ALGO_DIR: "$\\epsilon$-GreedyCB",
    _LEGACY_TS_CB_ALGO_DIR: "TSCB",
}

PROBLEM_LABELS = {
    "topk": "Top-k",
    "shortest_path": "Shortest Path",
    "pricing": "Pricing",
}

MODEL_LABELS = {
    "linear": "Linear",
    "two_layer_nn": "Two-layer NN",
    "nn": "Two-layer NN",
    "cnf": "CNF",
    "diffusion": "Diffusion",
}

MODEL_FILE_LABELS = {
    "linear": "linear",
    "two_layer_nn": "nn",
    "nn": "nn",
    "cnf": "cnf",
    "diffusion": "diffusion",
}

ACTOR_FAMILY_ORDER = ["gaussian_linear", "gaussian_nn", "cnf", "diffusion"]
ACTOR_FAMILY_LABELS = {
    "gaussian_linear": "Gaussian Linear",
    "gaussian_nn": "Gaussian NN",
    "cnf": "CNF",
    "diffusion": "Diffusion",
}
ACTOR_FAMILY_STYLES = {
    "gaussian_linear": {"color": PALETTE["gray"], "marker": "o", "linestyle": (0, (1.2, 1.2))},
    "gaussian_nn": {"color": PALETTE["blue"], "marker": "s", "linestyle": (0, (6.0, 2.0))},
    "cnf": {"color": PALETTE["bluish_green"], "marker": "^", "linestyle": "-"},
    "diffusion": {"color": PALETTE["vermillion"], "marker": "D", "linestyle": (0, (3.0, 1.5))},
}

FEEDBACK_LABELS = {
    "scalar_bandit": "Pure bandit",
    "semi_bandit": "Semi bandit",
    "full_information": "Full feedback",
}

FEEDBACK_ORDER = ["scalar_bandit", "semi_bandit", "full_information"]
FEEDBACK_BAR_METHODS = {"DFHPG", "DFHPG-0"}
FEEDBACK_BAR_EXCLUDED_PROBLEMS: set[str] = set()
FEEDBACK_BANDIT_VS_SEMI_METHODS = {
    "DFHPG",
    "DFHPG-0",
    "GreedyCB",
    "$\\epsilon$-GreedyCB",
    "TSCB",
}
FEEDBACK_BANDIT_VS_SEMI_ORDER = ["scalar_bandit", "semi_bandit"]
FEEDBACK_BAR_COLORS = {
    "DFHPG": "#D55E00",
    "DFHPG-0": "#0072B2",
}

PNG_DPI = 300
AXIS_LABEL_FONTSIZE = 12
TICK_LABEL_FONTSIZE = 10
LEGEND_FONTSIZE = 9.5
HEATMAP_AXIS_LABEL_FONTSIZE = 15
HEATMAP_TICK_LABEL_FONTSIZE = 13
HEATMAP_CELL_FONTSIZE = 11
HEATMAP_COLORBAR_LABEL_FONTSIZE = 13
HEATMAP_COLORBAR_TICK_FONTSIZE = 11
DEGREE_BAR_AXIS_LABEL_FONTSIZE = 14
DEGREE_BAR_TICK_LABEL_FONTSIZE = 12
DEGREE_BAR_LEGEND_FONTSIZE = 11
DEGREE_BAR_MIN_WIDTH = 6.6
DEGREE_BAR_EDGE_COLOR = "#333333"
DEGREE_BAR_HATCHES = {
    "GreedyCB": "",
    "$\\epsilon$-GreedyCB": "//",
    "TSCB": "\\\\",
    "DFHPG-1": "xx",
    "DFHPG-0": "..",
    "DFHPG": "++",
}
LINE_WIDTH = 1.8
MARKER_SIZE = 4.0

METRIC_SPECS = {
    "avg_regret_per_round": {
        "source": "avg_regret_per_round",
        "ylabel": "Avg. regret per iter.",
        "filename": "avg_regret_per_round",
    },
    "avg_cum_expected_objective": {
        "source": "avg_cum_expected_objective",
        "ylabel": "Avg. objective per iter.",
        "filename": "avg_cum_expected_objective",
    },
}


def _save(fig, outdir: Path, stem: str) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [outdir / f"{stem}.png", outdir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=PNG_DPI if path.suffix == ".png" else None)
    plt.close(fig)
    return paths


def _read_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, usecols=usecols)
    if "method" in df.columns:
        df["method"] = df["method"].replace(METHOD_RENAMES)
    return df


def _maybe_replace_energy_alpha0_from_block01(campaign: Path, block_id: str) -> None:
    """Patch energy block-03 alpha=0 from block-01 DFHPG-0 before plotting.

    For energy linear runs, fixed alpha=0 is equivalent to DFHPG-0.  Block 01
    contains the complete 30-seed DFHPG-0 series, while block 03 may have only a
    partial dedicated fixed-alpha-0 backfill.  Keep the plot path deterministic
    by replacing block-03 alpha=0 rows with block-01 DFHPG-0 whenever this is an
    energy campaign.
    """

    if block_id != "03_alpha_diagnostics":
        return
    block01 = campaign / "results" / "01_main_point_models"
    block03 = campaign / "results" / "03_alpha_diagnostics"
    raw01 = block01 / "raw" / "traces.csv"
    raw03 = block03 / "raw" / "traces.csv"
    summary01 = block01 / "summary" / "summary.csv"
    summary03 = block03 / "summary" / "summary.csv"
    if not (raw01.exists() and raw03.exists() and summary01.exists() and summary03.exists()):
        return

    alpha0_label = "DFHPG fixed alpha=0"
    raw1 = pd.read_csv(raw01)
    if "problem" not in raw1.columns or "energy" not in set(raw1["problem"].dropna().astype(str)):
        return
    raw3 = pd.read_csv(raw03)
    alpha0_raw = raw1[
        (raw1["problem"].astype(str) == "energy")
        & (raw1["actor_family"].astype(str) == "gaussian_linear")
        & (raw1["method"].astype(str) == "DFHPG-0")
    ].copy()
    if not alpha0_raw.empty:
        alpha0_raw["method"] = alpha0_label
        if "algo_internal" in alpha0_raw.columns:
            alpha0_raw["algo_internal"] = "DFHPG"
        raw3 = raw3[raw3["method"].astype(str) != alpha0_label].copy()
        pd.concat([raw3, alpha0_raw], ignore_index=True, sort=False).to_csv(raw03, index=False)

    summary1 = pd.read_csv(summary01)
    summary3 = pd.read_csv(summary03)
    alpha0_summary = summary1[
        (summary1["problem"].astype(str) == "energy")
        & (summary1["actor_family"].astype(str) == "gaussian_linear")
        & (summary1["method"].astype(str) == "DFHPG-0")
    ].copy()
    if not alpha0_summary.empty:
        alpha0_summary["method"] = alpha0_label
        if "algo_internal" in alpha0_summary.columns:
            alpha0_summary["algo_internal"] = "DFHPG"
        summary3 = summary3[summary3["method"].astype(str) != alpha0_label].copy()
        pd.concat([summary3, alpha0_summary], ignore_index=True, sort=False).to_csv(summary03, index=False)


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _prepare_trace_data(path: Path) -> pd.DataFrame:
    columns = [
        "seed",
        "problem",
        "point_model",
        "actor_family",
        "degree",
        "round",
        "method",
        "cum_cost",
        "cum_regret",
        "relative_regret",
    ]
    try:
        df = _read_csv(path, usecols=columns)
    except ValueError:
        compact = _read_csv(path)
        required = {"problem", "actor", "seed", "round", "avg_regret_per_round"}
        if not required.issubset(compact.columns):
            raise
        df = compact.rename(columns={"actor": "actor_family"}).copy()
        df["actor_family"] = df["actor_family"].replace({"linear": "gaussian_linear", "nn": "gaussian_nn"})
        df = _coerce_numeric(df, ["seed", "round", "avg_regret_per_round"])
        if "method" not in df.columns:
            df["method"] = "DFHPG"
        if "degree" not in df.columns:
            df["degree"] = np.where(df["problem"].astype(str) == "energy", 0, 8)
        if "point_model" not in df.columns:
            df["point_model"] = "generative"
        df["degree"] = pd.to_numeric(df["degree"], errors="coerce").fillna(0)
        df["cum_regret"] = df["avg_regret_per_round"] * df["round"].clip(lower=1)
        df["relative_regret"] = np.nan
        df["avg_cum_expected_objective"] = np.nan
        return df.dropna(subset=["seed", "degree", "round"])
    df = _coerce_numeric(df, ["seed", "degree", "round", "cum_cost", "cum_regret", "relative_regret"])
    # Energy benchmark has no polynomial degree; treat NaN as a sentinel 0 so
    # the rest of the pipeline (filtering, sort, filename suffix) still works.
    df["degree"] = df["degree"].fillna(0)
    denominator = df["round"].clip(lower=1)
    df["avg_cum_expected_objective"] = df["cum_cost"] / denominator
    df["avg_regret_per_round"] = df["cum_regret"] / denominator
    return df.dropna(subset=["seed", "degree", "round"])


def _mean_ci(group: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = (
        group.groupby(["method", "round"], sort=True)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["sem"] = grouped["std"].fillna(0.0) / np.sqrt(grouped["count"].clip(lower=1))
    grouped["band"] = grouped["sem"]
    return grouped


def _methods_in_order(methods: set[str]) -> list[str]:
    ordered = [method for method in METHOD_ORDER if method in methods]
    ordered.extend(sorted(methods.difference(ordered)))
    return ordered


def _actor_families_in_order(actor_families: set[str]) -> list[str]:
    ordered = [actor for actor in ACTOR_FAMILY_ORDER if actor in actor_families]
    ordered.extend(sorted(actor_families.difference(ordered)))
    return ordered


def _plot_convergence_one(
    df: pd.DataFrame,
    *,
    problem: str,
    point_model: str,
    degree: int,
    metric: str,
    outdir: Path,
    width: float,
    height: float,
    max_round: int | None = None,
    figure_prefix: str = "convergence",
) -> list[Path]:
    spec = METRIC_SPECS[metric]
    subset = df[
        (df["problem"] == problem)
        & (df["point_model"] == point_model)
        & (df["degree"] == degree)
    ].dropna(subset=[spec["source"]])
    if max_round is not None:
        subset = subset[subset["round"] <= max_round]
    if subset.empty:
        return []

    stats = _mean_ci(subset, spec["source"])
    fig, ax = plt.subplots(figsize=(width, height))

    for method in _methods_in_order(set(stats["method"])):
        series = stats[stats["method"] == method].sort_values("round")
        if series.empty:
            continue
        style = METHOD_STYLES.get(method, {"label": method, "color": PALETTE["black"]})
        x = series["round"].to_numpy(dtype=float)
        mean = series["mean"].to_numpy(dtype=float)
        band = series["band"].to_numpy(dtype=float)
        line_zorder = int(style.get("zorder", 3))
        ax.plot(
            x,
            mean,
            label=style.get("label", method),
            color=style.get("color"),
            linestyle=style.get("linestyle", "-"),
            linewidth=LINE_WIDTH * float(style.get("linewidth_scale", 1.0)),
            marker=style.get("marker"),
            markersize=MARKER_SIZE,
            markeredgewidth=0.0,
            markevery=max(1, len(x) // 8),
            zorder=line_zorder,
        )
        if np.nanmax(band) > 0:
            ax.fill_between(x, mean - band, mean + band, color=style.get("color"), alpha=0.08, zorder=1)

    ax.set_xlabel("Iteration", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(spec["ylabel"], fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.margins(x=0.02)
    ax.legend(
        loc="best",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#DDDDDD",
        fontsize=LEGEND_FONTSIZE,
        borderpad=0.5,
        labelspacing=0.4,
        handlelength=2.8,
    )
    fig.tight_layout()

    model_slug = MODEL_FILE_LABELS.get(point_model, point_model)
    horizon_tag = f"_t{max_round}" if max_round is not None else ""
    stem = f"{figure_prefix}_{problem}_{spec['filename']}_deg{degree}_{model_slug}{horizon_tag}"
    return _save(fig, outdir, stem)


def plot_generative_horizon_tradeoff_bars(
    *,
    campaign: Path,
    width: float,
    height: float,
    block_id: str = "02_generative_ablation",
) -> list[Path]:
    result_dir = campaign / "results" / block_id
    df = _prepare_trace_data(result_dir / "raw" / "traces.csv")
    outdir = result_dir / "figures" / "generative_horizon_bars"
    written: list[Path] = []
    if df.empty or "actor_family" not in df.columns:
        return written

    df = df.copy()
    df["degree"] = df["degree"].fillna(0)
    horizon_specs = [20, 100, 500, 2000]
    settings = (
        df[["problem", "degree", "method"]]
        .drop_duplicates()
        .sort_values(["problem", "degree", "method"])
    )

    for row in settings.itertuples(index=False):
        problem = str(row.problem)
        degree = int(row.degree)
        method = str(row.method)
        sub = df[
            (df["problem"] == problem)
            & (df["degree"] == degree)
            & (df["method"] == method)
        ].dropna(subset=["avg_regret_per_round", "actor_family", "round"])
        if sub.empty:
            continue

        rounds = np.asarray(sorted(sub["round"].dropna().unique()), dtype=float)
        if len(rounds) == 0:
            continue
        selected: list[tuple[str, int]] = []
        seen_rounds: set[int] = set()
        for target_round in horizon_specs:
            target = float(target_round)
            selected_round = int(rounds[np.argmin(np.abs(rounds - target))])
            if selected_round in seen_rounds:
                continue
            seen_rounds.add(selected_round)
            selected.append((f"t={selected_round}", selected_round))

        actor_families = _actor_families_in_order(set(sub["actor_family"]))
        x = np.arange(len(selected), dtype=float)
        bar_width = min(0.19, 0.78 / max(1, len(actor_families)))
        offsets = (np.arange(len(actor_families)) - (len(actor_families) - 1) / 2.0) * bar_width

        fig, ax = plt.subplots(figsize=(width, height))
        per_round_stats: dict[int, pd.DataFrame] = {}
        for _, selected_round in selected:
            at_round = sub[sub["round"] == selected_round]
            stats = at_round.groupby("actor_family", sort=True)["avg_regret_per_round"].agg(["mean", "std", "count"]).reset_index()
            stats["sem"] = stats["std"].fillna(0.0) / np.sqrt(stats["count"].clip(lower=1))
            per_round_stats[selected_round] = stats
        for idx, actor_family in enumerate(actor_families):
            style = ACTOR_FAMILY_STYLES.get(actor_family, {"color": PALETTE["gray"]})
            values = []
            errors = []
            for _, selected_round in selected:
                stats = per_round_stats[selected_round]
                match = stats[stats["actor_family"] == actor_family]
                if match.empty:
                    values.append(np.nan)
                    errors.append(0.0)
                    continue
                values.append(float(match["mean"].iloc[0]))
                sem = float(match["sem"].iloc[0])
                errors.append(sem if math.isfinite(sem) else 0.0)
            ax.bar(
                x + offsets[idx],
                values,
                width=bar_width,
                yerr=errors,
                capsize=3,
                label=ACTOR_FAMILY_LABELS.get(actor_family, actor_family),
                color=style.get("color", PALETTE["gray"]),
                hatch=style.get("hatch", ""),
                edgecolor="white",
                linewidth=0.7,
            )

        ax.set_xlabel("Horizon", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Avg. regret per iter.", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xticks(x)
        ax.set_xticklabels([label for label, _ in selected], fontsize=TICK_LABEL_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
        ax.grid(True, axis="y", alpha=0.18, linestyle="--", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            ncol=min(4, len(actor_families)),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.14),
        )
        fig.tight_layout()
        method_slug = (
            method.lower()
            .replace("$", "")
            .replace("\\", "")
            .replace(" ", "_")
            .replace("=", "")
            .replace("-", "_")
        )
        written.extend(
            _save(fig, outdir, f"generative_horizon_tradeoff_{problem}_deg{degree}_{method_slug}")
        )
    return written


def plot_main_style_convergence(
    *,
    campaign: Path,
    block_id: str,
    metrics: list[str],
    width: float,
    height: float,
    max_round: int | None = None,
    point_model_filter: str | None = None,
    degree_filter: int | None = None,
    figure_prefix: str | None = None,
    figures_subdir: str | None = None,
) -> list[Path]:
    result_dir = campaign / "results" / block_id
    df = _prepare_trace_data(result_dir / "raw" / "traces.csv")
    if figure_prefix is None:
        figure_prefix = "alpha_ablation" if block_id == "03_alpha_diagnostics" else "convergence"
    if figures_subdir is None:
        figures_subdir = "alpha_ablation" if block_id == "03_alpha_diagnostics" else "main_style"
    outdir = result_dir / "figures" / figures_subdir
    written: list[Path] = []
    df = df.copy()
    df["degree"] = df["degree"].fillna(0)
    settings = (
        df[["problem", "point_model", "degree"]]
        .drop_duplicates()
        .sort_values(["problem", "point_model", "degree"])
    )
    if point_model_filter is not None:
        settings = settings[settings["point_model"] == point_model_filter]
    if degree_filter is not None:
        settings = settings[settings["degree"] == degree_filter]
    for row in settings.itertuples(index=False):
        for metric in metrics:
            written.extend(
                _plot_convergence_one(
                    df,
                    problem=str(row.problem),
                    point_model=str(row.point_model),
                    degree=int(row.degree),
                    metric=metric,
                    outdir=outdir,
                    width=width,
                    height=height,
                    max_round=max_round,
                    figure_prefix=figure_prefix,
                )
            )
    return written


def plot_generative_comparison_convergence(
    *,
    campaign: Path,
    metrics: list[str],
    width: float,
    height: float,
    xscale: str = "linear",
    block_id: str = "02_generative_ablation",
) -> list[Path]:
    result_dir = campaign / "results" / block_id
    df = _prepare_trace_data(result_dir / "raw" / "traces.csv")
    outdir = result_dir / "figures" / "generative_comparison"
    if xscale not in {"linear", "log"}:
        raise ValueError(f"xscale must be 'linear' or 'log'; got {xscale!r}")
    written: list[Path] = []
    df = df.copy()
    df["degree"] = df["degree"].fillna(0)
    settings = (
        df[["problem", "degree", "method"]]
        .drop_duplicates()
        .sort_values(["problem", "degree", "method"])
    )

    for row in settings.itertuples(index=False):
        problem = str(row.problem)
        degree = int(row.degree)
        method = str(row.method)
        subset = df[
            (df["problem"] == problem)
            & (df["degree"] == degree)
            & (df["method"] == method)
        ].copy()
        if subset.empty:
            continue

        for metric in metrics:
            spec = METRIC_SPECS[metric]
            metric_subset = subset.dropna(subset=[spec["source"]])
            if metric_subset.empty:
                continue
            stats = (
                metric_subset.groupby(["actor_family", "round"], sort=True)[spec["source"]]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            stats["sem"] = stats["std"].fillna(0.0) / np.sqrt(stats["count"].clip(lower=1))
            stats["band"] = stats["sem"]

            fig, ax = plt.subplots(figsize=(width, height))
            for actor_family in _actor_families_in_order(set(stats["actor_family"])):
                series = stats[stats["actor_family"] == actor_family].sort_values("round")
                if series.empty:
                    continue
                style = ACTOR_FAMILY_STYLES.get(actor_family, {"color": PALETTE["black"]})
                x = series["round"].to_numpy(dtype=float)
                mean = series["mean"].to_numpy(dtype=float)
                band = series["band"].to_numpy(dtype=float)
                ax.plot(
                    x,
                    mean,
                    label=ACTOR_FAMILY_LABELS.get(actor_family, actor_family),
                    color=style.get("color"),
                    linestyle=style.get("linestyle", "-"),
                    linewidth=LINE_WIDTH,
                    marker=style.get("marker"),
                    markersize=MARKER_SIZE,
                    markeredgewidth=0.0,
                    markevery=max(1, len(x) // 8),
                    zorder=3,
                )
                if np.nanmax(band) > 0:
                    ax.fill_between(x, mean - band, mean + band, color=style.get("color"), alpha=0.08, zorder=1)

            ax.set_xlabel("Iteration", fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_ylabel(spec["ylabel"], fontsize=AXIS_LABEL_FONTSIZE)
            ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
            ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_axisbelow(True)
            ax.margins(x=0.02)
            if xscale == "log":
                ax.set_xscale("log")
                positive = stats[stats["round"] > 0]["round"]
                if not positive.empty:
                    ax.set_xlim(left=max(1.0, float(positive.min())))
            ax.legend(
                loc="best",
                frameon=True,
                framealpha=0.92,
                facecolor="white",
                edgecolor="#DDDDDD",
                fontsize=LEGEND_FONTSIZE,
                borderpad=0.5,
                labelspacing=0.4,
                handlelength=2.8,
            )
            fig.tight_layout()
            scale_tag = "" if xscale == "linear" else f"_{xscale}x"
            stem = (
                f"generative_convergence{scale_tag}_{problem}_{spec['filename']}"
                f"_deg{degree}_{method.lower().replace(' ', '_').replace('=', '')}"
            )
            written.extend(_save(fig, outdir, stem))
    return written


def plot_generative_t15000_legacy_folder(
    *,
    campaign: Path,
    width: float,
    height: float,
    block_id: str = "02_generative_ablation",
) -> list[Path]:
    """Write the paper-facing T=15000 block-02 trajectory folder.

    The folder/name is retained for existing manuscript links, but the data are
    the canonical block-02 traces and the labels use the current terminology.
    """

    result_dir = campaign / "results" / block_id
    df = _prepare_trace_data(result_dir / "raw" / "traces.csv")
    outdir = result_dir / "figures" / "generative_comparison_t15000"
    written: list[Path] = []
    df = df.dropna(subset=["avg_regret_per_round", "actor_family", "round"]).copy()
    if df.empty:
        return written

    for problem in sorted(df["problem"].unique()):
        sub = df[df["problem"] == problem].copy()
        if sub.empty:
            continue
        stats = (
            sub.groupby(["actor_family", "round"], sort=True)["avg_regret_per_round"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        stats["sem"] = stats["std"].fillna(0.0) / np.sqrt(stats["count"].clip(lower=1))

        fig, ax = plt.subplots(figsize=(width, height))
        for actor_family in _actor_families_in_order(set(stats["actor_family"])):
            series = stats[stats["actor_family"] == actor_family].sort_values("round")
            if series.empty:
                continue
            style = ACTOR_FAMILY_STYLES.get(actor_family, {"color": PALETTE["black"]})
            x = series["round"].to_numpy(dtype=float)
            mean = series["mean"].to_numpy(dtype=float)
            sem = series["sem"].to_numpy(dtype=float)
            ax.plot(
                x,
                mean,
                label=ACTOR_FAMILY_LABELS.get(actor_family, actor_family),
                color=style.get("color"),
                linestyle=style.get("linestyle", "-"),
                linewidth=LINE_WIDTH,
                marker=style.get("marker"),
                markersize=MARKER_SIZE,
                markeredgewidth=0.0,
                markevery=max(1, len(x) // 8),
                zorder=3,
            )
            if np.nanmax(sem) > 0:
                ax.fill_between(x, mean - sem, mean + sem, color=style.get("color"), alpha=0.08, zorder=1)

        ax.set_xlabel("Iteration", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Avg. regret per iter.", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.margins(x=0.02)
        ax.legend(
            loc="best",
            frameon=True,
            framealpha=0.92,
            facecolor="white",
            edgecolor="#DDDDDD",
            fontsize=LEGEND_FONTSIZE,
            borderpad=0.5,
            labelspacing=0.4,
            handlelength=2.8,
        )
        fig.tight_layout()
        written.extend(_save(fig, outdir, f"block02_t15000_{problem}_avg_regret_per_round"))
    return written


def plot_generative_final_regret_bars(
    *,
    campaign: Path,
    width: float,
    height: float,
    block_id: str = "02_generative_ablation",
) -> list[Path]:
    result_dir = campaign / "results" / block_id
    df = _read_csv(result_dir / "summary" / "summary.csv")
    compact_summary = "mean_cum_regret" not in df.columns and "mean_final_avg_regret_per_round" in df.columns
    if compact_summary:
        df = df.rename(
            columns={
                "actor": "actor_family",
                "mean_final_avg_regret_per_round": "mean_cum_regret",
                "sem_final_avg_regret_per_round": "sem_cum_regret",
            }
        ).copy()
        df["actor_family"] = df["actor_family"].replace({"linear": "gaussian_linear", "nn": "gaussian_nn"})
        if "method" not in df.columns:
            df["method"] = "DFHPG"
        if "degree" not in df.columns:
            df["degree"] = np.where(df["problem"].astype(str) == "energy", 0, 8)
    df = _coerce_numeric(df, ["degree", "mean_cum_regret", "sem_cum_regret"])
    if compact_summary:
        value_col = "mean_cum_regret"
        sem_col = "sem_cum_regret"
    else:
        df = _add_final_avg_regret_per_iteration(
            df,
            result_dir,
            value_col="mean_cum_regret",
            sem_col="sem_cum_regret",
        )
        value_col = "final_avg_regret_per_iteration"
        sem_col = "sem_final_avg_regret_per_iteration"
    outdir = result_dir / "figures" / "generative_comparison"
    written: list[Path] = []
    if df.empty:
        return written

    df = df.copy()
    df["degree"] = df["degree"].fillna(0)
    settings = (
        df[["problem", "degree"]]
        .drop_duplicates()
        .sort_values(["problem", "degree"])
    )
    for row in settings.itertuples(index=False):
        problem = str(row.problem)
        degree = int(row.degree)
        sub = df[(df["problem"] == problem) & (df["degree"] == degree)].copy()
        if sub.empty:
            continue
        actor_families = _actor_families_in_order(set(sub["actor_family"]))
        methods = _methods_in_order(set(sub["method"]))
        x = np.arange(len(actor_families), dtype=float)
        bar_width = min(0.26, 0.78 / max(1, len(methods)))
        offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * bar_width

        fig, ax = plt.subplots(figsize=(width, height))
        for method_idx, method in enumerate(methods):
            method_style = METHOD_STYLES.get(method, {"label": method, "color": PALETTE["black"]})
            values = []
            errors = []
            for actor_family in actor_families:
                match = sub[(sub["actor_family"] == actor_family) & (sub["method"] == method)]
                if match.empty:
                    values.append(np.nan)
                    errors.append(0.0)
                    continue
                values.append(float(match[value_col].iloc[0]))
                sem = float(match[sem_col].iloc[0])
                errors.append(sem if math.isfinite(sem) else 0.0)
            ax.bar(
                x + offsets[method_idx],
                values,
                width=bar_width,
                yerr=errors,
                capsize=3,
                label=method_style.get("label", method),
                color=method_style.get("color"),
                edgecolor="white",
                linewidth=0.6,
            )

        ax.set_xlabel("Distribution mode", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Final avg. regret per iter.", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [ACTOR_FAMILY_LABELS.get(actor_family, actor_family) for actor_family in actor_families],
            fontsize=TICK_LABEL_FONTSIZE,
            rotation=15,
            ha="right",
        )
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
        ax.grid(True, axis="y", alpha=0.18, linestyle="--", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, ncol=min(2, len(methods)))
        fig.tight_layout()
        written.extend(_save(fig, outdir, f"generative_final_avg_regret_per_iteration_{problem}_deg{degree}"))
    return written


def _mean_sem_ci(df: pd.DataFrame, metric: str, group_cols: list[str]) -> pd.DataFrame:
    out = df.groupby(group_cols, sort=True)[metric].agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"].fillna(0.0) / np.sqrt(out["count"].clip(lower=1))
    out["ci95"] = out["sem"]
    return out


def _problem_horizons(result_dir: Path) -> dict[str, float]:
    trace_path = result_dir / "raw" / "traces.csv"
    if not trace_path.exists():
        return {}
    traces = _read_csv(trace_path, usecols=["problem", "round"])
    traces = _coerce_numeric(traces, ["round"])
    horizons = traces.groupby("problem")["round"].max().dropna()
    return {str(problem): float(rounds) for problem, rounds in horizons.items() if float(rounds) > 0.0}


def _add_final_avg_regret_per_iteration(
    df: pd.DataFrame,
    result_dir: Path,
    *,
    value_col: str,
    sem_col: str | None = None,
) -> pd.DataFrame:
    out = df.copy()
    horizons = _problem_horizons(result_dir)
    horizon = out["problem"].astype(str).map(horizons).fillna(1.0).clip(lower=1.0)
    out["final_avg_regret_per_iteration"] = out[value_col] / horizon
    if sem_col is not None and sem_col in out.columns:
        out["sem_final_avg_regret_per_iteration"] = out[sem_col] / horizon
    return out


def plot_feedback_bars(*, campaign: Path, width: float, height: float, log_scale: bool = False) -> list[Path]:
    block_id = "04_feedback_ablation"
    result_dir = campaign / "results" / block_id
    df = _read_csv(result_dir / "raw" / "per_seed_final.csv")
    df = _coerce_numeric(df, ["cum_regret"])
    df = df[df["method"].isin(FEEDBACK_BAR_METHODS)].copy()
    outdir = result_dir / "figures" / "feedback_bars"
    written: list[Path] = []
    if df.empty:
        return written

    for metric, ylabel, suffix in [
        ("cum_regret", "Final cumulative regret", "cum_regret"),
    ]:
        scale_suffix = "_log" if log_scale else ""
        scale_label = " (log scale)" if log_scale else ""
        stats = _mean_sem_ci(df, metric, ["problem", "feedback", "method"])
        for problem in sorted(stats["problem"].unique()):
            if problem in FEEDBACK_BAR_EXCLUDED_PROBLEMS:
                continue
            sub = stats[stats["problem"] == problem]
            methods = _methods_in_order(set(sub["method"]))
            feedbacks = [fb for fb in FEEDBACK_ORDER if fb in set(sub["feedback"])]
            x = np.arange(len(feedbacks), dtype=float)
            bar_width = min(0.18, 0.78 / max(1, len(methods)))

            fig, ax = plt.subplots(figsize=(width, height))
            offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * bar_width
            positive_values: list[float] = []
            for method_idx, method in enumerate(methods):
                style = METHOD_STYLES.get(method, {"label": method, "color": PALETTE["black"]})
                values = []
                errors = []
                for fb in feedbacks:
                    row = sub[(sub["feedback"] == fb) & (sub["method"] == method)]
                    if row.empty:
                        values.append(np.nan)
                        errors.append(0.0)
                    else:
                        values.append(float(row["mean"].iloc[0]))
                        errors.append(float(row["ci95"].iloc[0]))
                positive_values.extend([value for value in values if np.isfinite(value) and value > 0.0])
                ax.bar(
                    x + offsets[method_idx],
                    values,
                    width=bar_width,
                    yerr=errors,
                    capsize=3,
                    label=style.get("label", method),
                    color=FEEDBACK_BAR_COLORS.get(method, _bar_color(method)),
                    edgecolor="white",
                    linewidth=0.6,
                )

            if log_scale:
                ax.set_yscale("log")
                if positive_values:
                    ax.set_ylim(bottom=max(min(positive_values) * 0.5, 1.0e-12))
            else:
                _use_compact_scientific_axis(ax, axis="y")
            ax.set_xlabel("Feedback mode", fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_ylabel(f"{ylabel}{scale_label}", fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_xticks(x)
            ax.set_xticklabels([FEEDBACK_LABELS.get(fb, fb) for fb in feedbacks], fontsize=TICK_LABEL_FONTSIZE)
            ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
            ax.grid(True, axis="y", alpha=0.18, linestyle="--", linewidth=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, ncol=2)
            fig.tight_layout()
            written.extend(_save(fig, outdir, f"feedback_final_{suffix}{scale_suffix}_{problem}"))
    return written


def _compact_regret(value: float) -> str:
    if not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.2e}"


def _latex_escape(value: object) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _write_feedback_bandit_vs_semi_table(result_dir: Path, stats: pd.DataFrame) -> None:
    table_dir = result_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    stats = stats.copy()
    stats["mean_cum_regret"] = stats["mean"]
    stats["sem_cum_regret"] = stats["sem"]
    stats[["problem", "feedback", "method", "count", "mean_cum_regret", "sem_cum_regret"]].to_csv(
        table_dir / "04_feedback_bandit_vs_semibandit_summary.csv",
        index=False,
    )

    rows = stats.sort_values(["problem", "feedback", "method"])
    lines = [
        "\\begin{tabular}{llllr}",
        "Problem & Feedback & Method & Final cumulative regret & N \\\\",
        "\\hline",
    ]
    for row in rows.itertuples(index=False):
        mean_text = _compact_regret(float(row.mean))
        sem_text = _compact_regret(float(row.sem))
        method_label = str(METHOD_STYLES.get(str(row.method), {}).get("label", row.method))
        method_label = method_label.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
        values = [
            _latex_escape(row.problem),
            _latex_escape(FEEDBACK_LABELS.get(str(row.feedback), str(row.feedback))),
            method_label,
            f"{mean_text} $\\pm$ {sem_text}",
            _latex_escape(int(row.count)),
        ]
        lines.append(" & ".join(values) + " \\\\")
    lines.append("\\end{tabular}")
    (table_dir / "04_feedback_bandit_vs_semibandit_summary.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def plot_feedback_bandit_vs_semibandit_bars(
    *,
    campaign: Path,
    width: float,
    height: float,
    log_scale: bool = False,
) -> list[Path]:
    block_id = "04_feedback_ablation"
    result_dir = campaign / "results" / block_id
    df = _read_csv(result_dir / "raw" / "per_seed_final.csv")
    df = _coerce_numeric(df, ["cum_regret"])
    df = df[
        df["method"].isin(FEEDBACK_BANDIT_VS_SEMI_METHODS)
        & df["feedback"].isin(FEEDBACK_BANDIT_VS_SEMI_ORDER)
    ].copy()
    outdir = result_dir / "figures" / "feedback_bandit_vs_semibandit"
    written: list[Path] = []
    if df.empty:
        return written

    stats = _mean_sem_ci(df, "cum_regret", ["problem", "feedback", "method"])
    if not log_scale:
        _write_feedback_bandit_vs_semi_table(result_dir, stats)

    scale_suffix = "_log" if log_scale else ""
    scale_label = " (log scale)" if log_scale else ""
    for problem in sorted(stats["problem"].unique()):
        sub = stats[stats["problem"] == problem]
        methods = _methods_in_order(set(sub["method"]))
        feedbacks = [fb for fb in FEEDBACK_BANDIT_VS_SEMI_ORDER if fb in set(sub["feedback"])]
        x = np.arange(len(feedbacks), dtype=float)
        bar_width = min(0.14, 0.78 / max(1, len(methods)))

        fig, ax = plt.subplots(figsize=(max(width, 5.8), height))
        offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * bar_width
        positive_values: list[float] = []
        for method_idx, method in enumerate(methods):
            style = METHOD_STYLES.get(method, {"label": method, "color": PALETTE["black"]})
            values = []
            errors = []
            for fb in feedbacks:
                row = sub[(sub["feedback"] == fb) & (sub["method"] == method)]
                if row.empty:
                    values.append(np.nan)
                    errors.append(0.0)
                else:
                    values.append(float(row["mean"].iloc[0]))
                    errors.append(float(row["ci95"].iloc[0]))
            positive_values.extend([value for value in values if np.isfinite(value) and value > 0.0])
            ax.bar(
                x + offsets[method_idx],
                values,
                width=bar_width,
                yerr=errors,
                capsize=2.5,
                label=style.get("label", method),
                color=_bar_color(method),
                edgecolor="white",
                linewidth=0.6,
            )

        if log_scale:
            ax.set_yscale("log")
            if positive_values:
                ax.set_ylim(bottom=max(min(positive_values) * 0.5, 1.0e-12))
        else:
            _use_compact_scientific_axis(ax, axis="y")
        ax.set_xlabel("Feedback mode", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(f"Final cumulative regret{scale_label}", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xticks(x)
        ax.set_xticklabels([FEEDBACK_LABELS.get(fb, fb) for fb in feedbacks], fontsize=TICK_LABEL_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
        ax.grid(True, axis="y", alpha=0.18, linestyle="--", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            ncol=min(3, max(1, len(methods))),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            borderaxespad=0.0,
        )
        fig.subplots_adjust(left=0.16, right=0.985, bottom=0.16, top=0.76)
        written.extend(_save(fig, outdir, f"feedback_bandit_vs_semibandit_cum_regret{scale_suffix}_{problem}"))
    return written


def plot_degree_final_regret_bars(
    *,
    campaign: Path,
    width: float,
    height: float,
    log_scale: bool = False,
) -> list[Path]:
    block_id = "01_main_point_models"
    result_dir = campaign / "results" / block_id
    df = _read_csv(result_dir / "summary" / "summary.csv")
    df = _coerce_numeric(df, ["degree", "mean_cum_regret", "sem_cum_regret"])
    df = _add_final_avg_regret_per_iteration(
        df,
        result_dir,
        value_col="mean_cum_regret",
        sem_col="sem_cum_regret",
    )
    outdir = result_dir / "figures" / "degree_bars"
    written: list[Path] = []

    settings = (
        df[["problem", "point_model"]]
        .drop_duplicates()
        .sort_values(["problem", "point_model"])
    )
    for setting in settings.itertuples(index=False):
        problem = str(setting.problem)
        point_model = str(setting.point_model)
        sub = df[(df["problem"] == problem) & (df["point_model"] == point_model)].copy()
        if sub.empty:
            continue

        degrees = sorted(int(deg) for deg in sub["degree"].dropna().unique())
        if len(degrees) < 2:
            continue
        methods = _methods_in_order(set(sub["method"]))
        x = np.arange(len(degrees), dtype=float)
        bar_width = min(0.16, 0.82 / max(1, len(methods)))
        offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * bar_width

        fig, ax = plt.subplots(figsize=(max(width, DEGREE_BAR_MIN_WIDTH), height))
        for method_idx, method in enumerate(methods):
            style = METHOD_STYLES.get(method, {"label": method, "color": PALETTE["black"]})
            values = []
            errors = []
            for degree in degrees:
                row = sub[(sub["degree"] == degree) & (sub["method"] == method)]
                if row.empty:
                    values.append(np.nan)
                    errors.append(0.0)
                    continue
                values.append(float(row["final_avg_regret_per_iteration"].iloc[0]))
                sem = float(row["sem_final_avg_regret_per_iteration"].iloc[0])
                errors.append(sem if math.isfinite(sem) else 0.0)
            ax.bar(
                x + offsets[method_idx],
                values,
                width=bar_width,
                yerr=errors,
                capsize=2.5,
                label=style.get("label", method),
                color=_bar_color(method),
                edgecolor=DEGREE_BAR_EDGE_COLOR,
                hatch=DEGREE_BAR_HATCHES.get(method, ""),
                linewidth=0.45,
            )

        suffix = "_log" if log_scale else ""
        scale_label = " (log scale)" if log_scale else ""
        ax.set_xlabel("Degree", fontsize=DEGREE_BAR_AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(f"Final avg. regret per iter.{scale_label}", fontsize=DEGREE_BAR_AXIS_LABEL_FONTSIZE)
        if log_scale:
            ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(degree) for degree in degrees], fontsize=DEGREE_BAR_TICK_LABEL_FONTSIZE)
        ax.tick_params(axis="y", labelsize=DEGREE_BAR_TICK_LABEL_FONTSIZE)
        ax.grid(True, axis="y", alpha=0.18, linestyle="--", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        legend_cols = 3 if len(methods) > 4 else min(2, max(1, len(methods)))
        ax.legend(
            frameon=False,
            fontsize=DEGREE_BAR_LEGEND_FONTSIZE,
            ncol=legend_cols,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            borderaxespad=0.0,
        )
        fig.subplots_adjust(left=0.16, right=0.985, bottom=0.17, top=0.74)

        model_slug = MODEL_FILE_LABELS.get(point_model, point_model)
        written.extend(_save(fig, outdir, f"degree_final_avg_regret_per_iteration{suffix}_{problem}_{model_slug}"))
    return written


def _loss_order(df: pd.DataFrame, problem: str, value_col: str) -> list[str]:
    sub = df[df["problem"] == problem].copy()
    means = sub.groupby("loss")[value_col].mean().sort_values()
    return list(means.index)


def _heatmap_value_label(value: float) -> str:
    return f"{float(value):.2e}"


def _use_compact_scientific_axis(ax: plt.Axes, *, axis: str) -> None:
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    formatter.set_useOffset(False)
    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    elif axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        raise ValueError(f"Unsupported axis: {axis}")


def plot_loss_heatmaps(*, campaign: Path, width: float, height: float) -> list[Path]:
    block_id = "05_surrogate_loss_grid"
    result_dir = campaign / "results" / block_id
    df = _read_csv(result_dir / "summary" / "summary.csv")
    df = _coerce_numeric(df, ["mean_cum_regret"])
    value_col = "mean_cum_regret"
    outdir = result_dir / "figures" / "loss_heatmaps"
    written: list[Path] = []

    for problem in sorted(df["problem"].unique()):
        sub_problem = df[df["problem"] == problem].copy()
        methods = [m for m in ["DFHPG", "DFHPG-0"] if m in set(sub_problem["method"])]
        if not methods:
            methods = sorted(sub_problem["method"].unique())
        models = [m for m in ["linear", "two_layer_nn"] if m in set(sub_problem["point_model"])]
        if not models:
            models = sorted(sub_problem["point_model"].unique())

        # Sort losses by mean DFHPG performance on the first model column
        # (so the cleanest cell ordering is "best on DFHPG → top").
        ref_method = "DFHPG" if "DFHPG" in methods else methods[0]
        ref_model = models[0]
        ref_rank = (
            sub_problem[(sub_problem["method"] == ref_method) & (sub_problem["point_model"] == ref_model)]
            .groupby("loss")[value_col]
            .mean()
            .sort_values()
        )
        losses = list(ref_rank.index)
        # Append any losses missing from the ref slice at the bottom.
        all_losses = sorted(sub_problem["loss"].dropna().unique())
        for extra in all_losses:
            if extra not in losses:
                losses.append(extra)

        # Build a single matrix: rows = losses (sorted by DFHPG), columns = (method, model)
        # combinations. For the common single-model case (energy / linear-only block-05)
        # this yields a clean 2-column heatmap (DFHPG | DFHPG-0) with no model x-axis label.
        col_specs: list[tuple[str, str]] = []
        for method in methods:
            for model in models:
                col_specs.append((method, model))
        suppress_model_label = len(models) == 1

        matrix = np.full((len(losses), len(col_specs)), np.nan)
        for j, (method, model) in enumerate(col_specs):
            pivot = (
                sub_problem[
                    (sub_problem["method"] == method)
                    & (sub_problem["point_model"] == model)
                ]
                .pivot_table(index="loss", values=value_col, aggfunc="mean")
            )
            for i, loss in enumerate(losses):
                if loss in pivot.index:
                    matrix[i, j] = float(pivot.loc[loss, value_col])

        vmin = float(np.nanmin(matrix))
        vmax = float(np.nanmax(matrix))
        cmap = plt.get_cmap("viridis_r")

        cell_h = 0.46
        fig_height = max(height, cell_h * len(losses) + 2.6)
        fig_width = max(width, 1.45 * len(col_specs) + 4.0)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(col_specs)))
        if suppress_model_label:
            xticklabels = [
                METHOD_STYLES.get(method, {}).get("label", method)
                for method, _ in col_specs
            ]
        else:
            xticklabels = [
                f"{METHOD_STYLES.get(method, {}).get('label', method)}\n{MODEL_LABELS.get(model, model)}"
                for method, model in col_specs
            ]
        ax.set_xticklabels(xticklabels, rotation=0, fontsize=HEATMAP_TICK_LABEL_FONTSIZE)
        ax.set_yticks(np.arange(len(losses)))
        ax.set_yticklabels(losses, fontsize=HEATMAP_TICK_LABEL_FONTSIZE)
        ax.set_xlabel("Method", fontsize=HEATMAP_AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Surrogate loss", fontsize=HEATMAP_AXIS_LABEL_FONTSIZE)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if math.isfinite(matrix[i, j]):
                    denom = max(vmax - vmin, 1e-12)
                    rgba = cmap((matrix[i, j] - vmin) / denom)
                    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    text_color = "black" if luminance > 0.58 else "white"
                    ax.text(
                        j,
                        i,
                        _heatmap_value_label(matrix[i, j]),
                        ha="center",
                        va="center",
                        fontsize=HEATMAP_CELL_FONTSIZE,
                        color=text_color,
                    )

        cbar_formatter = ticker.ScalarFormatter(useMathText=True)
        cbar_formatter.set_powerlimits((0, 0))
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, format=cbar_formatter)
        cbar.set_label("Final cumulative regret (lower is better)", fontsize=HEATMAP_COLORBAR_LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=HEATMAP_COLORBAR_TICK_FONTSIZE)
        fig.tight_layout()
        written.extend(_save(fig, outdir, f"surrogate_loss_heatmap_{problem}"))
    return written


def plot_loss_final_regret_bars(*, campaign: Path, width: float, height: float) -> list[Path]:
    block_id = "05_surrogate_loss_grid"
    result_dir = campaign / "results" / block_id
    df = _read_csv(result_dir / "summary" / "summary.csv")
    df = _coerce_numeric(df, ["mean_cum_regret", "sem_cum_regret"])
    value_col = "mean_cum_regret"
    sem_col = "sem_cum_regret"
    outdir = result_dir / "figures" / "loss_bars"
    written: list[Path] = []

    for problem in sorted(df["problem"].unique()):
        sub_problem = df[df["problem"] == problem].copy()
        methods = [m for m in ["DFHPG", "DFHPG-0"] if m in set(sub_problem["method"])]
        if not methods:
            methods = sorted(sub_problem["method"].unique())
        losses = _loss_order(df, problem, value_col)
        if not losses or not methods:
            continue

        y = np.arange(len(losses), dtype=float)
        bar_height = min(0.34, 0.78 / max(1, len(methods)))
        offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * bar_height
        fig_height = max(height, 0.34 * len(losses) + 1.8)
        fig, ax = plt.subplots(figsize=(width, fig_height))

        for method_idx, method in enumerate(methods):
            style = METHOD_STYLES.get(method, {"label": method, "color": PALETTE["black"]})
            values = []
            errors = []
            for loss in losses:
                row = sub_problem[(sub_problem["loss"] == loss) & (sub_problem["method"] == method)]
                if row.empty:
                    values.append(np.nan)
                    errors.append(0.0)
                    continue
                values.append(float(row[value_col].iloc[0]))
                sem = float(row[sem_col].iloc[0]) if sem_col in row else 0.0
                errors.append(sem if math.isfinite(sem) else 0.0)
            ax.barh(
                y + offsets[method_idx],
                values,
                height=bar_height,
                xerr=errors,
                capsize=2.5,
                label=style.get("label", method),
                color=_bar_color(method),
                edgecolor="white",
                linewidth=0.6,
            )

        _use_compact_scientific_axis(ax, axis="x")
        ax.set_xlabel("Final cumulative regret", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Surrogate loss", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_yticks(y)
        ax.set_yticklabels(losses, fontsize=TICK_LABEL_FONTSIZE)
        ax.tick_params(axis="x", labelsize=TICK_LABEL_FONTSIZE)
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.18, linestyle="--", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            ncol=len(methods),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.06),
        )
        fig.tight_layout()
        written.extend(_save(fig, outdir, f"surrogate_loss_final_cum_regret_{problem}"))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=Path("paper_runs/paper_main_deg6_noise_20260502"))
    parser.add_argument(
        "--blocks",
        default="01_main_point_models,02_generative_ablation,03_alpha_diagnostics,04_feedback_ablation,05_surrogate_loss_grid",
        help="Comma-separated blocks to plot.",
    )
    parser.add_argument("--metrics", default="avg_regret_per_round")
    parser.add_argument("--width", type=float, default=5.2)
    parser.add_argument("--height", type=float, default=3.8)
    args = parser.parse_args()

    blocks = [block.strip() for block in args.blocks.split(",") if block.strip()]
    metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
    unknown_metrics = sorted(set(metrics).difference(METRIC_SPECS))
    if unknown_metrics:
        raise ValueError(f"Unknown metric(s): {unknown_metrics}; choices are {sorted(METRIC_SPECS)}")

    written: list[Path] = []
    for block_id in blocks:
        _maybe_replace_energy_alpha0_from_block01(args.campaign, block_id)
        if block_id in {
            "01_main_point_models",
            "03_alpha_diagnostics",
            "energy_main",
        }:
            written.extend(
                plot_main_style_convergence(
                    campaign=args.campaign,
                    block_id=block_id,
                    metrics=metrics,
                    width=args.width,
                    height=args.height,
                )
            )
            if block_id == "01_main_point_models":
                written.extend(
                    plot_main_style_convergence(
                        campaign=args.campaign,
                        block_id=block_id,
                        metrics=metrics,
                        width=args.width,
                        height=args.height,
                        max_round=600,
                        point_model_filter="linear",
                        degree_filter=8,
                    )
                )
                written.extend(plot_degree_final_regret_bars(campaign=args.campaign, width=args.width, height=args.height))
                written.extend(
                    plot_degree_final_regret_bars(
                        campaign=args.campaign,
                        width=args.width,
                        height=args.height,
                        log_scale=True,
                    )
                )
        elif block_id in {"02_generative_ablation", "energy_distribution_family"}:
            written.extend(
                plot_generative_comparison_convergence(
                    campaign=args.campaign,
                    metrics=metrics,
                    width=args.width,
                    height=args.height,
                    block_id=block_id,
                )
            )
            written.extend(
                plot_generative_comparison_convergence(
                    campaign=args.campaign,
                    metrics=metrics,
                    width=args.width,
                    height=args.height,
                    xscale="log",
                    block_id=block_id,
                )
            )
            written.extend(
                plot_generative_final_regret_bars(
                    campaign=args.campaign, width=args.width, height=args.height, block_id=block_id
                )
            )
            if block_id == "02_generative_ablation":
                written.extend(
                    plot_generative_t15000_legacy_folder(
                        campaign=args.campaign,
                        width=args.width,
                        height=args.height,
                        block_id=block_id,
                    )
                )
            written.extend(
                plot_generative_horizon_tradeoff_bars(
                    campaign=args.campaign, width=args.width, height=args.height, block_id=block_id
                )
            )
        elif block_id == "04_feedback_ablation":
            written.extend(plot_feedback_bars(campaign=args.campaign, width=args.width, height=args.height))
            written.extend(plot_feedback_bars(campaign=args.campaign, width=args.width, height=args.height, log_scale=True))
            written.extend(
                plot_feedback_bandit_vs_semibandit_bars(campaign=args.campaign, width=args.width, height=args.height)
            )
            written.extend(
                plot_feedback_bandit_vs_semibandit_bars(
                    campaign=args.campaign,
                    width=args.width,
                    height=args.height,
                    log_scale=True,
                )
            )
        elif block_id == "05_surrogate_loss_grid":
            written.extend(plot_loss_final_regret_bars(campaign=args.campaign, width=11.5, height=8.4))
            written.extend(plot_loss_heatmaps(campaign=args.campaign, width=8.5, height=7.6))
        else:
            raise ValueError(f"Unsupported block for custom plotting: {block_id}")

    print(f"Wrote {len(written)} files")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
