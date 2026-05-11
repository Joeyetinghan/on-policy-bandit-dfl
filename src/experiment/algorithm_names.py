"""Shared algorithm names and legacy artifact lookup helpers."""

from __future__ import annotations

from pathlib import Path


DFHPG_ALGO_NAME = "DFHPG"
_LEGACY_DFHPG_ALGO_NAME = "Hybrid" + "Actor" + "Critic" + "Bandit"
DFHPG_METRICS_DIR_NAMES = (DFHPG_ALGO_NAME, _LEGACY_DFHPG_ALGO_NAME)


def dfhpg_metrics_path(run_dir: str | Path) -> Path:
    """Return the preferred DFHPG metrics path, falling back to legacy artifacts."""
    root = Path(run_dir)
    for algo_name in DFHPG_METRICS_DIR_NAMES:
        path = root / algo_name / "metrics.json"
        if path.exists():
            return path
    return root / DFHPG_ALGO_NAME / "metrics.json"


def iter_dfhpg_metrics_paths(output_root: str | Path):
    """Yield DFHPG metrics paths from current and legacy output directories."""
    root = Path(output_root)
    seen: set[Path] = set()
    for algo_name in DFHPG_METRICS_DIR_NAMES:
        for path in root.glob(f"*/{algo_name}/metrics.json"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path
