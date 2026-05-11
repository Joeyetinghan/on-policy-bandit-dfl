"""Shared IO helpers for experiment scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


class _NoAliasSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
SWEEPS_DIR = CONFIGS_DIR / "sweeps"
PAPER_DIR = CONFIGS_DIR / "paper"
GENERATED_CONFIG_DIR = CONFIGS_DIR / "generated"


SEED_POOL = [
    42,
    123,
    456,
    789,
    1011,
    1213,
    1415,
    1617,
    1819,
    2021,
    2223,
    2425,
    2627,
    2829,
    3031,
    3233,
    3435,
    3637,
    3839,
    4041,
    4243,
    4445,
    4647,
    4849,
    5051,
    5253,
    5455,
    5657,
    5859,
    6061,
    6263,
    6465,
    6667,
    6869,
    7071,
    7273,
    7475,
    7677,
    7879,
    8081,
    8283,
    8485,
    8687,
    8889,
    9091,
    9293,
    9495,
    9697,
    9899,
    10103,
]


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.dump(payload, Dumper=_NoAliasSafeDumper, sort_keys=False), encoding="utf-8")


def seed_list(num_seeds: int, *, start_index: int = 0) -> list[int]:
    if num_seeds < 1:
        raise ValueError("num_seeds must be positive")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    end_index = start_index + num_seeds
    if end_index > len(SEED_POOL):
        raise ValueError(
            f"Requested seeds[{start_index}:{end_index}] but only {len(SEED_POOL)} seeds are predefined"
        )
    return SEED_POOL[start_index:end_index]


def loss_mix_overrides(loss_name: str, *, weight_key: str) -> dict[str, float]:
    compact = str(loss_name).strip().lower().replace("_", "").replace("-", "").replace(" ", "").replace("+", "")
    if compact in {"weightedmsespoplus", "msespoplus", "mixedmsespoplus", "hybridmsespoplus"}:
        return {weight_key: 0.5}
    return {}
