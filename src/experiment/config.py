"""Configuration helpers for experiments."""

from __future__ import annotations

import os
import re
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config from path."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def safe_name(text: Any) -> str:
    """Normalize arbitrary text into a filesystem-safe token."""
    name = str(text)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return name or "run"


def resolve_output_dir(config: Dict[str, Any]) -> str:
    """Resolve output directory from config fields."""
    if config.get("output_dir"):
        return str(config["output_dir"])

    output_root = str(config.get("output_root", "outputs"))
    seed = config.get("seed", "na")
    if config.get("experiment_name"):
        base_name = safe_name(config["experiment_name"])
    elif config.get("output_name"):
        base_name = safe_name(config["output_name"])
    else:
        base_name = safe_name(
            f"{config.get('benchmark', 'run')}_T{config.get('T', 'na')}_{config.get('model_type', 'model')}"
        )
    return os.path.join(output_root, f"{base_name}_seed{seed}")
