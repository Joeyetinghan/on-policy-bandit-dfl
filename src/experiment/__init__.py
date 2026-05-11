"""Experiment execution utilities."""

from .config import load_config, resolve_output_dir

__all__ = ["ExperimentRunner", "load_config", "resolve_output_dir"]


def __getattr__(name: str):
    if name == "ExperimentRunner":
        from .engine import ExperimentRunner

        return ExperimentRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
