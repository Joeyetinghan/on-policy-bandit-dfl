"""CLI entrypoint for running one experiment config."""

from __future__ import annotations

import argparse

from src.experiment.config import load_config
from src.experiment.engine import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    runner = ExperimentRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
