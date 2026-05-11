# Scripts

- `paper_main.py`: reproduce the main synthetic paper campaign.
- `paper_energy.py`: reproduce the energy paper campaign.

Paper campaign helpers:

- `paper/`: shared paper manifest, execution, aggregation, and plotting support.

Tuning utilities:

- `tune.py`: single per-problem tuning-manifest entrypoint for point-model and generative campaigns; `summarize-point` also writes selected tuned-config YAMLs.
- `tuning/`: tuning-grid builders and summarizers.
- `utils/`: generic manifest runner and shared utility builders.
- `lib/`: shared IO helpers.
