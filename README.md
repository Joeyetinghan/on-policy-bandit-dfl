# Decision-Focused On-Policy Learning for Contextual Linear Optimization with Partial Feedback

Code accompanying the paper *Decision-Focused On-Policy Learning
for Contextual Linear Optimization with Partial Feedback*
([arXiv:2606.01081](https://arxiv.org/abs/2606.01081)).

## Benchmarks

- `topk`: top-k selection
- `shortest_path`: shortest path on a directed acyclic grid
- `pricing`: pricing with a promotion budget
- `energy`: energy scheduling

## Algorithms

- `DFHPG`: hybrid update with adaptive alpha schedule
- `DFHPG-0`, `DFHPG-1`: plug-in-only and score-only ablations (alpha = 0, 1)
- `GreedyContextualBandit`, `EpsilonGreedyContextualBandit`,
  `ThompsonSamplingContextualBandit`: bandit baselines

Model families: `linear`, `nn` (Gaussian point models); `cnf`, `diffusion`
(generative models).

## Installation

Requires Python 3.11+.

```bash
conda env create -f environment.yml
conda activate dfl
```

The synthetic benchmarks (`topk`, `shortest_path`, `pricing`) need only the
conda environment. The `energy` benchmark additionally requires an unrestricted
Gurobi license and the empirical scheduling instances under
`data/Energy/SchedulingInstances/`, obtained from the MIT-licensed
[PredOpt Benchmarks](https://github.com/PredOpt/predopt-benchmarks/tree/main/Energy);
see [`data/Energy/README.md`](data/Energy/README.md) for license terms.

## Quick Start

```bash
python -m src.main --config configs/topk.yaml
```

This is the fastest smoke check and finishes in roughly a minute on one CPU.
Outputs land under `outputs/`. Substitute the other YAMLs in
[`configs/`](configs/) for the remaining benchmarks.

## Paper Reproduction

Each campaign uses selected hyperparameters from
[`configs/tuned/`](configs/tuned/); rerunning the tuning grids is optional.
Before launching a full campaign, these quick manifest checks verify that the
entrypoints, tuned configs, and output layout are wired correctly:

```bash
python scripts/paper_main.py prepare --dry-run --quick --num-seeds 1 \
  --campaign paper_main_smoke \
  --tuned-configs configs/tuned/main.yaml

python scripts/paper_energy.py prepare --dry-run --quick --num-seeds 1 \
  --campaign paper_energy_smoke \
  --tuned-configs configs/tuned/energy.yaml
```

Full reproduction runs use:

```bash
python scripts/paper_main.py all \
  --campaign paper_main \
  --tuned-configs configs/tuned/main.yaml

python scripts/paper_energy.py all \
  --campaign paper_energy \
  --tuned-configs configs/tuned/energy.yaml
```

Each campaign writes manifests, per-run outputs, and aggregated tables and
figures under `paper_runs/<campaign>/`.

## Tuning

Tuning is optional for reproducing the paper because selected configs are
committed under [`configs/tuned/`](configs/tuned/). To rerun tuning campaigns,
use the single entrypoint:

```bash
python scripts/tune.py point --problem topk --campaign topk_tune
python scripts/tune.py summarize-point \
  --outputs tuning_runs/topk_tune/outputs \
  --outdir tuning_runs/topk_tune/summary \
  --selected-out tuning_runs/topk_tune/selected_configs.yaml
python scripts/tune.py generative --problem topk --campaign topk_gen_tune
```

## Citation

If you use this code, please cite:

```bibtex
@article{benslimane2026dfonpolicy,
  title   = {Decision-Focused On-Policy Learning for Contextual Linear
             Optimization with Partial Feedback},
  author  = {Benslimane, Wyame and Ye, Tinghan and
             Van Hentenryck, Pascal and Grigas, Paul},
  journal = {arXiv preprint arXiv:2606.01081},
  year    = {2026}
}
```
