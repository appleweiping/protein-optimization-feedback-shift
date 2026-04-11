# protein-optimization-feedback-shift

Research repository for the version 2 paper project on uncertainty-aware closed-loop protein sequence optimization under feedback shift.

## Project Layout

The active research code lives under:

- `protein_bo_conformal/`

That subdirectory contains the executable project, configs, experiments, tests, and benchmark hooks. The repository root mainly serves as the GitHub-facing wrapper for the paper project and local planning materials.

## Current Stage

The implementation is currently at the end of `Day 10`.

- `Day 1`: reproducible execution shell
- `Day 2`: real data environment layer
- `Day 3`: oracle consistency and split validation
- `Day 4`: shared representation layer and cache infrastructure
- `Day 5`: surrogate layer and deep ensemble baseline
- `Day 6`: acquisition layer and prediction-to-selection decision path
- `Day 7`: executable closed-loop runner and multi-round optimization traces
- `Day 8`: standardized baseline evaluation suite
- `Day 9`: full evaluation package with tables and stage-wise diagnostics
- `Day 10`: failure-analysis package with shift diagnostics and mechanism notes

## What Works Now

Inside `protein_bo_conformal/`, the project can:

- load real `FLIP`, `FLIP2`, and local `ProteinGym` benchmark files
- standardize them into a shared in-memory dataset format
- construct shift-aware splits and immutable oracles
- encode sequences with one-hot and real ESM backends
- train a deep-ensemble surrogate baseline
- run random, greedy, UCB, EI, and conformal-UCB acquisition rules
- execute real multi-round closed-loop optimization
- aggregate baseline metrics, plots, CSV/LaTeX tables, and analysis notes
- diagnose failure modes through shift metrics, sigma-vs-error analysis, and embedding-drift reports

## How To Run

From the repository root:

```powershell
cd protein_bo_conformal
py -3.12 main.py
```

Useful entrypoints:

```powershell
cd protein_bo_conformal
py -3.12 experiments\data_sanity_check.py
py -3.12 experiments\representation_sanity_check.py
py -3.12 experiments\surrogate_sanity_check.py
py -3.12 experiments\acquisition_sanity_check.py
py -3.12 experiments\baseline_eval.py --config config\experiment\day9_evaluation_package.yaml --name day9-evaluation-package
py -3.12 experiments\failure_analysis.py --config config\experiment\day10_failure_analysis.yaml --name day10-failure-analysis
py -3.12 -m unittest discover -s tests -p "test_*.py"
```

## Notes

- The detailed project README is in `protein_bo_conformal/README.md`.
- Raw benchmark data stays local and is not pushed to GitHub.
- Processed caches and experiment outputs also stay local.
- The local `paper/` directory is intentionally excluded from GitHub pushes.
- `version 1/`, `version 2/`, and `version 2.zip` are local research materials, not part of the GitHub code payload.
