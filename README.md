# protein-optimization-feedback-shift

Research repository for the version 2 paper project on uncertainty-aware closed-loop protein sequence optimization under feedback shift.

## Project Layout

The active research code currently lives under:

- `protein_bo_conformal/`

That subdirectory contains the executable project, configs, experiments, tests, and local benchmark hooks. The repository root is mainly a wrapper so GitHub can host the paper project, local notes, and versioned planning materials together.

## Current Stage

The implementation is currently at the end of `Day 6`.

- `Day 1`: reproducible execution shell
- `Day 2`: real data environment layer
- `Day 3`: oracle consistency and split validation
- `Day 4`: shared representation layer and cache infrastructure
- `Day 5`: surrogate layer and deep ensemble baseline
- `Day 6`: acquisition layer and prediction-to-selection decision path

## What Works Now

Inside `protein_bo_conformal/`, the project can:

- load real `FLIP`, `FLIP2`, and local `ProteinGym` benchmark files
- standardize them into a shared in-memory dataset format
- construct shift-aware splits and immutable oracles
- encode sequences with one-hot and real ESM backends
- train a deep-ensemble surrogate baseline
- run random, greedy, UCB, EI, and conformal-UCB acquisition rules
- export run summaries, diagnostics, plots, checkpoints, and sanity-check reports

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
py -3.12 -m unittest discover -s tests -p "test_*.py"
```

## Notes

- The detailed project README is in `protein_bo_conformal/README.md`.
- Raw benchmark data stays local and is not pushed to GitHub.
- The local `paper/` directory is also intentionally excluded from GitHub pushes.
- `version 1/`, `version 2/`, and `version 2.zip` are local research materials, not part of the GitHub code payload.
