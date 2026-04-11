# protein_bo_conformal

Research codebase for the version 2 paper project on uncertainty-aware closed-loop protein sequence optimization under feedback shift.

## Current Stage

The project is currently at the end of `Day 8`.

- `Day 1` established the reproducible execution shell.
- `Day 2` established the real data environment layer.
- `Day 3` hardened oracle consistency and split validation.
- `Day 4` established the shared representation layer and cache infrastructure.
- `Day 5` established the surrogate layer, deep ensemble baseline, training loop, and checkpoint flow.
- `Day 6` established the acquisition layer and the prediction-to-selection decision path.
- `Day 7` established the executable closed-loop runner, explicit loop state, recorder, and multi-round optimization traces.
- `Day 8` established the standardized baseline evaluation suite, aggregate metrics, baseline plots, and first analysis note.

At this point, the repository can:

- load real `FLIP`, `FLIP2`, and local `ProteinGym` benchmark files
- standardize them into a shared in-memory dataset format
- construct shift-aware splits
- build an immutable oracle
- validate the data environment and oracle through `main.py`
- encode sequences through a unified representation interface
- cache representation vectors by sequence hash
- use a real `transformers` ESM backend by default, with optional `fair-esm` support
- run a cross-benchmark data sanity check entrypoint
- run a representation sanity check entrypoint
- train a deep-ensemble surrogate baseline from shared sequence representations
- save per-round and per-member checkpoints for surrogate reruns
- export `mu`, `sigma`, member diversity, and basic surrogate plots
- build decision rules from surrogate outputs through random, greedy, UCB, EI, and conformal-UCB interfaces
- run acquisition sanity checks and system-level prediction-to-selection validation
- run standardized multi-seed baseline evaluation over `random`, `greedy`, and `ucb`
- aggregate best-so-far, simple regret, and selected-sample statistics from recorder outputs
- run a real train→predict→select→query→update closed loop with structured per-round logs
- run unit tests for oracle and validation behavior
- run unit tests for representation behavior
- run unit tests for surrogate behavior
- run unit tests for acquisition behavior
- run unit tests for loop state, buffer, and stopping behavior
- run unit tests for evaluation metrics
- export run-local summaries, logs, and simple diagnostic plots

The project has not yet entered the method layer for:

- conformal / weighted conformal uncertainty modules
- advanced acquisition policy experiments beyond the current baseline layer

## Implemented Data Coverage

The current registry supports:

- `flip.gb1`
- `flip.aav`
- `flip.meltome_human`
- `flip.meltome_cross_species`
- multiple `flip2.*` tasks from local CSVs
- locally unpacked `proteingym.*` substitution assays

## Current Default Run

The default configuration in `config/base.yaml` is:

- dataset: `flip.gb1`
- split: `mutation_extrapolation`
- train budget: `32`
- candidate pool: `256`

Run it with:

```powershell
py -3.12 main.py
```

For the Day 3 sanity sweep, run:

```powershell
py -3.12 experiments\data_sanity_check.py
```

For the Day 4 representation sweep, run:

```powershell
py -3.12 experiments\representation_sanity_check.py
```

For the Day 5 surrogate sanity sweep, run:

```powershell
py -3.12 experiments\surrogate_sanity_check.py
```

For the Day 6 acquisition sanity sweep, run:

```powershell
py -3.12 experiments\acquisition_sanity_check.py
```

For the Day 7 closed-loop sanity sweep, run:

```powershell
py -3.12 main.py --config config\experiment\day7_closed_loop_sanity.yaml --name day7-closed-loop-sanity
```

For the Day 8 baseline evaluation suite, run:

```powershell
py -3.12 experiments\baseline_eval.py --config config\experiment\day8_baseline_greedy_ucb.yaml
```

For lightweight automated tests, run:

```powershell
py -3.12 -m unittest discover -s tests -p "test_*.py"
```

## Output Artifacts

Each run creates a timestamped directory under `outputs/results/` containing:

- `config_snapshot.yaml`
- `run_manifest.json`
- `logs/execution.log`
- `artifacts/dataset_summary.json`
- `artifacts/split_summary.json`
- `artifacts/split_suite_summary.json`
- `artifacts/oracle_validation.json`
- `artifacts/runner_summary.json`
- `plots/*.svg`

Day 4 also writes representation summaries under `data/processed/metadata/`, including train/candidate embedding distance and cache statistics.
Day 5 additionally writes surrogate summaries, prediction tables, and per-member checkpoints under each run directory.
Day 6 additionally writes acquisition selections, decision diagnostics, and selection comparison plots.
Day 7 additionally writes per-method loop round traces, selected-sample tables, best-so-far trajectories, and closed-loop suite comparisons.
Day 8 additionally writes multi-seed baseline summaries under `outputs/results/baseline/`, including aggregate curves, scalar comparison plots, and a baseline analysis note.

## Notes

- Raw benchmark data is expected to exist locally under `data/raw/`.
- Processed split caches and run artifacts are ignored by Git.
- The local `paper/` directory is intentionally excluded from GitHub pushes.
- `ESMEncoder` now defaults to the `transformers` backend with `facebook/esm2_t6_8M_UR50D`.
- `fair-esm` is also installed and can be selected with `representation.esm_backend: esm`.
- `stub` mode is still available for lightweight tests or offline fallback.
- The first real ESM run downloads model weights into the local Hugging Face or Torch cache.
