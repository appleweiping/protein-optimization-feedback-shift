# protein_bo_conformal

Research codebase for the version 2 paper project on uncertainty-aware closed-loop protein sequence optimization under feedback shift.

## Current Stage

The project is currently at the end of `Day 5`.

- `Day 1` established the reproducible execution shell.
- `Day 2` established the real data environment layer.
- `Day 3` hardened oracle consistency and split validation.
- `Day 4` established the shared representation layer and cache infrastructure.
- `Day 5` established the surrogate layer, deep ensemble baseline, training loop, and checkpoint flow.

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
- run unit tests for oracle and validation behavior
- run unit tests for representation behavior
- run unit tests for surrogate behavior
- export run-local summaries, logs, and simple diagnostic plots

The project has not yet entered the method layer for:

- conformal / weighted conformal uncertainty modules
- acquisition policy experiments beyond configuration placeholders

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

## Notes

- Raw benchmark data is expected to exist locally under `data/raw/`.
- Processed split caches and run artifacts are ignored by Git.
- The local `paper/` directory is intentionally excluded from GitHub pushes.
- `ESMEncoder` now defaults to the `transformers` backend with `facebook/esm2_t6_8M_UR50D`.
- `fair-esm` is also installed and can be selected with `representation.esm_backend: esm`.
- `stub` mode is still available for lightweight tests or offline fallback.
- The first real ESM run downloads model weights into the local Hugging Face or Torch cache.
