# protein_bo_conformal

Research codebase for the version 2 paper project on uncertainty-aware closed-loop protein sequence optimization under feedback shift.

## Current Stage

The project is currently at the end of `Day 2`.

- `Day 1` established the reproducible execution shell.
- `Day 2` established the real data environment layer.

At this point, the repository can:

- load real `FLIP`, `FLIP2`, and local `ProteinGym` benchmark files
- standardize them into a shared in-memory dataset format
- construct shift-aware splits
- build an immutable oracle
- validate the data environment through `main.py`
- export run-local summaries, logs, and simple diagnostic plots

The project has not yet entered the method layer for:

- representations beyond the current shell interface
- surrogate training beyond the current environment scaffold
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

## Output Artifacts

Each run creates a timestamped directory under `outputs/results/` containing:

- `config_snapshot.yaml`
- `run_manifest.json`
- `logs/execution.log`
- `artifacts/dataset_summary.json`
- `artifacts/split_summary.json`
- `artifacts/oracle_validation.json`
- `artifacts/runner_summary.json`
- `plots/*.svg`

## Notes

- Raw benchmark data is expected to exist locally under `data/raw/`.
- Processed split caches and run artifacts are ignored by Git.
- The local `paper/` directory is intentionally excluded from GitHub pushes.
