"""Registry for supported datasets and benchmark metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    """Declarative description of a benchmark dataset entry."""

    name: str
    benchmark: str
    task: str
    format_name: str
    raw_paths: tuple[str, ...]
    description: str
    path_mode: str = "all"

    def resolve_paths(self, project_root: Path) -> tuple[Path, ...]:
        """Resolve required dataset paths from the project root."""
        resolved = tuple(project_root / raw_path for raw_path in self.raw_paths)
        if self.path_mode == "all":
            missing = [str(path) for path in resolved if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Dataset '{self.name}' is not available locally. Missing required paths: "
                    + ", ".join(missing)
                )
            return resolved

        if self.path_mode == "any":
            for path in resolved:
                if path.exists():
                    return (path,)
            raise FileNotFoundError(
                f"Dataset '{self.name}' is not available locally. Expected one of: "
                + ", ".join(str(path) for path in resolved)
            )

        raise ValueError(f"Unsupported dataset path_mode '{self.path_mode}' for '{self.name}'.")

    def resolve_path(self, project_root: Path) -> Path:
        """Resolve the primary dataset path for single-source loaders."""
        return self.resolve_paths(project_root)[0]


_REGISTRY: dict[str, DatasetSpec] = {}
_MODULE_ROOT = Path(__file__).resolve().parents[1]


def register_dataset(spec: DatasetSpec) -> None:
    """Register a dataset specification in the global registry."""
    _REGISTRY[spec.name] = spec


def get_dataset(name: str) -> DatasetSpec:
    """Fetch a dataset specification by its registry name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown dataset '{name}'. Available datasets: {available}")
    return _REGISTRY[name]


def resolve_dataset_name(benchmark: str, task: str) -> str:
    """Resolve a benchmark/task pair to the canonical registry key."""
    return f"{benchmark.strip().lower()}.{task.strip().lower()}"


def resolve_dataset(
    registry_name: str | None = None,
    benchmark: str | None = None,
    task: str | None = None,
) -> DatasetSpec:
    """Resolve a dataset spec either directly or from benchmark/task."""
    if registry_name:
        return get_dataset(registry_name)
    if benchmark and task:
        return get_dataset(resolve_dataset_name(benchmark, task))
    raise ValueError("Either registry_name or benchmark/task must be provided.")


def list_datasets() -> list[str]:
    """Return sorted dataset names for reporting and debugging."""
    return sorted(_REGISTRY)


def _register_local_proteingym_substitutions() -> None:
    """Register locally available ProteinGym substitution assays."""
    substitutions_dir = _MODULE_ROOT / "data" / "raw" / "proteinGym" / "substitutions" / "DMS_ProteinGym_substitutions"
    if not substitutions_dir.exists():
        return

    for csv_path in sorted(substitutions_dir.glob("*.csv")):
        relative_path = csv_path.relative_to(_MODULE_ROOT).as_posix()
        task_name = csv_path.stem
        register_dataset(
            DatasetSpec(
                name=f"proteingym.{task_name.lower()}",
                benchmark="proteingym",
                task=task_name,
                format_name="generic_tabular",
                raw_paths=(relative_path,),
                description=f"ProteinGym substitution assay '{task_name}'.",
            )
        )


def _register_builtin_datasets() -> None:
    """Register expected benchmark dataset paths."""
    register_dataset(
        DatasetSpec(
            name="flip.gb1",
            benchmark="flip",
            task="gb1",
            format_name="flip_gb1_landscape",
            raw_paths=("data/raw/flip/gb1/MutLandscapes.txt",),
            description="FLIP GB1 mutational landscape matrix with single-mutant backgrounds.",
        )
    )
    register_dataset(
        DatasetSpec(
            name="flip.aav",
            benchmark="flip",
            task="aav",
            format_name="flip_aav",
            raw_paths=(
                "data/raw/flip/aav/processed_data.csv",
                "data/raw/flip/aav/P03135.fasta",
            ),
            description="FLIP AAV sequence-score table plus wild-type reference FASTA.",
        )
    )
    register_dataset(
        DatasetSpec(
            name="flip.meltome_human",
            benchmark="flip",
            task="meltome_human",
            format_name="flip_meltome_human",
            raw_paths=(
                "data/raw/flip/meltome/human.csv",
                "data/raw/flip/meltome/human_sequences.tsv",
            ),
            description="FLIP meltome human assay table plus UniProt sequence mapping.",
        )
    )
    register_dataset(
        DatasetSpec(
            name="flip.meltome_cross_species",
            benchmark="flip",
            task="meltome_cross_species",
            format_name="flip_meltome_cross_species",
            raw_paths=(
                "data/raw/flip/meltome/cross-species.csv",
                "data/raw/flip/meltome/sequences.fasta",
            ),
            description="FLIP meltome cross-species assay table plus FASTA sequence archive.",
        )
    )
    flip2_tasks = {
        "alpha_amylase.by_mutation": "data/raw/flip2/alpha_amylase/by_mutation.csv",
        "alpha_amylase.close_to_far": "data/raw/flip2/alpha_amylase/close_to_far.csv",
        "alpha_amylase.far_to_close": "data/raw/flip2/alpha_amylase/far_to_close.csv",
        "alpha_amylase.one_to_many": "data/raw/flip2/alpha_amylase/one_to_many.csv",
        "hydro.low_to_high": "data/raw/flip2/hydro/low_to_high.csv",
        "hydro.three_to_many": "data/raw/flip2/hydro/three_to_many.csv",
        "hydro.to_p01053": "data/raw/flip2/hydro/to_P01053.csv",
        "hydro.to_p06241": "data/raw/flip2/hydro/to_P06241.csv",
        "hydro.to_p0a9x9": "data/raw/flip2/hydro/to_P0A9X9.csv",
        "imine_reductase.two_to_many": "data/raw/flip2/imine_reductase/two_to_many.csv",
        "nuclease_b.two_to_many": "data/raw/flip2/nuclease_b/two_to_many.csv",
        "pdz3.single_to_double": "data/raw/flip2/pdz3/single_to_double.csv",
        "rhodopsin.by_wild_type": "data/raw/flip2/rhodopsin/by_wild_type.csv",
        "trpb.by_position": "data/raw/flip2/trpb/by_position.csv",
        "trpb.one_to_many": "data/raw/flip2/trpb/one_to_many.csv",
        "trpb.two_to_many": "data/raw/flip2/trpb/two_to_many.csv",
    }
    for task_name, raw_path in flip2_tasks.items():
        register_dataset(
            DatasetSpec(
                name=f"flip2.{task_name}",
                benchmark="flip2",
                task=task_name,
                format_name="generic_tabular",
                raw_paths=(raw_path,),
                description=f"FLIP2 task '{task_name}' sequence-target table.",
            )
        )
    register_dataset(
        DatasetSpec(
            name="proteingym.demo",
            benchmark="proteingym",
            task="demo",
            format_name="generic_tabular",
            raw_paths=("data/raw/proteinGym/demo.csv", "data/raw/proteinGym/demo.tsv"),
            description="Expected local ProteinGym-style validation file.",
            path_mode="any",
        )
    )
    _register_local_proteingym_substitutions()


_register_builtin_datasets()
