"""Device selection helpers with graceful standard-library fallback."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved device information for the current run."""

    requested: str
    resolved: str
    accelerator_available: bool
    backend: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def resolve_device(preference: str = "auto", logger: Any | None = None) -> DeviceInfo:
    """Resolve the effective device without requiring torch to be installed."""
    preference = preference.lower()

    try:
        import torch  # type: ignore

        cuda_available = bool(torch.cuda.is_available())
        if preference == "cpu":
            info = DeviceInfo("cpu", "cpu", cuda_available, "torch", "CPU forced by config.")
        elif preference in {"cuda", "gpu"}:
            resolved = "cuda" if cuda_available else "cpu"
            reason = "CUDA requested." if cuda_available else "CUDA requested but unavailable; fell back to CPU."
            info = DeviceInfo(preference, resolved, cuda_available, "torch", reason)
        else:
            resolved = "cuda" if cuda_available else "cpu"
            reason = "Auto-selected CUDA." if cuda_available else "Auto-selected CPU because CUDA is unavailable."
            info = DeviceInfo(preference, resolved, cuda_available, "torch", reason)
    except Exception:
        info = DeviceInfo(
            requested=preference,
            resolved="cpu",
            accelerator_available=False,
            backend="none",
            reason="Torch is unavailable; using CPU-only shell.",
        )

    if logger is not None:
        logger.info("Device report: %s", info.to_dict())
    return info
