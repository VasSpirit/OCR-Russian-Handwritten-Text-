"""Device (CPU/CUDA) resolution shared by GUI, CLI and pipeline."""
from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)

DeviceMode = Literal["auto", "cpu", "cuda"]
DeviceName = Literal["cpu", "cuda"]


class CudaUnavailableError(RuntimeError):
    """Raised when the user explicitly requested CUDA but it is not available."""


def resolve_device(mode: str, *, cuda_available: bool | None = None) -> str:
    """Resolve a processing device from a user-selected mode.



    If ``cuda_available`` is omitted the value is probed via torch (no model
    initialization, just the runtime probe, cheap).

    Modes:

    * ``auto`` -> ``cuda`` if CUDA is available else ``cpu``.

    * ``cpu``  -> ``cpu`` always.


    * ``cuda`` -> ``cuda`` if CUDA is available, otherwise raise
      ``CudaUnavailableError`` (no silent fallback).
    """
    mode = (mode or "auto").lower().strip()
    if mode not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unknown device mode: {mode!r}; expected auto|cpu|cuda")
    if cuda_available is None:

        cuda_available = cuda_is_available()
    if mode == "auto":
        device = "cuda" if cuda_available else "cpu"
        logger.info("DEVICE_RESOLVED mode=%s device=%s", mode, device)
        return device
    if mode == "cpu":
        logger.info("DEVICE_RESOLVED mode=%s device=cpu", mode)
        return "cpu"
    if not cuda_available:
        logger.error("DEVICE_RESOLVED mode=cuda error=GPU (CUDA) unavailable")
        raise CudaUnavailableError(
            "GPU (CUDA) недоступна. Проверьте NVIDIA-драйвер и установку PyTorch с поддержкой CUDA."
        )
    logger.info("DEVICE_RESOLVED mode=%s device=cuda", mode)
    return "cuda"


def cuda_is_available() -> bool:
    """Cheap torch probe (no model initialization)."""
    import torch

    return bool(torch.cuda.is_available())


def cuda_device_name(device_index: int = 0) -> str:
    """Return the CUDA device name, or empty string if unavailable."""
    if not cuda_is_available():
        return ""
    import torch

    try:
        return str(torch.cuda.get_device_name(device_index))
    except (RuntimeError, AssertionError, IndexError):
        return ""


def gpu_status() -> dict[str, object]:
    """Describe GPU/CUDA availability for the GUI status panel."""
    available = cuda_is_available()
    status: dict[str, object] = {
        "device_found": available,
        "cuda_available": available,
        "device_name": cuda_device_name() if available else "",
    }
    return status


def pin_memory_for(device: str) -> bool:
    """``pin_memory=True`` only when an actual accelerator is used."""
    return device == "cuda"