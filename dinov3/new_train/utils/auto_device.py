# utils/device.py
from __future__ import annotations

import torch
from typing import Literal


DeviceType = Literal["cpu", "cuda", "npu", "mps", "xpu"]


def get_device(
    *,
    cpu: bool = False,
) -> torch.device:
    """
    Device auto-selection (Transformers / LLaMA-Factory style)

    Priority:
      1. cpu=True
      2. auto backend order
      3. cpu fallback
    """

    if cpu:
        return torch.device("cpu")

    # --- auto order (Transformers-like) ---
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")

    return torch.device("cpu")


def set_device_index(local_rank: int | None = None) -> None:
    """
    Set device index for multi-GPU training
    """
    device = get_device()
    if device.type == "cuda" and local_rank is not None:
        torch.cuda.set_device(local_rank)
    if device.type == "npu" and hasattr(torch, "npu") and local_rank is not None:
        torch.npu.set_device(local_rank)
    if device.type == "xpu" and hasattr(torch, "xpu") and local_rank is not None:
        torch.xpu.set_device(local_rank)


def get_distributed_backend(device: torch.device | None = None) -> str:
    if device is None:
        device = get_device()

    if device.type == "cuda":
        return "nccl"
    if device.type == "npu":
        return "hccl"
    if device.type in {"cpu", "mps", "xpu"}:
        return "gloo"
    return "gloo"


def empty_cache(device: torch.device | None = None) -> None:
    if device is None:
        device = get_device()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.empty_cache()


def synchronize(device_type: torch.device | None = None):
    if device_type is None:
        device_type = get_device()
    if device_type.type == "cuda":
        torch.cuda.synchronize()
    elif device_type.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif device_type.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()
        
