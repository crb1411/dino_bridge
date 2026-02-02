from typing import Callable, Optional
import logging
import torch

logger = logging.getLogger('dinov3')
DEFAULT_NAME_WIDTH = 12
DEFAULT_NUM_WIDTH = 8

def _log_line(msg, enabled=True):
    if not enabled:
        return
    if logger is not None and logger.hasHandlers():
        logger.info(msg)
    else:
        print(msg, flush=True)

def _normalize_prefix(prefix: str) -> str:
    label = prefix.strip()
    if label.endswith(":"):
        label = label[:-1]
    return f"{label},"


def _format_topk(indices: torch.Tensor, values: torch.Tensor, value_fmt: str) -> str:
    fmt = value_fmt[1:] if value_fmt.startswith(":") else value_fmt
    return ", ".join(
        f"{int(i)}:{format(float(v), fmt)}"
        for i, v in zip(indices.tolist(), values.tolist())
    )


def format_last_row_stats(
    tensor: torch.Tensor,
    k: int,
    prefix: str,
    *,
    name_width: int = DEFAULT_NAME_WIDTH,
    num_width: int = DEFAULT_NUM_WIDTH,
    value_fmt: str = ".4f",
    tag: str = "",
) -> str:
    assert tensor.dim() in (1, 2)
    row = tensor if tensor.dim() == 1 else tensor[-1]
    row_cpu = row.detach().float().cpu()
    label = _normalize_prefix(prefix)
    if row_cpu.numel() == 0:
        return f"{tag}{label:<{name_width}} empty"
    k_eff = min(max(int(k), 1), row_cpu.numel())
    max_val = row_cpu.max().item()
    mean_val = row_cpu.mean().item()
    min_val = row_cpu.min().item()
    top_vals, top_idx = torch.topk(row_cpu, k_eff, largest=True, sorted=True)
    bottom_vals, bottom_idx = torch.topk(row_cpu, k_eff, largest=False, sorted=True)
    fmt = value_fmt[1:] if value_fmt.startswith(":") else value_fmt
    max_str = f"{max_val:>{num_width}{fmt}}"
    mean_str = f"{mean_val:>{num_width}{fmt}}"
    min_str = f"{min_val:>{num_width}{fmt}}"
    return (
        f"{tag}{label:<{name_width}} max={max_str}, mean={mean_str}, min={min_str}| "
        f"top{k_eff}=[{_format_topk(top_idx, top_vals, fmt)}] "
        f"bottom{k_eff}=[{_format_topk(bottom_idx, bottom_vals, fmt)}]"
    )


def log_last_row_stats(
    tensor: torch.Tensor,
    k: int,
    prefix: str,
    *,
    log_fn: Optional[Callable[[str], None]] = _log_line,
    enabled: bool = True,
    name_width: int = DEFAULT_NAME_WIDTH,
    num_width: int = DEFAULT_NUM_WIDTH,
    value_fmt: str = ".4f",
    tag: str = "",
) -> None:
    if not enabled:
        return
    line = format_last_row_stats(
        tensor,
        k,
        prefix,
        name_width=name_width,
        num_width=num_width,
        value_fmt=value_fmt,
        tag=tag,
    )
    if log_fn is None:
        print(line)
    else:
        log_fn(line)
