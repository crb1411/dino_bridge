import logging
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger('dinov3')

from dinov3.distributed import get_process_subgroup, get_subgroup_size
from dinov3.new_train.utils.log_status import log_last_row_stats

NAME_WIDTH = 18
NUM_WIDTH = 10


def _log_line(msg, enabled=True):
    if not enabled:
        return
    if logger is not None and logger.hasHandlers():
        logger.info(msg)
    else:
        print(msg, flush=True)


def _format_topk_int(indices, values):
    return ", ".join(f"{int(i)}:{int(v)}" for i, v in zip(indices.tolist(), values.tolist()))


def _argmax_counts(tensor, k):
    if tensor.numel() == 0 or tensor.shape[0] == 0:
        counts = torch.zeros(k, device=tensor.device, dtype=torch.float32)
    else:
        idx = torch.argmax(tensor, dim=-1)
        counts = torch.bincount(idx, minlength=k).to(dtype=torch.float32)
    if dist.is_initialized():
        dist.all_reduce(counts, group=get_process_subgroup())
    return counts


def _print_counts_stats(name, counts, enabled=True):
    if not enabled or counts is None:
        return
    counts_cpu = counts.detach().float().cpu()
    if counts_cpu.numel() == 0:
        label = f"{name},"
        _log_line(f"{label:<{NAME_WIDTH}} empty", enabled=enabled)
        return
    max_val = counts_cpu.max().item()
    mean_val = counts_cpu.mean().item()
    min_val = counts_cpu.min().item()
    k = min(5, counts_cpu.numel())
    top_vals, top_idx = torch.topk(counts_cpu, k, largest=True, sorted=True)
    bottom_vals, bottom_idx = torch.topk(counts_cpu, k, largest=False, sorted=True)
    label = f"{name},"
    _log_line(
        f"{label:<{NAME_WIDTH}} {max_val:>{NUM_WIDTH}.0f}, {mean_val:>{NUM_WIDTH}.2f}, {min_val:>{NUM_WIDTH}.0f}| "
        f"top5=[{_format_topk_int(top_idx, top_vals)}] "
        f"bottom5=[{_format_topk_int(bottom_idx, bottom_vals)}]",
        enabled=enabled,
    )


class SinkhornKnoppTeacher(nn.Module):
    """
    NOTE: This is a module and not a function in the `iBOTPatchLoss` class
    This is because we want to torch.compile it, and torch.compil-ing a single
    function with the `@torch.compile` decorator is bad.
    It's better to `module.compile()` it, as we can control when we enable or
    disable compilation globally.
    """

    @torch.no_grad()
    def forward(self, teacher_output, teacher_temp, n_masked_patches_tensor=None, n_iterations=3):
        teacher_output = teacher_output.float()
        # world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
        # B = Q.shape[1] * world_size # number of samples to assign
        B = n_masked_patches_tensor or teacher_output.shape[0]
        dist.all_reduce(B, group=get_process_subgroup())
        K = Q.shape[0]  # how many prototypes

        should_print = not dist.is_initialized() or dist.get_rank() == 0
        log_last_row_stats(
            teacher_output,
            5,
            "teacher_output",
            log_fn=_log_line,
            enabled=should_print,
        )
        log_last_row_stats(
            Q,
            5,
            "exp_t",
            log_fn=_log_line,
            enabled=should_print,
        )
        teacher_counts = _argmax_counts(teacher_output, K)

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q, group=get_process_subgroup())
        Q /= sum_Q
        log_last_row_stats(
            Q,
            5,
            "prob_norm",
            log_fn=_log_line,
            enabled=should_print,
        )

        for _ in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows, group=get_process_subgroup())
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        out = Q.t()
        log_last_row_stats(
            out,
            5,
            "sk_out",
            log_fn=_log_line,
            enabled=should_print,
        )
        out_counts = _argmax_counts(out, K)
        _print_counts_stats("teacher_index", teacher_counts, enabled=should_print)
        _print_counts_stats("sk_index", out_counts, enabled=should_print)
        return out
