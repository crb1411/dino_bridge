import logging

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import dinov3.distributed as distributed
from dinov3.new_train.dtch import DTCH_BALANCE

logger = logging.getLogger("dinov3")


def _collect_dtch_modules(model):
    return [
        m
        for m in model.modules()
        if isinstance(m, DTCH_BALANCE) and getattr(m, "history_update", "") == "cache"
    ]


def _format_hist(vec: torch.Tensor, k: int = 5):
    v = vec.reshape(-1)
    k_eff = min(k, v.numel())
    top_vals, top_idx = torch.topk(v, k_eff)
    bottom_vals, bottom_idx = torch.topk(-v, k_eff)
    bottom_vals = -bottom_vals
    top_list = [
        f"{int(i)}:{float(val):.3e}" for i, val in zip(top_idx.tolist(), top_vals.tolist())
    ]
    bottom_list = [
        f"{int(i)}:{float(val):.3e}" for i, val in zip(bottom_idx.tolist(), bottom_vals.tolist())
    ]
    return (
        float(v.mean().item()),
        float(v.max().item()),
        float(v.min().item()),
        top_list,
        bottom_list,
    )


def warmup_history_cache(
    cfg,
    model,
    data_iter,
    *,
    teacher_temp,
    global_batch_size,
    start_iter=0,
):
    dtch_enabled = OmegaConf.select(cfg, "dtch.enabled")
    if not bool(dtch_enabled):
        return data_iter
    dtch_modules = _collect_dtch_modules(model)
    if not dtch_modules:
        return data_iter

    if distributed.is_main_process():
        logger.info("Warmup history cache: initializing and filling cache")

    orig_backprop = model.backprop_loss
    model.backprop_loss = lambda loss: None
    steps_done = 0
    warmup_total = 0
    pbar = None
    try:
        with torch.no_grad():
            # First step to initialize caches and determine capacity.
            data = next(data_iter)
            data["global_batch_size"] = global_batch_size
            model.forward_backward(
                data,
                teacher_temp=teacher_temp,
                iteration=start_iter,
                logger_freq=0,
            )
            steps_done = 1
            capacities = [
                int(getattr(m, "_history_cache_capacity", 0))
                for m in dtch_modules
                if int(getattr(m, "_history_cache_capacity", 0)) > 0
            ]
            warmup_total = max(capacities) if capacities else 0
            remaining = max(0, warmup_total - steps_done)
            if distributed.is_main_process():
                n_local_crops = int(getattr(cfg.crops, "local_crops_number", 0) or 0)
                local_batch = (
                    data["collated_local_crops"].shape[0] // n_local_crops
                    if n_local_crops > 0 and "collated_local_crops" in data
                    else None
                )
                logger.info(
                    "Warmup history cache: local_batch=%s global_batch=%s steps=%s",
                    local_batch,
                    global_batch_size,
                    warmup_total,
                )
                info_lines = []
                for idx, m in enumerate(dtch_modules):
                    cap = int(getattr(m, "_history_cache_capacity", 0))
                    eff = int(getattr(m, "_history_cache_size_eff", 0) or 0)
                    b_local = int(getattr(m, "_history_cache_batch", 0) or 0)
                    info_lines.append(
                        f"dtch[{idx}] cap={cap} eff={eff} b_local={b_local} hist_cache={int(m.history_cache_size)}"
                    )
                if info_lines:
                    logger.info("Warmup history cache details:\n%s", "\n".join(info_lines))
            if remaining > 0 and distributed.is_main_process():
                pbar = tqdm(total=remaining, desc="Warmup history cache", ncols=100)
            for _ in range(remaining):
                data = next(data_iter)
                data["global_batch_size"] = global_batch_size
                model.forward_backward(
                    data,
                    teacher_temp=teacher_temp,
                    iteration=start_iter,
                    logger_freq=0,
                )
                steps_done += 1
                if pbar is not None:
                    pbar.update(1)
    except StopIteration:
        logger.warning(
            "Warmup history cache stopped early at %d/%d steps (data exhausted).",
            steps_done,
            warmup_total,
        )
    finally:
        if pbar is not None:
            pbar.close()
        if dtch_modules:
            rank = distributed.get_rank() if distributed.is_enabled() else 0
            with torch.no_grad():
                for idx, m in enumerate(dtch_modules):
                    hist = m.history_Q.detach()
                    mean_h, max_h, min_h, top_h, bottom_h = _format_hist(hist, k=5)
                    logger.info(
                        "[rank%d][warmup][dtch%d] history mean=%.3e max=%.3e min=%.3e top5=%s bottom5=%s",
                        rank,
                        idx,
                        mean_h,
                        max_h,
                        min_h,
                        top_h,
                        bottom_h,
                    )
        model.backprop_loss = orig_backprop
    return data_iter
