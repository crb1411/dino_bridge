from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
import logging

from dinov3.new_train.utils.log_status import log_last_row_stats


logger = logging.getLogger('dinov3')
try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None


def _cfg_select(cfg, key, default):
    if cfg is None:
        return default
    if OmegaConf is not None:
        try:
            return OmegaConf.select(cfg, key, default=default)
        except Exception:
            pass
    cur = cfg
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _is_dist_initialized():
    return dist.is_available() and dist.is_initialized()


def _get_world_size():
    return dist.get_world_size() if _is_dist_initialized() else 1


def _all_reduce(tensor, op=dist.ReduceOp.SUM):
    if not _is_dist_initialized():
        return
    # HCCL on NPU does not support float64 all_reduce. Reduce in float32 and cast back.
    if tensor.is_floating_point() and tensor.dtype == torch.float64 and tensor.device.type == "npu":
        reduced = tensor.to(dtype=torch.float32)
        dist.all_reduce(reduced, op=op)
        tensor.copy_(reduced.to(dtype=torch.float64))
        return
    dist.all_reduce(tensor, op=op)


def _is_main_process():
    return (not _is_dist_initialized()) or dist.get_rank() == 0


def _log_line(msg: str) -> None:
    if logger is not None and logger.hasHandlers():
        logger.info(msg)
    elif _is_main_process():
        print(msg)


def _dist_min_max_int(value: int, device: torch.device) -> tuple:
    if not _is_dist_initialized():
        return value, value
    t = torch.tensor(int(value), device=device, dtype=torch.long)
    t_min = t.clone()
    t_max = t.clone()
    dist.all_reduce(t_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
    return int(t_min.item()), int(t_max.item())


def _entropy_from_probs(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    probs = probs / probs.sum().clamp_min(eps)
    probs = probs.clamp_min(eps)
    return -(probs * probs.log()).sum()


def _entropy_from_logits(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    probs = probs.clamp_min(eps)
    return -(probs * probs.log()).sum()


def compute_balance_weights(history_Q: torch.Tensor, beta: float, eps: float) -> torch.Tensor:
    hist = history_Q
    if _is_dist_initialized():
        hist = hist.clone()
        _all_reduce(hist)
    hist = hist.clamp_min(eps)
    beta = float(beta)
    if beta < 0.0:
        beta = 0.0
    if beta <= 1.0:
        h_sum = hist.sum()
        k = float(hist.numel())
        w = (1.0 - beta) / h_sum + beta / (k * hist)
    else:
        w = hist.pow(-beta)
    return w


def _compute_balance_log_weights(history_Q: torch.Tensor, beta: float, eps: float) -> torch.Tensor:
    hist = history_Q.clamp_min(eps)
    log_hist = torch.log(hist)
    beta = float(beta)
    if beta < 0.0:
        beta = 0.0
    if beta <= 0.0:
        log_w = torch.full_like(hist, -torch.log(hist.sum()))
    elif beta == 1.0:
        log_k = torch.log(hist.new_tensor(float(hist.numel())))
        log_w = -log_k - log_hist
    elif beta < 1.0:
        log_h_sum = torch.log(hist.sum())
        log_k = torch.log(hist.new_tensor(float(hist.numel())))
        log_a = torch.log(hist.new_tensor(1.0 - beta)) - log_h_sum
        log_b = torch.log(hist.new_tensor(beta)) - log_k - log_hist
        log_w = torch.logaddexp(log_a, log_b)
    else:
        log_w = -beta * log_hist
    return log_w


class DTCH_BALANCE(nn.Module):
    """
    Dual-temperature history-balanced assignments without Sinkhorn.

    - soft temp for history update
    - history-based balance prior
    - sharpen by power and renormalize
    """

    def __init__(
        self,
        K: int,
        history_cache_size: int = 20000,
        cfg=None,
        *,
        boost_alpha: float = 0.3,
        boost_w_max: float = 50.0,
        boost_threshold_divisor: float = 500.0,
        boost_eps: float = 1e-6,
        boost_enabled: bool = True,
        logits_temp_max=None,
        dt_temp_scale: float = 1,
        dt_exp_power: Optional[float] = None,
        balance_beta: float = 1,
        history_update: str = "cache",
        history_cache_recompute_interval: int = 5000,
    ):
        super().__init__()
        self.K = K
        self.history_cache_size = history_cache_size
        self.register_buffer("history_Q", torch.full((K,), float("nan"), dtype=torch.float64))
        self.register_buffer("history_batch", torch.tensor(0, dtype=torch.long))
        self.argmax_count_soft = None
        self.argmax_count_sharp = None
        self.argmax_ring_soft = None
        self.argmax_ring_sharp = None
        self.argmax_pos_soft = 0
        self.argmax_pos_sharp = 0
        self.argmax_filled_soft = 0
        self.argmax_filled_sharp = 0
        self._history_Q_initialized = False
        self._history_loaded_from_ckpt = False
        self._history_batch_from_ckpt = None
        self.logits_temp_max = 30.0 if logits_temp_max is None else logits_temp_max

        if cfg is not None:
            boost_alpha = _cfg_select(cfg, "ch_sk.boost_alpha", boost_alpha)
            boost_w_max = _cfg_select(cfg, "ch_sk.boost_w_max", boost_w_max)
            boost_threshold_divisor = _cfg_select(
                cfg, "ch_sk.boost_threshold_divisor", boost_threshold_divisor
            )
            boost_eps = _cfg_select(cfg, "ch_sk.boost_eps", boost_eps)
            boost_enabled = _cfg_select(cfg, "ch_sk.boost_enabled", boost_enabled)
        if cfg is not None:
            dt_temp_scale = _cfg_select(cfg, "ch_sk.dt_temp_scale", dt_temp_scale)
            dt_exp_power = _cfg_select(cfg, "ch_sk.dt_exp_power", dt_exp_power)
            balance_beta = _cfg_select(cfg, "ch_sk.balance_beta", balance_beta)
            history_update = _cfg_select(cfg, "ch_sk.history_update", history_update)
            history_update = _cfg_select(cfg, "ch_sk.history_update_mode", history_update)
            history_cache_recompute_interval = _cfg_select(
                cfg,
                "ch_sk.history_cache_recompute_interval",
                history_cache_recompute_interval,
            )

        self.boost_alpha = float(boost_alpha)
        self.boost_w_max = float(boost_w_max)
        self.boost_threshold_divisor = float(boost_threshold_divisor)
        self.boost_eps = float(boost_eps)
        self.boost_enabled = bool(boost_enabled)

        self.dt_temp_scale = float(dt_temp_scale)
        if dt_exp_power is None:
            dt_exp_power = self.dt_temp_scale
        self.dt_exp_power = float(dt_exp_power)
        self.balance_beta = float(balance_beta)

        mode = str(history_update).lower()
        if mode not in ("ema", "cache"):
            if _is_main_process():
                _log_line(f"Unknown history_update {history_update}, fallback to ema")
            mode = "ema"
        self.history_update = mode
        self.history_cache_recompute_interval = int(history_cache_recompute_interval)
        if self.history_cache_recompute_interval < 0:
            self.history_cache_recompute_interval = 0

        self._history_cache = None  # [n_cache, K], LRU ring buffer (rank-local)
        self._history_cache_pos = 0
        self._history_cache_batch = None
        self._history_cache_size_eff = None
        self._history_cache_capacity = 0
        self._history_cache_update_steps = 0
        self._history_log_init = False
        self.log_enabled = True
        self.log_fn = _log_line

    def _log(self, msg: str) -> None:
        if self.log_fn is not None:
            self.log_fn(msg)

    def _format_vec(self, vec: torch.Tensor, k: int = 5):
        vec = vec.reshape(-1)
        k_eff = min(k, vec.numel())
        top_vals, top_idx = torch.topk(vec, k_eff)
        bottom_vals, bottom_idx = torch.topk(-vec, k_eff)
        bottom_vals = bottom_vals.neg()
        top_list = [
            '%d:%.3e' % (int(i), v)
            for i, v in zip(top_idx.tolist(), top_vals.tolist())
        ]
        bottom_list = [
            '%d:%.3e' % (int(i), v)
            for i, v in zip(bottom_idx.tolist(), bottom_vals.tolist())
        ]
        return (
            vec.mean().item(),
            vec.max().item(),
            vec.min().item(),
            top_list,
            bottom_list,
        )

    def _log_vec_line(self, name: str, vec: torch.Tensor, loss_tag: str = "") -> None:
        log_last_row_stats(
            vec,
            5,
            name,
            log_fn=self._log,
            tag=loss_tag,
            name_width=28,
            num_width=11,
            value_fmt=".3e",
        )

    def _log_history_block(
        self,
        *,
        loss_tag: str,
        iteration: int,
        alpha: float,
        w_max: float,
        divisor: float,
        boost_mask: torch.Tensor,
        mean_hist: torch.Tensor,
        threshold: torch.Tensor,
        hist_snapshot: torch.Tensor,
        exp_power: float,
        Q_batch_soft: torch.Tensor,
        teacher_output: torch.Tensor,
    ) -> None:
        boost_cnt = int(boost_mask.sum().item())
        self._log(
            f"{loss_tag}[CHSK-BALANCE] "
            f"alpha={alpha:.3g} w_max={w_max:.3g} divisor={divisor:.3g} | "
            f"boost_cnt={boost_cnt} | "
            f"mean={mean_hist.item():.3e} thr={threshold.item():.3e}"
        )
        hist_mean, hist_max, hist_min, hist_top, hist_bottom = self._format_vec(
            hist_snapshot
        )
        self._log(
            f"{loss_tag}history: mean={hist_mean:.3e} "
            f"max={hist_max:.3e} min={hist_min:.3e} | "
            f"history_cache_size = {self._history_cache_size_eff} |"
            f"top5={hist_top} | bottom5={hist_bottom}"
        )
        hist_vec = torch.nan_to_num(
            hist_snapshot.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0
        )
        hist_entropy = _entropy_from_probs(hist_vec)
        self._log(
            f"{loss_tag}history_entropy={hist_entropy.item():.3e} "
            f"blance_factor={self.balance_beta:.3g} exp_power={exp_power:.3g} "
        )
        teacher_input_last = teacher_output
        if teacher_input_last.dim() > 1:
            teacher_input_last = teacher_input_last[-1, :]
        teacher_input_last = teacher_input_last.reshape(-1)
        self._log_vec_line("teacher_input_last_col:", teacher_input_last, loss_tag=loss_tag)
        input_history_last = Q_batch_soft.t()
        if input_history_last.dim() > 1:
            input_history_last = input_history_last[-1, :]
        input_history_last = input_history_last.reshape(-1)
        self._log_vec_line("input_history_last_col:", input_history_last, loss_tag=loss_tag)
        

    def _log_entropy_block(
        self,
        *,
        loss_tag: str,
        q: torch.Tensor,
        logits_temp_clamp: torch.Tensor,
        q_balance: torch.Tensor,
        p: torch.Tensor,
        teacher_output: torch.Tensor,
        teacher_temp: float,
    ) -> None:
        p_last = p
        if p_last.dim() > 1:
            p_last = p_last[-1, :]
        p_last = p_last.reshape(-1)
        self._log_vec_line("teacher_out_scale_prob:", p_last, loss_tag=loss_tag)
        ent_eps = 1e-12
        orig_last = teacher_output
        if orig_last.dim() > 1:
            orig_last = orig_last[-1, :]
        orig_last = orig_last.reshape(-1)
        orig_logits = orig_last / teacher_temp
        ent_orig = _entropy_from_logits(orig_logits, ent_eps)

        soft_last = logits_temp_clamp
        if soft_last.dim() > 1:
            soft_last = soft_last[-1, :]
        soft_last = soft_last.reshape(-1)
        ent_soft = _entropy_from_logits(soft_last, ent_eps)

        bal_last = q_balance
        if bal_last.dim() > 1:
            bal_last = bal_last[-1, :]
        bal_last = bal_last.reshape(-1)
        ent_balance = _entropy_from_probs(bal_last, ent_eps)
        bal_mean, bal_max, bal_min, bal_top, bal_bottom = self._format_vec(bal_last)

        sharp_last = q
        if sharp_last.dim() > 1:
            sharp_last = sharp_last[-1, :]
        sharp_last = sharp_last.reshape(-1)
        ent_sharp = _entropy_from_probs(sharp_last, ent_eps)

        self._log_vec_line(
            "teacher_out_balance_prob:",
            bal_last,
            loss_tag=loss_tag,
        )
        q_last = q
        if q_last.dim() > 1:
            q_last = q_last[-1, :]
        q_last = q_last.reshape(-1)
        self._log_vec_line("teacher_out_prob:", q_last, loss_tag=loss_tag)
        self._log(
            f"{loss_tag}entropy_last: orig={ent_orig.item():.6e} "
            f"soft={ent_soft.item():.6e} "
            f"balance={ent_balance.item():.6e} "
            f"exppower={ent_sharp.item():.6e}"
        )

    def _log_argmax_block(
        self,
        *,
        loss_tag: str,
        soft_counts: torch.Tensor,
        sharp_counts: torch.Tensor,
    ) -> None:
        def _format_top_bottom(counts, k=5):
            total_val = int(counts.sum().item())
            total_den = max(1, total_val)
            k_eff = min(k, counts.numel())
            counts_f = counts.float()
            top_vals, top_idx = torch.topk(counts_f, k_eff)
            bottom_vals, bottom_idx = torch.topk(-counts_f, k_eff)
            bottom_vals = -bottom_vals

            def _fmt(idx, val):
                cnt = int(val.item())
                pct = 100.0 * cnt / total_den if total_val > 0 else 0.0
                return f"{int(idx.item())}:{cnt}({pct:.3f}%)"

            top_list = [_fmt(i, v) for i, v in zip(top_idx, top_vals)]
            bottom_list = [_fmt(i, v) for i, v in zip(bottom_idx, bottom_vals)]
            return total_val, top_list, bottom_list

        soft_total, soft_top, soft_bottom = _format_top_bottom(soft_counts)
        sharp_total, sharp_top, sharp_bottom = _format_top_bottom(sharp_counts)
        soft_nz = int((soft_counts > 0).sum().item())
        sharp_nz = int((sharp_counts > 0).sum().item())
        k_total = soft_counts.numel()
        soft_zero = k_total - soft_nz
        sharp_zero = k_total - sharp_nz
        self._log(
            f"{loss_tag}argmax_soft: total={soft_total} "
            f"nz={soft_nz}/{k_total} zero={soft_zero} | "
            f"top5={soft_top} | bottom5={soft_bottom}"
        )
        self._log(
            f"{loss_tag}argmax_sharp: total={sharp_total} "
            f"nz={sharp_nz}/{k_total} zero={sharp_zero} | "
            f"top5={sharp_top} | bottom5={sharp_bottom}"
        )

    def _history_scale(self, b_local: int) -> int:
        if self._use_cache_history():
            return self._effective_history_size(b_local)
        return int(self.history_cache_size)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        history_key = prefix + "history_Q"
        history_batch_key = prefix + "history_batch"
        history_in_ckpt = history_key in state_dict
        history_batch_in_ckpt = history_batch_key in state_dict
        cache_keys = (
            prefix + "_history_cache",
            prefix + "_history_cache_pos",
            prefix + "_history_cache_batch",
            prefix + "_history_cache_size_eff",
            prefix + "_history_cache_capacity",
        )
        cache_state = {}
        for key in cache_keys:
            if key in state_dict:
                cache_state[key] = state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if history_key in missing_keys:
            missing_keys.remove(history_key)
        if history_batch_key in missing_keys:
            missing_keys.remove(history_batch_key)

        history_ok = torch.isfinite(self.history_Q).all()
        if not history_ok:
            self.history_Q.fill_(float("nan"))
        self._history_Q_initialized = bool(history_ok)
        self._history_loaded_from_ckpt = history_in_ckpt and history_ok
        self._history_batch_from_ckpt = None
        if history_batch_in_ckpt:
            batch_val = int(self.history_batch.item())
            if batch_val > 0:
                self._history_batch_from_ckpt = batch_val

        cache_ok = False
        cache = None
        cache_pos = 0
        cache_batch = 0
        cache_size_eff = 0
        cache_capacity = 0

        def _as_int(val, default=0):
            if val is None:
                return default
            if torch.is_tensor(val):
                return int(val.item())
            return int(val)

        if self._use_cache_history() and cache_state:
            cache = cache_state.get(prefix + "_history_cache", None)
            cache_pos = _as_int(cache_state.get(prefix + "_history_cache_pos"), 0)
            cache_batch = _as_int(cache_state.get(prefix + "_history_cache_batch"), 0)
            cache_size_eff = _as_int(cache_state.get(prefix + "_history_cache_size_eff"), 0)
            cache_capacity = _as_int(cache_state.get(prefix + "_history_cache_capacity"), 0)

            cache_ok = (
                torch.is_tensor(cache)
                and cache.dim() == 2
                and cache.shape[1] == self.K
                and cache_batch > 0
                and cache_size_eff > 0
                and cache_capacity > 0
            )
            if cache_ok:
                eff_expected = self._effective_history_size(cache_batch)
                cap_expected = max(1, eff_expected // cache_batch)
                cache_ok = (
                    cache.shape[0] == cache_capacity
                    and cache_capacity == cap_expected
                    and cache_size_eff == eff_expected
                    and 0 <= cache_pos < cache_capacity
                )
            if cache_ok:
                cache_ok = torch.isfinite(cache).all()

        if not (self._use_cache_history() and cache_ok):
            self._history_cache = None
            self._history_cache_pos = 0
            self._history_cache_batch = None
            self._history_cache_size_eff = None
            self._history_cache_capacity = 0
        else:
            self._history_cache = cache
            self._history_cache_pos = cache_pos
            self._history_cache_batch = cache_batch
            self._history_cache_size_eff = cache_size_eff
            self._history_cache_capacity = cache_capacity
            if not self._history_Q_initialized:
                self.history_Q = cache.sum(dim=0, dtype=self.history_Q.dtype)
                self._history_Q_initialized = True

        if _is_main_process():
            history_state = "ok" if history_in_ckpt and self._history_Q_initialized else "missing"
            if history_in_ckpt and not self._history_Q_initialized:
                history_state = "invalid"
            if history_batch_in_ckpt:
                batch_state = (
                    "ok" if self._history_batch_from_ckpt is not None else "invalid"
                )
            else:
                batch_state = "missing"
            msg = (
                f"[CHSK-BALANCE][load] history_Q={history_state} "
                f"history_batch={batch_state}"
            )
            if self._use_cache_history():
                if cache_state:
                    cache_state_msg = "ok" if cache_ok else "invalid"
                    msg += f" cache={cache_state_msg}"
                    if cache_ok:
                        msg += (
                            f" cap={cache_capacity} batch={cache_batch} "
                            f"eff={cache_size_eff} pos={cache_pos}"
                        )
                else:
                    msg += " cache=missing"
            else:
                msg += " cache=disabled"
            _log_line(msg)

    def _use_cache_history(self) -> bool:
        return self.history_update == "cache"

    def _log_history_init(self, msg: str) -> None:
        if self._history_log_init:
            return
        if _is_main_process():
            _log_line(msg)
        self._history_log_init = True

    def _ensure_argmax_stats(self, device: torch.device, window_size: int) -> None:
        if window_size <= 0:
            return

        def _init(kind: str):
            if kind == "soft":
                self.argmax_count_soft = torch.zeros(self.K, dtype=torch.long, device=device)
                self.argmax_ring_soft = torch.full(
                    (window_size,), -1, dtype=torch.long, device=device
                )
                self.argmax_pos_soft = 0
                self.argmax_filled_soft = 0
            else:
                self.argmax_count_sharp = torch.zeros(self.K, dtype=torch.long, device=device)
                self.argmax_ring_sharp = torch.full(
                    (window_size,), -1, dtype=torch.long, device=device
                )
                self.argmax_pos_sharp = 0
                self.argmax_filled_sharp = 0

        if (
            self.argmax_count_soft is None
            or self.argmax_ring_soft is None
            or self.argmax_count_soft.numel() != self.K
            or self.argmax_ring_soft.numel() != window_size
        ):
            _init("soft")
        if (
            self.argmax_count_sharp is None
            or self.argmax_ring_sharp is None
            or self.argmax_count_sharp.numel() != self.K
            or self.argmax_ring_sharp.numel() != window_size
        ):
            _init("sharp")

        if self.argmax_count_soft.device != device:
            self.argmax_count_soft = self.argmax_count_soft.to(device)
        if self.argmax_ring_soft.device != device:
            self.argmax_ring_soft = self.argmax_ring_soft.to(device)
        if self.argmax_count_sharp.device != device:
            self.argmax_count_sharp = self.argmax_count_sharp.to(device)
        if self.argmax_ring_sharp.device != device:
            self.argmax_ring_sharp = self.argmax_ring_sharp.to(device)

    def _update_argmax_window(self, indices: torch.Tensor, window_size: int, *, kind: str) -> None:
        if window_size <= 0 or indices.numel() == 0:
            return
        if kind == "soft":
            counts = self.argmax_count_soft
            ring = self.argmax_ring_soft
            pos = self.argmax_pos_soft
            filled = self.argmax_filled_soft
        else:
            counts = self.argmax_count_sharp
            ring = self.argmax_ring_sharp
            pos = self.argmax_pos_sharp
            filled = self.argmax_filled_sharp

        n = int(indices.numel())
        if n >= window_size:
            indices = indices[-window_size:]
            counts = torch.bincount(indices, minlength=self.K).to(
                device=counts.device, dtype=counts.dtype
            )
            ring = indices.clone()
            pos = 0
            filled = window_size
        else:
            idx_pos = (torch.arange(n, device=indices.device) + pos) % window_size
            old = ring[idx_pos]
            if filled > 0:
                valid = old >= 0
                if valid.any():
                    old_counts = torch.bincount(old[valid], minlength=self.K).to(
                        device=counts.device, dtype=counts.dtype
                    )
                    counts = counts - old_counts
            new_counts = torch.bincount(indices, minlength=self.K).to(
                device=counts.device, dtype=counts.dtype
            )
            counts = counts + new_counts
            ring[idx_pos] = indices
            pos = int((pos + n) % window_size)
            filled = min(window_size, filled + n)

        if kind == "soft":
            self.argmax_count_soft = counts
            self.argmax_ring_soft = ring
            self.argmax_pos_soft = pos
            self.argmax_filled_soft = filled
        else:
            self.argmax_count_sharp = counts
            self.argmax_ring_sharp = ring
            self.argmax_pos_sharp = pos
            self.argmax_filled_sharp = filled

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        hist_key = prefix + "history_Q"
        if hist_key in state:
            hist_avg = self.history_Q.detach().clone()
            if _is_dist_initialized():
                _all_reduce(hist_avg)
                hist_avg = hist_avg / _get_world_size()
            state[hist_key] = hist_avg

        batch_key = prefix + "history_batch"
        if batch_key in state:
            batch_val = int(self.history_batch.item())
            if _is_dist_initialized():
                dev = self.history_Q.device
                b_min, b_max = _dist_min_max_int(batch_val, dev)
                batch_val = b_min if b_min == b_max else 0
            state[batch_key] = torch.tensor(batch_val, device=self.history_Q.device, dtype=torch.long)

        cache_ok = False
        if self._use_cache_history():
            cache = self._history_cache
            cache_ok = torch.is_tensor(cache) and cache.dim() == 2 and cache.numel() > 0
            if cache_ok:
                cache_ok = torch.isfinite(cache).all()
            if cache_ok:
                dev = cache.device
                cap = int(cache.shape[0])
                k_dim = int(cache.shape[1])
                batch = int(self._history_cache_batch or 0)
                eff = int(self._history_cache_size_eff or 0)
                cap_meta = int(self._history_cache_capacity or 0)
                if _is_dist_initialized():
                    cap_min, cap_max = _dist_min_max_int(cap, dev)
                    k_min, k_max = _dist_min_max_int(k_dim, dev)
                    batch_min, batch_max = _dist_min_max_int(batch, dev)
                    eff_min, eff_max = _dist_min_max_int(eff, dev)
                    capm_min, capm_max = _dist_min_max_int(cap_meta, dev)
                    if (
                        cap_min != cap_max
                        or k_min != k_max
                        or batch_min != batch_max
                        or eff_min != eff_max
                        or capm_min != capm_max
                    ):
                        cache_ok = False
                if cache_ok:
                    cache_avg = cache.detach().clone()
                    if _is_dist_initialized():
                        _all_reduce(cache_avg)
                        cache_avg = cache_avg / _get_world_size()
                    state[prefix + "_history_cache"] = cache_avg
                    state[prefix + "_history_cache_pos"] = torch.tensor(
                        int(self._history_cache_pos), device=dev, dtype=torch.long
                    )
                    state[prefix + "_history_cache_batch"] = torch.tensor(
                        batch, device=dev, dtype=torch.long
                    )
                    state[prefix + "_history_cache_size_eff"] = torch.tensor(
                        eff, device=dev, dtype=torch.long
                    )
                    state[prefix + "_history_cache_capacity"] = torch.tensor(
                        cap_meta, device=dev, dtype=torch.long
                    )

        if not (self._use_cache_history() and cache_ok):
            for key in (
                prefix + "_history_cache",
                prefix + "_history_cache_pos",
                prefix + "_history_cache_batch",
                prefix + "_history_cache_size_eff",
                prefix + "_history_cache_capacity",
            ):
                state.pop(key, None)

        return state

    def _effective_history_size(self, b_local: int) -> int:
        if b_local <= 0:
            return int(self.history_cache_size)
        eff = (int(self.history_cache_size) // b_local) * b_local
        if eff <= 0:
            eff = b_local
        return eff

    def _cache_init_mode(self, b_local: int) -> str:
        if b_local <= 0:
            return "batch"
        eff_size = self._effective_history_size(b_local)
        capacity = max(1, eff_size // b_local)
        cache_ok = (
            self._history_cache is not None
            and self._history_cache_batch == b_local
            and self._history_cache_size_eff == eff_size
            and self._history_cache_capacity == capacity
            and self._history_cache.shape[0] == capacity
            and self._history_cache.shape[1] == self.K
            and torch.isfinite(self.history_Q).all()
        )
        return "checkpoint" if cache_ok else "batch"

    def _init_history_cache(self, Q_local: torch.Tensor, *, init_mode: str) -> None:
        _, b_local = Q_local.shape
        eff_size = self._effective_history_size(b_local)
        if init_mode == "checkpoint":
            self._history_cache_update_steps = 0
            return
        capacity = max(1, eff_size // b_local)
        device = Q_local.device
        dtype = Q_local.dtype
        self._log_history_init(
            f"[CHSK-BALANCE][init] history/cache from batch "
            f"(b_local={b_local} cap={capacity} eff={eff_size})"
        )
        sum_Q_local = torch.sum(Q_local, dim=1)
        sum_Q_local_f64 = sum_Q_local.to(dtype=self.history_Q.dtype)
        scale = self._history_scale(b_local)
        self.history_Q = sum_Q_local_f64 * (scale / b_local)
        self._history_Q_initialized = True
        cache = self.history_Q.to(device=device, dtype=dtype).unsqueeze(0).repeat(capacity, 1)
        cache = cache / capacity
        self._history_cache = cache
        self._history_cache_pos = 0
        self._history_cache_batch = b_local
        self._history_cache_size_eff = eff_size
        self._history_cache_capacity = capacity
        self._history_cache_update_steps = 0

    def _ensure_history_Q(self, Q_local):
        _, b_local = Q_local.shape
        if not self._use_cache_history():
            if self._history_loaded_from_ckpt:
                if self._history_batch_from_ckpt is None or self._history_batch_from_ckpt != b_local:
                    self._log_history_init(
                        f"[CHSK-BALANCE][load] history batch mismatch "
                        f"(ckpt={self._history_batch_from_ckpt} cur={b_local}), "
                        f"reinit from batch"
                    )
                    self.history_Q.fill_(float("nan"))
                    self._history_Q_initialized = False
                self._history_loaded_from_ckpt = False
                self._history_batch_from_ckpt = None
            if not self._history_Q_initialized:
                sum_Q_local = torch.sum(Q_local, dim=1)
                sum_Q_local_f64 = sum_Q_local.to(dtype=self.history_Q.dtype)
                scale = self._history_scale(b_local)
                self._log_history_init(
                    f"[CHSK-BALANCE][init] history from batch "
                    f"(b_local={b_local} scale={scale})"
                )
                self.history_Q = sum_Q_local_f64 * (scale / b_local)
                self._history_Q_initialized = True
            self.history_batch.fill_(b_local)
            return
        init_mode = self._cache_init_mode(b_local)
        if init_mode != "checkpoint" and self._history_loaded_from_ckpt:
            eff_size = self._effective_history_size(b_local)
            capacity = max(1, eff_size // b_local)
            self._log_history_init(
                f"[CHSK-BALANCE][load] cache mismatch, reinit from batch "
                f"(b_local={b_local} cap={capacity} eff={eff_size})"
            )
            self.history_Q.fill_(float("nan"))
            self._history_Q_initialized = False
            self._history_cache = None
            self._history_cache_pos = 0
            self._history_cache_batch = None
            self._history_cache_size_eff = None
            self._history_cache_capacity = 0
            init_mode = "batch"
        self._init_history_cache(Q_local, init_mode=init_mode)
        self.history_batch.fill_(b_local)
        self._history_loaded_from_ckpt = False
        self._history_batch_from_ckpt = None

    def _update_history(self, Q_local):
        if not self._use_cache_history():
            self._ensure_history_Q(Q_local)
            _, b_local = Q_local.shape
            sum_Q_local = torch.sum(Q_local, dim=1)
            sum_Q_local_f64 = sum_Q_local.to(dtype=self.history_Q.dtype)
            scale = self._history_scale(b_local)
            self.history_Q = ((scale - b_local) / scale) * self.history_Q + sum_Q_local_f64
            return
        self._ensure_history_Q(Q_local)
        sum_Q_local = torch.sum(Q_local, dim=1).clamp(min=1e-2)
        sum_Q_local_f64 = sum_Q_local.to(dtype=self.history_Q.dtype)
        if self._history_cache is None or self._history_cache_capacity <= 0:
            self.history_Q = sum_Q_local_f64
            return
        if self._history_cache.device != self.history_Q.device:
            self._history_cache = self._history_cache.to(device=self.history_Q.device)
        pos = self._history_cache_pos
        old = self._history_cache[pos].to(dtype=self.history_Q.dtype)
        self.history_Q = self.history_Q - old + sum_Q_local_f64
        self._history_cache[pos] = sum_Q_local
        self._history_cache_pos = (pos + 1) % self._history_cache_capacity
        self._history_cache_update_steps += 1

        # Periodically rebuild the history sum from full cache to avoid drift
        # from long-running incremental add/sub updates.
        interval = int(self.history_cache_recompute_interval)
        if interval > 0 and (self._history_cache_update_steps % interval == 0):
            self.history_Q = torch.sum(
                self._history_cache,
                dim=0,
                dtype=self.history_Q.dtype,
            )

    @torch.no_grad()
    def forward(
        self,
        teacher_output,
        teacher_temp,
        n_masked_patches_tensor=None,
        n_iterations: int = 3,
        iteration: int = 0,
        logger_freq: int = 0,
        logger_loss: Optional[str] = None,
        *,
        boost_alpha: Optional[float] = None,
        boost_w_max: Optional[float] = None,
        boost_threshold_divisor: Optional[float] = None,
        boost_eps: Optional[float] = None,
        dt_temp_scale: Optional[float] = None,
        dt_exp_power: Optional[float] = None,
        boost_enabled: Optional[bool] = None,
        balance_beta: Optional[float] = None,
    ):
        teacher_output = teacher_output.float()

        scale = float(self.dt_temp_scale if dt_temp_scale is None else dt_temp_scale)
        scale = max(scale, 1e-6)
        exp_power = float(self.dt_exp_power if dt_exp_power is None else dt_exp_power)
        exp_power = max(exp_power, 1e-6)
        beta = float(self.balance_beta if balance_beta is None else balance_beta)

        logits_temp = teacher_output / (teacher_temp * scale)
        logits_temp_clamp = logits_temp.clamp(min=-30.0, max=self.logits_temp_max)

        # DTCH flow: soft temp -> update history -> balance prior -> sharpen temp.
        Q_batch_soft = torch.exp(logits_temp_clamp).t()

        b_local = int(logits_temp_clamp.shape[0])
        window_size = int(self._history_scale(b_local))
        with torch.no_grad():
            self._ensure_argmax_stats(logits_temp_clamp.device, window_size)
            soft_idx = logits_temp_clamp.argmax(dim=-1).reshape(-1).to(torch.int64)
            self._update_argmax_window(soft_idx, window_size, kind="soft")

        self._ensure_history_Q(Q_batch_soft)
        self._update_history(Q_batch_soft)

        alpha = float(self.boost_alpha if boost_alpha is None else boost_alpha)
        w_max = float(self.boost_w_max if boost_w_max is None else boost_w_max)
        divisor = float(
            self.boost_threshold_divisor
            if boost_threshold_divisor is None
            else boost_threshold_divisor
        )
        eps = float(self.boost_eps if boost_eps is None else boost_eps)
        eps_t = torch.tensor(eps, device=Q_batch_soft.device, dtype=Q_batch_soft.dtype)
        use_boost = self.boost_enabled if boost_enabled is None else bool(boost_enabled)

        with torch.no_grad():
            hist = self.history_Q
            mean_hist = hist.mean()
            threshold = mean_hist / divisor

            if use_boost:
                low_mask = hist < threshold
                w = abs((mean_hist / (hist + eps_t) - divisor)).pow(alpha)
                w = w.clamp(1.0, w_max)
                boost_mask = low_mask
                boost_w = w
            else:
                boost_mask = torch.zeros_like(hist, dtype=torch.bool)
                boost_w = None

        Q_batch = Q_batch_soft.clone()
        if boost_w is not None and boost_mask.any():
            Q_batch[boost_mask, :] *= boost_w[boost_mask, None]

        do_log = (
            self.log_enabled
            and logger_freq
            and iteration % logger_freq == 0
            and _is_main_process()
        )
        hist_snapshot = None
        loss_tag = f"[{logger_loss}] " if logger_loss else ""
        if do_log:
            with torch.no_grad():
                hist_snapshot = self.history_Q.detach()
                self._log_history_block(
                    loss_tag=loss_tag,
                    iteration=iteration,
                    alpha=alpha,
                    w_max=w_max,
                    divisor=divisor,
                    boost_mask=boost_mask,
                    mean_hist=mean_hist,
                    threshold=threshold,
                    hist_snapshot=hist_snapshot,
                    exp_power=exp_power,
                    Q_batch_soft=Q_batch_soft,
                    teacher_output=teacher_output,
                )

        eps_hist = 1e-6
        eps_w = 1e-25
        eps_norm = 1e-25

        p = Q_batch.t()
        p = p / p.sum(dim=1, keepdim=True).clamp_min(eps_norm)

        # Use global history sum for weights while keeping per-rank history/cache.
        hist_for_weight = self.history_Q
        if _is_dist_initialized():
            hist_for_weight = hist_for_weight.clone()
            _all_reduce(hist_for_weight)

        log_w = _compute_balance_log_weights(hist_for_weight, beta, eps_hist).to(
            device=p.device, dtype=p.dtype
        )
        log_w = log_w.clamp_min(torch.log(p.new_tensor(eps_w)))

        log_q = p.clamp_min(eps_norm).log() + log_w[None, :]
        q = torch.exp(log_q - torch.logsumexp(log_q, dim=1, keepdim=True))
        q_balance = q

        if exp_power != 1.0:
            q = q.pow(exp_power)
            q = q / q.sum(dim=1, keepdim=True).clamp_min(eps_norm)

        with torch.no_grad():
            self._ensure_argmax_stats(q.device, window_size)
            sharp_idx = q.argmax(dim=-1).reshape(-1).to(torch.int64)
            self._update_argmax_window(sharp_idx, window_size, kind="sharp")
        
        if do_log:
            with torch.no_grad():
                teacher_out_0_1_pro = torch.softmax(teacher_output / 0.1, dim=-1)[-1, :]
                self._log_vec_line(
                    "teacher_out_0_1_prob:",
                    teacher_out_0_1_pro,
                    loss_tag=loss_tag,
                )
                self._log_entropy_block(
                    loss_tag=loss_tag,
                    q=q,
                    logits_temp_clamp=logits_temp_clamp,
                    q_balance=q_balance,
                    p=p,
                    teacher_output=teacher_output,
                    teacher_temp=teacher_temp,
                )

        do_sync = logger_freq and iteration % logger_freq == 0 and _is_dist_initialized()
        if do_sync:
            soft_counts = self.argmax_count_soft.clone()
            sharp_counts = self.argmax_count_sharp.clone()
            _all_reduce(soft_counts)
            _all_reduce(sharp_counts)
        else:
            soft_counts = self.argmax_count_soft
            sharp_counts = self.argmax_count_sharp

        if do_log:
            with torch.no_grad():
                self._log_argmax_block(
                    loss_tag=loss_tag,
                    soft_counts=soft_counts,
                    sharp_counts=sharp_counts,
                )

        return q
