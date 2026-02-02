import logging
from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("dinov3")


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    return -(p * log_p).sum(dim=-1)


def _extract_logits_and_temp(
    entry: Any, default_temp: float
) -> Tuple[Optional[torch.Tensor], float]:
    if entry is None:
        return None, default_temp
    if torch.is_tensor(entry):
        return entry, default_temp
    if isinstance(entry, (tuple, list)) and len(entry) == 2:
        logits, temp = entry
        if not torch.is_tensor(logits):
            return None, default_temp
        return logits, default_temp if temp is None else float(temp)
    if isinstance(entry, Mapping):
        logits = entry.get("logits")
        if not torch.is_tensor(logits):
            return None, default_temp
        temp = entry.get("temp", default_temp)
        return logits, default_temp if temp is None else float(temp)
    return None, default_temp


def _find_ref_tensor(sharpen_data: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(sharpen_data):
        return sharpen_data
    if isinstance(sharpen_data, Mapping):
        for entry in sharpen_data.values():
            logits, _ = _extract_logits_and_temp(entry, 1.0)
            if torch.is_tensor(logits):
                return logits
    return None


class SharpnessLoss(nn.Module):
    def __init__(self, weight: float = 0.0, entropy_thresh: float = 0.0, log_freq: int = 0):
        super().__init__()
        self.weight = float(weight)
        self.entropy_thresh = float(entropy_thresh)
        self.log_freq = int(log_freq)

    def forward(
        self,
        sharpen_data,
        *,
        temp: float = 1.0,
        iteration: int = 0,
        logger_freq: int = 0,
        logger_loss: str | None = None,
        entropy_thresh: float | None = None,
    ):
        if entropy_thresh is None:
            entropy_thresh = self.entropy_thresh
        entropy_thresh = float(entropy_thresh)
        temp = float(temp)
        temp = max(temp, 1e-6)

        log_freq = logger_freq if logger_freq > 0 else self.log_freq
        if self.weight <= 0.0 and log_freq <= 0:
            ref = _find_ref_tensor(sharpen_data)
            if ref is None:
                return torch.tensor(0.0), {}
            return ref.new_tensor(0.0), {}

        if not isinstance(sharpen_data, Mapping):
            sharpen_data = {"head": sharpen_data}

        loss_dict = {}
        head_stats = {}

        total_loss_sum = None
        total_count = 0
        total_active = 0
        total_entropy_sum = 0.0
        total_active_sum = 0.0

        for key, entry in sharpen_data.items():
            logits, head_temp = _extract_logits_and_temp(entry, temp)
            if not torch.is_tensor(logits) or logits.numel() == 0:
                continue
            head_temp = max(float(head_temp), 1e-6)
            ent = _entropy_from_logits(logits / head_temp).reshape(-1)
            if ent.numel() == 0:
                continue
            mask = ent > entropy_thresh
            mask_f = mask.float()
            ent_loss = ent * mask_f + ent.detach() * (1.0 - mask_f)

            head_loss_sum = ent_loss.sum()
            if total_loss_sum is None:
                total_loss_sum = head_loss_sum
            else:
                total_loss_sum = total_loss_sum + head_loss_sum

            head_count = int(ent.numel())
            head_active = int(mask.sum().item())
            head_mean = ent.mean().item()
            head_mean_active = (ent * mask_f).sum() / mask_f.sum().clamp_min(1.0)

            loss_dict[key] = head_loss_sum / max(1, head_count)
            head_stats[key] = {
                "count": head_count,
                "active": head_active,
                "mean": head_mean,
                "mean_active": head_mean_active.item(),
                "temp": head_temp,
            }
            total_count += head_count
            total_active += head_active
            total_entropy_sum += ent.sum().item()
            total_active_sum += (ent * mask_f).sum().item()

        if total_loss_sum is None:
            ref = _find_ref_tensor(sharpen_data)
            if ref is None:
                return torch.tensor(0.0), {}
            return ref.new_tensor(0.0), {}

        head_entropy_raw = total_loss_sum / max(1, total_count)
        head_entropy_loss = head_entropy_raw * self.weight
        loss_dict["head_entropy_raw"] = head_entropy_raw
        loss_dict["head_entropy_loss"] = head_entropy_loss

        if log_freq and iteration % log_freq == 0:
            tag = f"[{logger_loss}] " if logger_loss else ""
            total_mean = total_entropy_sum / max(1, total_count)
            total_mean_active = total_active_sum / max(1, total_active)
            logger.info(
                f"{tag}[HeadEntropy] weight={self.weight:.3g} "
                f"thresh={entropy_thresh:.3g} heads={len(head_stats)} "
                f"active={total_active}/{total_count} "
                f"mean={total_mean:.3e} mean_active={total_mean_active:.3e} "
                f"raw={head_entropy_raw.item():.3e}"
            )
            for key, stats in head_stats.items():
                logger.info(
                    f"{tag}[HeadEntropy] head={key} temp={stats['temp']:.3g} "
                    f"active={stats['active']}/{stats['count']} "
                    f"mean={stats['mean']:.3e} "
                    f"mean_active={stats['mean_active']:.3e}"
                )

        return head_entropy_loss, loss_dict
