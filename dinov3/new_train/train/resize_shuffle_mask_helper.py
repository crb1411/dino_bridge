from __future__ import annotations

import logging
import math
import random
from typing import Any, Mapping, MutableMapping, Optional, Tuple

import torch
from torch import Tensor

from dinov3.data.masking import MaskingGenerator

logger = logging.getLogger("dinov3")


def gather_tokens(x: Tensor, idx: Tensor) -> Tensor:
    if idx.dim() == 1:
        idx = idx.unsqueeze(0).expand(x.size(0), -1)
    expand = idx.unsqueeze(-1).expand(-1, -1, *x.shape[2:])
    return x.gather(1, expand)


def invert_perm_idx(perm_idx: Tensor) -> Tensor:
    if perm_idx.dim() == 1:
        n = perm_idx.numel()
        inv = torch.empty_like(perm_idx)
        inv[perm_idx] = torch.arange(n, device=perm_idx.device, dtype=perm_idx.dtype)
        return inv
    b, n = perm_idx.shape
    inv = torch.empty_like(perm_idx)
    ar = torch.arange(n, device=perm_idx.device, dtype=perm_idx.dtype).unsqueeze(0).expand(b, -1)
    inv.scatter_(1, perm_idx, ar)
    return inv


def _stack_info_field(info_list, field: str) -> Optional[Tensor]:
    if torch.is_tensor(info_list):
        return info_list if info_list.numel() > 0 else None
    if not isinstance(info_list, list) or len(info_list) == 0:
        return None
    v0 = info_list[0]
    if not isinstance(v0, dict) or field not in v0:
        return None
    vals = []
    for d in info_list:
        if not isinstance(d, dict) or field not in d:
            return None
        v = d[field]
        if not torch.is_tensor(v):
            return None
        vals.append(v)
    try:
        return torch.stack(vals, dim=0)
    except Exception:
        return None


class ResizeShuffleMaskHelper:
    def __init__(
        self,
        *,
        patch_size: int,
        patch_shuffle_patch_probability: float,
        patch_shuffle_patch_min_max: Tuple[float, float],
        use_all_shift_mask: bool,
        random_circular_shift: bool,
    ) -> None:
        self.patch_size = int(patch_size)
        self.patch_shuffle_patch_probability = float(patch_shuffle_patch_probability)
        self.patch_shuffle_patch_min_max = patch_shuffle_patch_min_max
        self.use_all_shift_mask = bool(use_all_shift_mask)
        self.random_circular_shift = bool(random_circular_shift)
        self._mask_generator: Optional[MaskingGenerator] = None
        self._mask_grid: Optional[Tuple[int, int]] = None
        self._mask_ratio_max: Optional[float] = None

    def get_resize_shuffle_mask(
        self,
        *,
        resize_shuffle: Mapping[str, Any],
        key: str,
        expected_batch: Optional[int],
    ) -> Optional[Tensor]:
        mask = resize_shuffle.get(key)
        if not torch.is_tensor(mask):
            return None
        if mask.dim() != 2:
            logger.warning("resize_shuffle %s mask shape: %s", key, mask.shape)
            return None
        if expected_batch is not None and mask.shape[0] != expected_batch:
            logger.warning("resize_shuffle %s mask batch mismatch: %s vs expected %s", key, mask.shape, expected_batch)
            return None
        return mask

    def get_shift_perm_idx(
        self,
        *,
        resize_shuffle_r: Mapping[str, Any],
        resize_shuffle_o: Mapping[str, Any],
        expected_batch: Optional[int],
    ) -> Optional[Tuple[Tensor, Tensor]]:
        if "shift_info" not in resize_shuffle_r or "shift_info" not in resize_shuffle_o:
            return None
        shift_r = resize_shuffle_r["shift_info"]
        shift_o = resize_shuffle_o["shift_info"]
        if torch.is_tensor(shift_r) and torch.is_tensor(shift_o):
            perm_idx_r = shift_r
            perm_idx_o = shift_o
        else:
            perm_idx_r = _stack_info_field(shift_r, "perm_idx")
            perm_idx_o = _stack_info_field(shift_o, "perm_idx")
        if perm_idx_r is None or perm_idx_o is None:
            return None
        if perm_idx_r.numel() == 0 or perm_idx_o.numel() == 0:
            return None
        if (perm_idx_r < 0).any() or (perm_idx_o < 0).any():
            return None
        if expected_batch is not None:
            if perm_idx_r.shape[0] != expected_batch:
                logger.warning("perm_idx_r batch mismatch: %s vs expected %s", perm_idx_r.shape, expected_batch)
                return None
            if perm_idx_o.shape[0] != expected_batch:
                logger.warning("perm_idx_o batch mismatch: %s vs expected %s", perm_idx_o.shape, expected_batch)
                return None
        if perm_idx_r.shape != perm_idx_o.shape:
            logger.warning("perm_idx shapes differ: %s vs %s", perm_idx_r.shape, perm_idx_o.shape)
            return None
        return perm_idx_r, perm_idx_o

    def apply_patchshuffle_exclusion(self, sample_mask: Tensor, exclude_mask: Optional[Tensor]) -> Tensor:
        if exclude_mask is None or not torch.is_tensor(exclude_mask):
            return sample_mask
        if exclude_mask.shape != sample_mask.shape:
            logger.warning("patchshuffle exclude mask shape mismatch: %s vs %s", exclude_mask.shape, sample_mask.shape)
            return sample_mask
        if exclude_mask.device != sample_mask.device:
            exclude_mask = exclude_mask.to(device=sample_mask.device)
        if self.use_all_shift_mask:
            return sample_mask
        return sample_mask & ~exclude_mask

    def _infer_patch_grid_hw(self, x: Optional[Tensor], n_tokens: int) -> Tuple[int, int]:
        if torch.is_tensor(x):
            h = int(x.shape[-2])
            w = int(x.shape[-1])
            if self.patch_size > 0 and h % self.patch_size == 0 and w % self.patch_size == 0:
                grid_h = h // self.patch_size
                grid_w = w // self.patch_size
                if grid_h * grid_w == n_tokens:
                    return grid_h, grid_w
        side = int(math.sqrt(n_tokens))
        if side * side == n_tokens:
            return side, side
        return 1, n_tokens

    def _get_mask_generator(self, grid_hw: Tuple[int, int], ratio_max: float) -> MaskingGenerator:
        if self._mask_generator is None or self._mask_grid != grid_hw or self._mask_ratio_max != ratio_max:
            n_tokens = grid_hw[0] * grid_hw[1]
            max_num_patches = max(1, int(n_tokens * ratio_max))
            self._mask_generator = MaskingGenerator(input_size=grid_hw, max_num_patches=max_num_patches)
            self._mask_grid = grid_hw
            self._mask_ratio_max = ratio_max
        return self._mask_generator

    def _sample_patchshuffle_mask(
        self,
        *,
        n_tokens: int,
        batch_size: int,
        device: torch.device,
        grid_source: Optional[Tensor],
    ) -> Optional[Tensor]:
        if batch_size <= 0 or n_tokens <= 0:
            return None
        ratio_min, ratio_max = self.patch_shuffle_patch_min_max
        ratio_min = max(0.0, min(1.0, ratio_min))
        ratio_max = max(0.0, min(1.0, ratio_max))
        if ratio_max <= 0.0:
            return torch.zeros((batch_size, n_tokens), dtype=torch.bool, device=device)
        n_samples_masked = int(batch_size * self.patch_shuffle_patch_probability)
        if n_samples_masked <= 0:
            return torch.zeros((batch_size, n_tokens), dtype=torch.bool, device=device)

        grid_hw = self._infer_patch_grid_hw(grid_source, n_tokens)
        mask_generator = self._get_mask_generator(grid_hw, ratio_max)
        probs = torch.linspace(ratio_min, ratio_max, n_samples_masked + 1)
        masks_list = []
        for i in range(0, n_samples_masked):
            prob_max = probs[i + 1]
            mask = torch.BoolTensor(mask_generator(int(n_tokens * prob_max)))
            if self.random_circular_shift:
                shift_x = random.randint(0, mask.shape[0] - 1)
                shift_y = random.randint(0, mask.shape[1] - 1)
                mask = torch.roll(mask, (shift_x, shift_y), (0, 1))
            masks_list.append(mask)
        for _ in range(n_samples_masked, batch_size):
            masks_list.append(torch.BoolTensor(mask_generator(0)))
        random.shuffle(masks_list)
        return torch.stack(masks_list).flatten(1).to(device=device)

    def get_patchshuffle_sample_mask(
        self,
        *,
        runtime_cache: Optional[MutableMapping[str, Any]],
        n_tokens: int,
        batch_size: int,
        device: torch.device,
        grid_source: Optional[Tensor],
        allow_create: bool = True,
    ) -> Optional[Tensor]:
        masks = None
        if runtime_cache is not None:
            masks = runtime_cache.get("patchshuffle_sample_mask")
        if torch.is_tensor(masks):
            if masks.shape == (batch_size, n_tokens):
                if masks.device != device:
                    masks = masks.to(device=device)
                    if runtime_cache is not None:
                        runtime_cache["patchshuffle_sample_mask"] = masks
                return masks
            logger.warning("patchshuffle_sample_mask shape mismatch: %s vs (%s, %s)", masks.shape, batch_size, n_tokens)
        if not allow_create:
            return None
        masks = self._sample_patchshuffle_mask(
            n_tokens=n_tokens,
            batch_size=batch_size,
            device=device,
            grid_source=grid_source,
        )
        if masks is not None and runtime_cache is not None:
            runtime_cache["patchshuffle_sample_mask"] = masks
        return masks
