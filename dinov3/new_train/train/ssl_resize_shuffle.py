from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor

from dinov3.layers.dino_head import frozen_dino_head_forward
from dinov3.loss import DINOLoss, iBOTPatchLoss
from dinov3.new_train.models.inverse_patch import InversePatchEmbeddingMLP
from dinov3.new_train.train.resize_shuffle_mask_helper import (
    ResizeShuffleMaskHelper,
    gather_tokens,
    invert_perm_idx,
)
from dinov3.new_train.train.resize_shuffle_types import (
    ResizeShuffleStudentOutputs,
    ResizeShuffleTeacherOutputs,
    student_outputs_from_any,
    teacher_outputs_from_any,
)
from dinov3.new_train.train.ssl_meta_arch import SSLMetaArch
from dinov3.new_train.utils import get_device
from dinov3.train.cosine_lr_scheduler import linear_warmup_cosine_decay

logger = logging.getLogger("dinov3")


def _to_device_any(
    x: Any,
    *,
    device: torch.device,
    non_blocking: bool = True,
) -> Any:
    if torch.is_tensor(x):
        return x.to(device=device, non_blocking=non_blocking)
    if isinstance(x, dict):
        return {k: _to_device_any(v, device=device, non_blocking=non_blocking) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_device_any(v, device=device, non_blocking=non_blocking) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_device_any(v, device=device, non_blocking=non_blocking) for v in x)
    return x


class SSLResizeShuffle(SSLMetaArch):
    """SSLMetaArch with resizepaste + patchshuffle + bridge losses."""

    def __init__(self, cfg):
        super().__init__(cfg)

        def _sel(key: str, default):
            value = OmegaConf.select(cfg, key)
            return default if value is None else value

        self._schedule_epoch_length = int(cfg.train.OFFICIAL_EPOCH_LENGTH)
        self._schedule_total_iters = int(cfg.optim.epochs * self._schedule_epoch_length)
        self._resize_shuffle_weight_schedules: Dict[str, Optional[object]] = {}
        self._current_weights: Dict[str, float] = {}

        self.resize_paste_weight, self._resize_shuffle_weight_schedules["resize_paste"] = self._resolve_weight_config(
            "resize_paste_loss_weight",
            0.0,
        )

        self.patchshuffle_weight, self._resize_shuffle_weight_schedules["patchshuffle"] = self._resolve_weight_config(
            "patch_shuffle_loss_weight",
            _sel("resize_shuffle_augmentor.patchshuffle_weight", 0.0),
        )
        self.patchshuffle_patch_weight, self._resize_shuffle_weight_schedules["patchshuffle_patch"] = (
            self._resolve_weight_config(
                "patch_shuffle_patch_weight",
                _sel("resize_shuffle_augmentor.patchshuffle_patch_weight", 1.0),
            )
        )
        self.patchshuffle_cls_weight, self._resize_shuffle_weight_schedules["patchshuffle_cls"] = (
            self._resolve_weight_config(
                "patch_shuffle_cls_weight",
                _sel("resize_shuffle_augmentor.patchshuffle_cls_weight", 1.0),
            )
        )
        self.patchshuffle_out_dim = int(_sel("resize_shuffle_augmentor.patchshuffle_out_dim", cfg.ibot.head_n_prototypes))
        self.use_all_shift_mask = bool(_sel("resize_shuffle_augmentor.use_all_shift_mask", False))

        patch_prob = OmegaConf.select(cfg, "resize_shuffle_augmentor.patch_shuffle_patch_probability")
        if patch_prob is None:
            resize_shuffle_ratio = OmegaConf.select(cfg, "resize_shuffle_augmentor.patch_shuffle_patch_ratio")
            if resize_shuffle_ratio is None:
                patch_prob = 1.0
            else:
                patch_prob = resize_shuffle_ratio
                logger.warning("patch_shuffle_patch_ratio is deprecated; using it as sample probability")
        self.patch_shuffle_patch_probability = float(patch_prob)
        self.patch_shuffle_patch_probability = max(0.0, min(1.0, self.patch_shuffle_patch_probability))

        raw_patch_min_max = OmegaConf.select(cfg, "resize_shuffle_augmentor.patch_shuffle_patch_min_max")
        if raw_patch_min_max is None:
            raw_patch_min_max = OmegaConf.select(cfg, "ibot.mask_ratio_min_max")
        self.patch_shuffle_patch_min_max: Tuple[float, float]
        if raw_patch_min_max is not None:
            try:
                min_v, max_v = float(raw_patch_min_max[0]), float(raw_patch_min_max[1])
                if 0.0 <= min_v <= max_v <= 1.0:
                    self.patch_shuffle_patch_min_max = (min_v, max_v)
                else:
                    raise ValueError
            except Exception:
                logger.warning("invalid patch_shuffle_patch_min_max: %s", raw_patch_min_max)
                self.patch_shuffle_patch_min_max = (0.0, 0.0)
        else:
            self.patch_shuffle_patch_min_max = (0.0, 0.0)

        legacy_bridge_weight = OmegaConf.select(cfg, "resize_shuffle_augmentor.bridge_roll_weight")
        if legacy_bridge_weight is None:
            legacy_bridge_weight = OmegaConf.select(cfg, "resize_shuffle_augmentor.bridge_weight")
        self.bridge_patchshuffle_weight, self._resize_shuffle_weight_schedules["bridge_patchshuffle"] = (
            self._resolve_weight_config(
                "bridge_patchshuffle_weight",
                0.0 if legacy_bridge_weight is None else legacy_bridge_weight,
            )
        )
        self.bridge_global_weight, self._resize_shuffle_weight_schedules["bridge_global"] = self._resolve_weight_config(
            "bridge_global_weight",
            0.0,
        )
        self.patch_size = int(_sel("resize_shuffle_augmentor.patch_size", cfg.student.patch_size))
        self.bridge_pre_head_clip = float(_sel("resize_shuffle_augmentor.bridge_pre_head_clip", 10.0))
        self.bridge_logits_clip = float(_sel("resize_shuffle_augmentor.bridge_logits_clip", 30.0))

        cls_hist_cache = getattr(cfg.dtch, "cls_hist_cache", None)
        patch_hist_cache = getattr(cfg.dtch, "patch_hist_cache", None)

        def _build_dino_loss():
            return DINOLoss(self.dino_out_dim, hist_cache=cls_hist_cache, neg_alpha=self.dino_neg_alpha)

        self.resize_paste_loss = _build_dino_loss()
        self.resize_paste_loss_resized = _build_dino_loss()
        self.patchshuffle_cls_loss = _build_dino_loss()
        self.patchshuffle_cls_loss_resized = _build_dino_loss()

        self.patchshuffle_patch_loss = iBOTPatchLoss(
            self.patchshuffle_out_dim,
            hist_cache=patch_hist_cache,
            neg_alpha=self.ibot_neg_alpha,
        )
        self.patchshuffle_patch_loss_resized = iBOTPatchLoss(
            self.patchshuffle_out_dim,
            hist_cache=patch_hist_cache,
            neg_alpha=self.ibot_neg_alpha,
        )

        bridge_patchshuffle_schedule = self._resize_shuffle_weight_schedules.get("bridge_patchshuffle")
        bridge_global_schedule = self._resize_shuffle_weight_schedules.get("bridge_global")
        bridge_schedule_active = (
            (bridge_patchshuffle_schedule is not None and float(np.max(bridge_patchshuffle_schedule)) > 0.0)
            or (bridge_global_schedule is not None and float(np.max(bridge_global_schedule)) > 0.0)
        )
        bridge_active = (
            self.bridge_patchshuffle_weight > 0.0 or self.bridge_global_weight > 0.0 or bridge_schedule_active
        )
        if bridge_active and "bridge_patch_mlp" not in self.student:
            bridge_hidden_dim = OmegaConf.select(cfg, "bridge.hidden_dim", default=None)
            self.student["bridge_patch_mlp"] = InversePatchEmbeddingMLP(self.embed_dim, hidden_dim=bridge_hidden_dim)
            self.teacher["bridge_patch_mlp"] = InversePatchEmbeddingMLP(self.embed_dim, hidden_dim=bridge_hidden_dim)
            self.teacher["bridge_patch_mlp"].load_state_dict(self.student["bridge_patch_mlp"].state_dict())

        self._batch_meta: Dict[str, Any] = {}
        self._resize_shuffle_outputs: Optional[dict] = None
        self._cached_data: Optional[dict] = None
        self._mask_helper = ResizeShuffleMaskHelper(
            patch_size=self.patch_size,
            patch_shuffle_patch_probability=self.patch_shuffle_patch_probability,
            patch_shuffle_patch_min_max=self.patch_shuffle_patch_min_max,
            use_all_shift_mask=self.use_all_shift_mask,
            random_circular_shift=bool(self.cfg.ibot.mask_random_circular_shift),
        )
        self._update_resize_shuffle_weights(iteration=0)

    def _build_weight_schedule(self, raw_value):
        if OmegaConf.is_config(raw_value):
            raw_value = OmegaConf.to_container(raw_value, resolve=True)
        if not isinstance(raw_value, dict):
            return None
        required = {"start", "peak", "end"}
        if not required.issubset(raw_value.keys()):
            return None
        start = float(raw_value["start"])
        peak = float(raw_value["peak"])
        end = float(raw_value["end"])
        zero_epochs = float(raw_value.get("zero_epochs", 0.0))
        warmup_epochs = float(raw_value.get("warmup_epochs", 0.0))
        zero_iters = int(self._schedule_epoch_length * zero_epochs)
        warmup_iters = int(self._schedule_epoch_length * warmup_epochs)
        cosine_epochs = raw_value.get("cosine_epochs", None)
        cosine_iters = None
        if cosine_epochs is not None:
            cosine_iters = int(self._schedule_epoch_length * float(cosine_epochs))
        if zero_iters >= self._schedule_total_iters:
            return np.full((self._schedule_total_iters,), fill_value=start, dtype=np.float64)
        total_iters = self._schedule_total_iters - zero_iters
        if warmup_iters > total_iters:
            logger.warning(
                "warmup_epochs too large for schedule (%s > %s); clamping.",
                warmup_iters,
                total_iters,
            )
            warmup_iters = total_iters
        if cosine_iters is not None:
            max_cosine = max(total_iters - warmup_iters, 0)
            if cosine_iters > max_cosine:
                logger.warning(
                    "cosine_epochs too large for schedule (%s > %s); clamping.",
                    cosine_iters,
                    max_cosine,
                )
                cosine_iters = max_cosine
        schedule = linear_warmup_cosine_decay(
            start=start,
            peak=peak,
            end=end,
            warmup_iterations=warmup_iters,
            total_iterations=total_iters,
            cosine_iterations=cosine_iters,
        )
        if zero_iters > 0:
            prefix = np.full((zero_iters,), fill_value=start, dtype=np.float64)
            schedule = np.concatenate([prefix, schedule])
        return schedule

    def _resolve_weight_config(self, key: str, default_value: Any):
        raw_value = OmegaConf.select(self.cfg, f"resize_shuffle_augmentor.{key}")
        if raw_value is None:
            raw_value = default_value
        if raw_value is None:
            return 0.0, None
        schedule = self._build_weight_schedule(raw_value)
        if schedule is not None:
            return float(schedule[0]), schedule
        try:
            return float(raw_value), None
        except (TypeError, ValueError):
            logger.warning("Invalid resize_shuffle_augmentor.%s: %s", key, raw_value)
            return float(default_value), None

    def _weight_at(self, name: str, iteration: int, fallback: float) -> float:
        schedule = self._resize_shuffle_weight_schedules.get(name)
        if schedule is None:
            return float(fallback)
        if iteration < 0:
            return float(schedule[0])
        if iteration >= len(schedule):
            return float(schedule[-1])
        return float(schedule[iteration])

    def _update_resize_shuffle_weights(self, iteration: int) -> None:
        self._current_weights = {
            "resize_paste": self._weight_at("resize_paste", iteration, self.resize_paste_weight),
            "patchshuffle": self._weight_at("patchshuffle", iteration, self.patchshuffle_weight),
            "patchshuffle_patch": self._weight_at("patchshuffle_patch", iteration, self.patchshuffle_patch_weight),
            "patchshuffle_cls": self._weight_at("patchshuffle_cls", iteration, self.patchshuffle_cls_weight),
            "bridge_patchshuffle": self._weight_at(
                "bridge_patchshuffle",
                iteration,
                self.bridge_patchshuffle_weight,
            ),
            "bridge_global": self._weight_at("bridge_global", iteration, self.bridge_global_weight),
        }

    def update_iteration_dependent_weights(self, iteration: int) -> None:
        # Called by the training loop before forward/backward so schedules are aligned with current iteration.
        self._update_resize_shuffle_weights(iteration)

    def _get_current_weight(self, name: str, fallback: float) -> float:
        return float(self._current_weights.get(name, fallback))

    def _get_resize_shuffle_weights(self) -> Dict[str, float]:
        return {
            "resize_paste": self._get_current_weight("resize_paste", self.resize_paste_weight),
            "patchshuffle": self._get_current_weight("patchshuffle", self.patchshuffle_weight),
            "patchshuffle_patch": self._get_current_weight("patchshuffle_patch", self.patchshuffle_patch_weight),
            "patchshuffle_cls": self._get_current_weight("patchshuffle_cls", self.patchshuffle_cls_weight),
            "bridge_global": self._get_current_weight("bridge_global", self.bridge_global_weight),
            "bridge_patchshuffle": self._get_current_weight("bridge_patchshuffle", self.bridge_patchshuffle_weight),
        }

    @staticmethod
    def _has_active_aux_path(weights: Mapping[str, float]) -> bool:
        return any(value > 0.0 for value in weights.values())

    def _get_teacher_temp(self) -> float:
        temp = self._batch_meta.get("teacher_temp")
        if temp is None:
            fallback = OmegaConf.select(self.cfg, "teacher.teacher_temp")
            if fallback is None:
                fallback = 0.1
            logger.warning("resize_shuffle teacher_temp missing; using %s", fallback)
            return float(fallback)
        return float(temp)

    def _combine_dino_loss(self, loss_out):
        if isinstance(loss_out, dict):
            return loss_out["pos"] + self.dino_neg_lambda * loss_out["neg"]
        return loss_out

    def _combine_ibot_loss(self, loss_out):
        if isinstance(loss_out, dict):
            return loss_out["pos"] + self.ibot_neg_lambda * loss_out["neg"]
        return loss_out

    def _get_bridge_patch_mlp(self):
        if "bridge_patch_mlp" not in self.student:
            return None
        return self.student["bridge_patch_mlp"]

    def _sanitize_bridge_tensor(self, x: Tensor, clip_value: float) -> Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if clip_value > 0.0:
            x = x.clamp(min=-clip_value, max=clip_value)
        return x

    @torch.no_grad()
    def _recover_bridge_mlp_if_non_finite(self) -> None:
        bridge_mlp = self._get_bridge_patch_mlp()
        if bridge_mlp is None:
            return
        has_non_finite = False
        for param in bridge_mlp.parameters():
            if not torch.isfinite(param).all():
                has_non_finite = True
                break
        if not has_non_finite:
            return
        logger.warning("Detected non-finite bridge_patch_mlp parameters; resetting bridge MLP.")
        bridge_mlp.reset_parameters()
        if "bridge_patch_mlp" in self.teacher:
            self.teacher["bridge_patch_mlp"].load_state_dict(bridge_mlp.state_dict())

    def _add_bridge_logits(
        self,
        outputs: ResizeShuffleStudentOutputs,
        *,
        prefix: str,
        patch_tokens_resized: Tensor,
        patch_tokens_original: Tensor,
        bridge_mlp,
    ) -> None:
        if not (torch.is_tensor(patch_tokens_resized) and torch.is_tensor(patch_tokens_original)):
            return
        if patch_tokens_resized.shape != patch_tokens_original.shape:
            logger.warning(
                "%s token shape mismatch: %s vs %s",
                prefix,
                patch_tokens_resized.shape,
                patch_tokens_original.shape,
            )
            return
        try:
            bridge_resized_pre = bridge_mlp(patch_tokens_resized)
            bridge_original_pre = bridge_mlp(patch_tokens_original)
        except Exception as exc:
            logger.warning("%s bridge input skipped: %s", prefix, exc)
            return
        bridge_resized_pre = self._sanitize_bridge_tensor(bridge_resized_pre, self.bridge_pre_head_clip)
        bridge_original_pre = self._sanitize_bridge_tensor(bridge_original_pre, self.bridge_pre_head_clip)

        resized_attr = f"{prefix}_logits_resized"
        original_attr = f"{prefix}_logits_original"
        if not hasattr(outputs, resized_attr) or not hasattr(outputs, original_attr):
            logger.warning("unknown bridge output prefix: %s", prefix)
            return
        bridge_resized_logits = frozen_dino_head_forward(self.student.dino_head, bridge_resized_pre)
        bridge_original_logits = frozen_dino_head_forward(self.student.dino_head, bridge_original_pre)
        bridge_resized_logits = self._sanitize_bridge_tensor(bridge_resized_logits, self.bridge_logits_clip)
        bridge_original_logits = self._sanitize_bridge_tensor(bridge_original_logits, self.bridge_logits_clip)
        setattr(outputs, resized_attr, bridge_resized_logits)
        setattr(outputs, original_attr, bridge_original_logits)

    def _build_teacher_cls_targets_for_resize_shuffle(
        self,
        *,
        resize_shuffle_teacher: ResizeShuffleTeacherOutputs,
        iteration: int = 0,
        logger_freq: int = 0,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        if resize_shuffle_teacher is None:
            return None
        teacher_cls_logits_resized = resize_shuffle_teacher.raw_cls_logits_resized
        teacher_cls_logits_original = resize_shuffle_teacher.raw_cls_logits_original
        if teacher_cls_logits_resized is None or teacher_cls_logits_original is None:
            return None
        teacher_temp = self._get_teacher_temp()
        teacher_r_cls_targets = self.patchshuffle_cls_loss_resized.sinkhorn_knopp_teacher(
            teacher_cls_logits_resized,
            teacher_temp=teacher_temp,
            iteration=iteration,
            logger_freq=logger_freq,
            logger_loss="raw_cls_resized",
        )
        teacher_o_cls_targets = self.patchshuffle_cls_loss.sinkhorn_knopp_teacher(
            teacher_cls_logits_original,
            teacher_temp=teacher_temp,
            iteration=iteration,
            logger_freq=logger_freq,
            logger_loss="raw_cls_original",
        )
        return teacher_r_cls_targets, teacher_o_cls_targets

    def _align_teacher_targets_for_bridge(
        self,
        *,
        student_logits: Tensor,
        teacher_targets: Tensor,
        side: str,
        branch: str,
    ) -> Optional[Tensor]:
        if student_logits.shape[0] == teacher_targets.shape[0]:
            return teacher_targets
        t_len = int(teacher_targets.shape[0])
        s_len = int(student_logits.shape[0])
        if t_len <= 0 or s_len % t_len != 0:
            logger.warning(
                "%s %s logits/targets length mismatch: student=%s teacher=%s",
                branch,
                side,
                s_len,
                t_len,
            )
            return None
        repeat_factor = s_len // t_len
        return teacher_targets.repeat((repeat_factor, 1))

    def _loss_bridge_branch_from_resize_shuffle(
        self,
        *,
        student_aux_outputs: ResizeShuffleStudentOutputs,
        teacher_cls_targets: Tuple[Tensor, Tensor],
        branch_prefix: str,
    ) -> Optional[Tuple[Tensor, Optional[Tensor], Optional[Tensor]]]:
        if student_aux_outputs is None:
            return None
        bridge_logits_resized = getattr(student_aux_outputs, f"{branch_prefix}_logits_resized", None)
        bridge_logits_original = getattr(student_aux_outputs, f"{branch_prefix}_logits_original", None)
        if not torch.is_tensor(bridge_logits_resized) and not torch.is_tensor(bridge_logits_original):
            return None

        teacher_r_cls_targets, teacher_o_cls_targets = teacher_cls_targets

        # For global-bridge, treat the two global views as one merged branch.
        # This avoids enforcing an artificial resized/original split on global student outputs.
        if branch_prefix == "bridge_global":
            student_chunks = []
            teacher_chunks = []
            if torch.is_tensor(bridge_logits_resized):
                student_chunks.append(bridge_logits_resized)
                teacher_chunks.append(teacher_r_cls_targets)
            if torch.is_tensor(bridge_logits_original):
                student_chunks.append(bridge_logits_original)
                teacher_chunks.append(teacher_o_cls_targets)
            if not student_chunks:
                return None
            merged_student_logits = (
                torch.cat(student_chunks, dim=0) if len(student_chunks) > 1 else student_chunks[0]
            )
            merged_teacher_targets = (
                torch.cat(teacher_chunks, dim=0) if len(teacher_chunks) > 1 else teacher_chunks[0]
            )
            merged_teacher_targets = self._align_teacher_targets_for_bridge(
                student_logits=merged_student_logits,
                teacher_targets=merged_teacher_targets,
                side="merged",
                branch=branch_prefix,
            )
            if merged_teacher_targets is None:
                return None
            merged_loss = self._combine_dino_loss(
                self.patchshuffle_cls_loss(
                    merged_student_logits.unsqueeze(0), merged_teacher_targets.unsqueeze(0)
                )
            )
            if not torch.isfinite(merged_loss):
                return None
            return merged_loss, None, None

        loss_resized = None
        loss_original = None
        if torch.is_tensor(bridge_logits_resized):
            teacher_r_cls_targets_aligned = self._align_teacher_targets_for_bridge(
                student_logits=bridge_logits_resized,
                teacher_targets=teacher_r_cls_targets,
                side="resized",
                branch=branch_prefix,
            )
            if teacher_r_cls_targets_aligned is not None:
                loss_resized = self._combine_dino_loss(
                    self.patchshuffle_cls_loss_resized(
                        bridge_logits_resized.unsqueeze(0), teacher_r_cls_targets_aligned.unsqueeze(0)
                    )
                )
                if not torch.isfinite(loss_resized):
                    loss_resized = None
        if torch.is_tensor(bridge_logits_original):
            teacher_o_cls_targets_aligned = self._align_teacher_targets_for_bridge(
                student_logits=bridge_logits_original,
                teacher_targets=teacher_o_cls_targets,
                side="original",
                branch=branch_prefix,
            )
            if teacher_o_cls_targets_aligned is not None:
                loss_original = self._combine_dino_loss(
                    self.patchshuffle_cls_loss(
                        bridge_logits_original.unsqueeze(0), teacher_o_cls_targets_aligned.unsqueeze(0)
                    )
                )
                if not torch.isfinite(loss_original):
                    loss_original = None

        if loss_resized is None and loss_original is None:
            return None
        total = (loss_resized if loss_resized is not None else 0.0) + (
            loss_original if loss_original is not None else 0.0
        )
        if not torch.is_tensor(total):
            total = teacher_r_cls_targets.new_tensor(float(total))
        return total, loss_resized, loss_original

    def forward_backward(self, data, *, teacher_temp, iteration=0, **ignored_kwargs):
        n_local_crops = self.n_local_crops
        batch_size = data["collated_local_crops"].shape[0] // n_local_crops
        self._batch_meta["B"] = batch_size
        self._batch_meta["teacher_temp"] = float(teacher_temp)
        self._cached_data = data
        self._resize_shuffle_outputs = {}
        self._update_resize_shuffle_weights(iteration)
        self._recover_bridge_mlp_if_non_finite()

        device = get_device()
        if "resize_shuffle_aug_resized" in data and isinstance(data["resize_shuffle_aug_resized"], dict):
            data["resize_shuffle_aug_resized"] = _to_device_any(data["resize_shuffle_aug_resized"], device=device)
        if "resize_shuffle_aug" in data and isinstance(data["resize_shuffle_aug"], dict):
            data["resize_shuffle_aug"] = _to_device_any(data["resize_shuffle_aug"], device=device)

        try:
            return super().forward_backward(data, teacher_temp=teacher_temp, iteration=iteration, **ignored_kwargs)
        finally:
            self._cached_data = None
            self._resize_shuffle_outputs = None

    def _get_resize_shuffle_mask(self, resize_shuffle: Mapping[str, Any], key: str) -> Optional[Tensor]:
        return self._mask_helper.get_resize_shuffle_mask(
            resize_shuffle=resize_shuffle,
            key=key,
            expected_batch=self._batch_meta.get("B"),
        )

    def _get_resize_shuffle_aug_inputs(
        self,
    ) -> Optional[Tuple[Mapping[str, Any], Mapping[str, Any], Optional[Tensor], Optional[Tensor]]]:
        if self._cached_data is None:
            return None
        resize_shuffle_r = self._cached_data.get("resize_shuffle_aug_resized", None)
        resize_shuffle_o = self._cached_data.get("resize_shuffle_aug", None)
        if resize_shuffle_r is None and resize_shuffle_o is None:
            return None
        if resize_shuffle_r is None:
            resize_shuffle_r = resize_shuffle_o
        if resize_shuffle_o is None:
            resize_shuffle_o = resize_shuffle_r
        if not isinstance(resize_shuffle_r, dict) or not isinstance(resize_shuffle_o, dict):
            return None
        def _select_image(x: Mapping[str, Any], side: str) -> Optional[Tensor]:
            for key in ("raw", "shift", "resizepaste"):
                value = x.get(key)
                if torch.is_tensor(value):
                    if key != "raw":
                        logger.warning("resize_shuffle %s uses fallback key '%s' (missing raw)", side, key)
                    return value
            return None

        resize_shuffle_r_raw = _select_image(resize_shuffle_r, "resized")
        resize_shuffle_o_raw = _select_image(resize_shuffle_o, "original")
        return resize_shuffle_r, resize_shuffle_o, resize_shuffle_r_raw, resize_shuffle_o_raw

    def _get_shift_perm_idx(
        self, resize_shuffle_r: Mapping[str, Any], resize_shuffle_o: Mapping[str, Any]
    ) -> Optional[Tuple[Tensor, Tensor]]:
        return self._mask_helper.get_shift_perm_idx(
            resize_shuffle_r=resize_shuffle_r,
            resize_shuffle_o=resize_shuffle_o,
            expected_batch=self._batch_meta.get("B"),
        )

    def _get_patchshuffle_sample_mask(
        self,
        *,
        n_tokens: int,
        batch_size: int,
        device: torch.device,
        grid_source: Optional[Tensor],
        allow_create: bool = True,
    ) -> Optional[Tensor]:
        return self._mask_helper.get_patchshuffle_sample_mask(
            runtime_cache=self._resize_shuffle_outputs,
            n_tokens=n_tokens,
            batch_size=batch_size,
            device=device,
            grid_source=grid_source,
            allow_create=allow_create,
        )

    def _apply_patchshuffle_exclusion(self, sample_mask: Tensor, exclude_mask: Optional[Tensor]) -> Tensor:
        return self._mask_helper.apply_patchshuffle_exclusion(sample_mask, exclude_mask)

    def _build_resize_shuffle_teacher_outputs(
        self,
        *,
        resize_shuffle_r: Mapping[str, Any],
        resize_shuffle_o: Mapping[str, Any],
        resize_shuffle_r_raw: Optional[Tensor],
        resize_shuffle_o_raw: Optional[Tensor],
    ) -> Optional[ResizeShuffleTeacherOutputs]:
        patchshuffle_weight = self._get_current_weight("patchshuffle", self.patchshuffle_weight)
        if resize_shuffle_r_raw is None or resize_shuffle_o_raw is None:
            return None

        teacher_resized_out, teacher_original_out = self.teacher.backbone(
            [resize_shuffle_r_raw, resize_shuffle_o_raw], masks=[None, None], is_training=True
        )
        teacher_cls_pre = torch.cat(
            [teacher_resized_out["x_norm_clstoken"], teacher_original_out["x_norm_clstoken"]], dim=0
        )
        teacher_cls_logits_all = self.teacher.dino_head(teacher_cls_pre)
        teacher_cls_logits_resized, teacher_cls_logits_original = teacher_cls_logits_all.chunk(2, dim=0)
        outputs = ResizeShuffleTeacherOutputs(
            raw_cls_logits_resized=teacher_cls_logits_resized,
            raw_cls_logits_original=teacher_cls_logits_original,
        )

        if patchshuffle_weight > 0.0:
            teacher_patch_tokens_resized = teacher_resized_out["x_norm_patchtokens"]
            teacher_patch_tokens_original = teacher_original_out["x_norm_patchtokens"]
            n_tokens = teacher_patch_tokens_resized.shape[1]
            perm_pair = self._get_shift_perm_idx(resize_shuffle_r, resize_shuffle_o)
            if perm_pair is not None:
                perm_idx_resized, perm_idx_original = perm_pair
                sample_mask = self._get_patchshuffle_sample_mask(
                    n_tokens=n_tokens,
                    batch_size=teacher_patch_tokens_resized.shape[0],
                    device=teacher_patch_tokens_resized.device,
                    grid_source=resize_shuffle_r_raw,
                )
                if torch.is_tensor(sample_mask) and sample_mask.any():
                    shift_mask_resized = self._get_resize_shuffle_mask(resize_shuffle_r, "shift_mask")
                    shift_mask_original = self._get_resize_shuffle_mask(resize_shuffle_o, "shift_mask")
                    patch_mask_resized = self._apply_patchshuffle_exclusion(sample_mask, shift_mask_resized)
                    patch_mask_original = self._apply_patchshuffle_exclusion(sample_mask, shift_mask_original)

                    inv_perm_resized = invert_perm_idx(perm_idx_resized)
                    inv_perm_original = invert_perm_idx(perm_idx_original)
                    teacher_patch_tokens_resized = gather_tokens(teacher_patch_tokens_resized, inv_perm_resized)
                    teacher_patch_tokens_original = gather_tokens(teacher_patch_tokens_original, inv_perm_original)

                    if patch_mask_resized.any():
                        teacher_patch_logits_resized = self.teacher.ibot_head(
                            teacher_patch_tokens_resized[patch_mask_resized]
                        )
                        outputs.raw_patch_logits_resized = teacher_patch_logits_resized
                        outputs.patchshuffle_masks_resized = patch_mask_resized
                    if patch_mask_original.any():
                        teacher_patch_logits_original = self.teacher.ibot_head(
                            teacher_patch_tokens_original[patch_mask_original]
                        )
                        outputs.raw_patch_logits_original = teacher_patch_logits_original
                        outputs.patchshuffle_masks_original = patch_mask_original
        return outputs

    def _build_resize_shuffle_student_outputs(
        self,
        *,
        resize_shuffle_r: Mapping[str, Any],
        resize_shuffle_o: Mapping[str, Any],
        student_global: Optional[Mapping[str, Tensor]] = None,
    ) -> Optional[ResizeShuffleStudentOutputs]:
        outputs = ResizeShuffleStudentOutputs()
        expected_b = self._batch_meta.get("B")
        weights = self._get_resize_shuffle_weights()
        bridge_mlp = self._get_bridge_patch_mlp()

        self._add_resize_paste_student_outputs(
            outputs=outputs,
            resize_shuffle_r=resize_shuffle_r,
            resize_shuffle_o=resize_shuffle_o,
            expected_b=expected_b,
            resize_paste_weight=weights["resize_paste"],
        )
        self._add_bridge_global_student_outputs(
            outputs=outputs,
            student_global=student_global,
            bridge_mlp=bridge_mlp,
            bridge_global_weight=weights["bridge_global"],
        )
        self._add_shift_student_outputs(
            outputs=outputs,
            resize_shuffle_r=resize_shuffle_r,
            resize_shuffle_o=resize_shuffle_o,
            expected_b=expected_b,
            patchshuffle_weight=weights["patchshuffle"],
            bridge_patchshuffle_weight=weights["bridge_patchshuffle"],
            bridge_mlp=bridge_mlp,
        )
        has_any_output = any(v is not None for v in vars(outputs).values())
        return outputs if has_any_output else None

    def _add_resize_paste_student_outputs(
        self,
        *,
        outputs: ResizeShuffleStudentOutputs,
        resize_shuffle_r: Mapping[str, Any],
        resize_shuffle_o: Mapping[str, Any],
        expected_b: Optional[int],
        resize_paste_weight: float,
    ) -> None:
        if resize_paste_weight <= 0.0:
            return
        if "resizepaste" not in resize_shuffle_r or "resizepaste" not in resize_shuffle_o:
            return
        resize_pasted = resize_shuffle_r["resizepaste"]
        crop_original = resize_shuffle_o["resizepaste"]
        if not (torch.is_tensor(resize_pasted) and torch.is_tensor(crop_original)):
            return
        if expected_b is not None and (
            resize_pasted.shape[0] != expected_b or crop_original.shape[0] != expected_b
        ):
            logger.warning(
                "resizepaste batch mismatch: got %s and %s, expected %s",
                resize_pasted.shape,
                crop_original.shape,
                expected_b,
            )
            return

        crop_mask_resized = self._get_resize_shuffle_mask(resize_shuffle_r, "resizepaste_mask")
        crop_mask_original = self._get_resize_shuffle_mask(resize_shuffle_o, "resizepaste_mask")
        if self.is_distillation_enabled:
            crop_mask_resized = None
            crop_mask_original = None
        student_resize_pasted_out, student_crop_original_out = self.student.backbone(
            [resize_pasted, crop_original],
            masks=[crop_mask_resized, crop_mask_original],
            is_training=True,
        )
        student_crop_cls_pre = torch.cat(
            [
                student_resize_pasted_out["x_norm_clstoken"],
                student_crop_original_out["x_norm_clstoken"],
            ],
            dim=0,
        )
        student_crop_cls_logits_all = self.student.dino_head(student_crop_cls_pre)
        student_crop_cls_logits_resized, student_crop_cls_logits_original = student_crop_cls_logits_all.chunk(2, dim=0)
        outputs.resize_paste_cls_logits_resized = student_crop_cls_logits_resized
        outputs.resize_paste_cls_logits_original = student_crop_cls_logits_original

    def _add_bridge_global_student_outputs(
        self,
        *,
        outputs: ResizeShuffleStudentOutputs,
        student_global: Optional[Mapping[str, Tensor]],
        bridge_mlp,
        bridge_global_weight: float,
    ) -> None:
        if bridge_global_weight <= 0.0 or bridge_mlp is None:
            return
        if not isinstance(student_global, Mapping):
            return
        global_patch_tokens = student_global.get("patch_pre_head")
        if not (torch.is_tensor(global_patch_tokens) and global_patch_tokens.dim() == 4 and global_patch_tokens.shape[0] >= 2):
            return
        self._add_bridge_logits(
            outputs,
            prefix="bridge_global",
            patch_tokens_resized=global_patch_tokens[0],
            patch_tokens_original=global_patch_tokens[1],
            bridge_mlp=bridge_mlp,
        )

    def _add_shift_student_outputs(
        self,
        *,
        outputs: ResizeShuffleStudentOutputs,
        resize_shuffle_r: Mapping[str, Any],
        resize_shuffle_o: Mapping[str, Any],
        expected_b: Optional[int],
        patchshuffle_weight: float,
        bridge_patchshuffle_weight: float,
        bridge_mlp,
    ) -> None:
        need_shift_forward = patchshuffle_weight > 0.0 or bridge_patchshuffle_weight > 0.0
        if not need_shift_forward:
            return
        if "shift" not in resize_shuffle_r or "shift" not in resize_shuffle_o:
            return
        shift_resized = resize_shuffle_r["shift"]
        shift_original = resize_shuffle_o["shift"]
        if not (torch.is_tensor(shift_resized) and torch.is_tensor(shift_original)):
            return
        if expected_b is not None and (
            shift_resized.shape[0] != expected_b or shift_original.shape[0] != expected_b
        ):
            logger.warning(
                "shift batch mismatch: got %s and %s, expected %s",
                shift_resized.shape,
                shift_original.shape,
                expected_b,
            )
            return

        perm_pair = self._get_shift_perm_idx(resize_shuffle_r, resize_shuffle_o)
        shift_mask_resized = self._get_resize_shuffle_mask(resize_shuffle_r, "shift_mask")
        shift_mask_original = self._get_resize_shuffle_mask(resize_shuffle_o, "shift_mask")
        if self.is_distillation_enabled:
            shift_mask_resized = None
            shift_mask_original = None
        student_shift_resized_out, student_shift_original_out = self.student.backbone(
            [shift_resized, shift_original],
            masks=[shift_mask_resized, shift_mask_original],
            is_training=True,
        )
        student_patch_tokens_resized = student_shift_resized_out["x_norm_patchtokens"]
        student_patch_tokens_original = student_shift_original_out["x_norm_patchtokens"]

        if bridge_patchshuffle_weight > 0.0 and bridge_mlp is not None:
            self._add_bridge_logits(
                outputs,
                prefix="bridge_patchshuffle",
                patch_tokens_resized=student_patch_tokens_resized,
                patch_tokens_original=student_patch_tokens_original,
                bridge_mlp=bridge_mlp,
            )

        if patchshuffle_weight > 0.0 and perm_pair is not None:
            sample_mask = self._get_patchshuffle_sample_mask(
                n_tokens=student_patch_tokens_resized.shape[1],
                batch_size=student_patch_tokens_resized.shape[0],
                device=student_patch_tokens_resized.device,
                grid_source=shift_resized,
                allow_create=False,
            )
            if torch.is_tensor(sample_mask) and sample_mask.any():
                patch_mask_resized = self._apply_patchshuffle_exclusion(sample_mask, shift_mask_resized)
                patch_mask_original = self._apply_patchshuffle_exclusion(sample_mask, shift_mask_original)
                if patch_mask_resized.any():
                    student_patch_logits_resized = self.student.ibot_head(student_patch_tokens_resized[patch_mask_resized])
                    outputs.patchshuffle_patch_logits_resized = student_patch_logits_resized
                    outputs.patchshuffle_masks_resized = patch_mask_resized
                if patch_mask_original.any():
                    student_patch_logits_original = self.student.ibot_head(student_patch_tokens_original[patch_mask_original])
                    outputs.patchshuffle_patch_logits_original = student_patch_logits_original
                    outputs.patchshuffle_masks_original = patch_mask_original

        if patchshuffle_weight > 0.0:
            student_shift_cls_pre = torch.cat(
                [
                    student_shift_resized_out["x_norm_clstoken"],
                    student_shift_original_out["x_norm_clstoken"],
                ],
                dim=0,
            )
            student_shift_cls_logits_all = self.student.dino_head(student_shift_cls_pre)
            student_shift_cls_logits_resized, student_shift_cls_logits_original = student_shift_cls_logits_all.chunk(2, dim=0)
            outputs.patchshuffle_cls_logits_resized = student_shift_cls_logits_resized
            outputs.patchshuffle_cls_logits_original = student_shift_cls_logits_original

    @torch.no_grad()
    def get_teacher_output(self, images, *, upperbound, mask_indices_list, teacher_temp, n_masked_patches_tensor, it=0, logger_freq=0):
        teacher_global = super().get_teacher_output(
            images,
            upperbound=upperbound,
            mask_indices_list=mask_indices_list,
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
            it=it,
            logger_freq=logger_freq,
        )
        weights = self._get_resize_shuffle_weights()
        if not self._has_active_aux_path(weights):
            return teacher_global
        resize_shuffle_inputs = self._get_resize_shuffle_aug_inputs()
        if resize_shuffle_inputs is None:
            return teacher_global
        resize_shuffle_r, resize_shuffle_o, resize_shuffle_r_raw, resize_shuffle_o_raw = resize_shuffle_inputs
        if self._resize_shuffle_outputs is None:
            return teacher_global
        teacher_resize_shuffle = self._build_resize_shuffle_teacher_outputs(
            resize_shuffle_r=resize_shuffle_r,
            resize_shuffle_o=resize_shuffle_o,
            resize_shuffle_r_raw=resize_shuffle_r_raw,
            resize_shuffle_o_raw=resize_shuffle_o_raw,
        )
        if teacher_resize_shuffle is not None:
            self._resize_shuffle_outputs["teacher"] = teacher_resize_shuffle
            teacher_global["resize_shuffle_aug"] = teacher_resize_shuffle
        return teacher_global

    def get_student_output(self, *, global_crops, local_crops, upperbound, masks, mask_indices_list):
        student_global, student_local = super().get_student_output(
            global_crops=global_crops,
            local_crops=local_crops,
            upperbound=upperbound,
            masks=masks,
            mask_indices_list=mask_indices_list,
        )
        weights = self._get_resize_shuffle_weights()
        if not self._has_active_aux_path(weights):
            return student_global, student_local
        resize_shuffle_inputs = self._get_resize_shuffle_aug_inputs()
        if resize_shuffle_inputs is None:
            return student_global, student_local
        resize_shuffle_r, resize_shuffle_o, resize_shuffle_r_raw, resize_shuffle_o_raw = resize_shuffle_inputs
        if self._resize_shuffle_outputs is None:
            return student_global, student_local
        student_resize_shuffle = self._build_resize_shuffle_student_outputs(
            resize_shuffle_r=resize_shuffle_r,
            resize_shuffle_o=resize_shuffle_o,
            student_global=student_global,
        )
        if student_resize_shuffle is not None:
            self._resize_shuffle_outputs["student"] = student_resize_shuffle
            student_global["resize_shuffle_aug"] = student_resize_shuffle
        return student_global, student_local

    def _loss_resize_paste_ce_from_resize_shuffle(
        self,
        *,
        resize_shuffle_student: ResizeShuffleStudentOutputs,
        resize_shuffle_teacher: ResizeShuffleTeacherOutputs,
        iteration: int = 0,
        logger_freq: int = 0,
    ) -> Optional[Tensor]:
        if resize_shuffle_student is None or resize_shuffle_teacher is None:
            return None
        student_cls_logits_resized = resize_shuffle_student.resize_paste_cls_logits_resized
        student_cls_logits_original = resize_shuffle_student.resize_paste_cls_logits_original
        teacher_cls_logits_resized = resize_shuffle_teacher.raw_cls_logits_resized
        teacher_cls_logits_original = resize_shuffle_teacher.raw_cls_logits_original
        if any(
            x is None
            for x in [
                student_cls_logits_resized,
                student_cls_logits_original,
                teacher_cls_logits_resized,
                teacher_cls_logits_original,
            ]
        ):
            return None

        teacher_temp = self._get_teacher_temp()
        teacher_r_targets = self.resize_paste_loss_resized.sinkhorn_knopp_teacher(
            teacher_cls_logits_resized,
            teacher_temp=teacher_temp,
            iteration=iteration,
            logger_freq=logger_freq,
            logger_loss="resize_paste_resized_cls",
        )
        teacher_o_targets = self.resize_paste_loss.sinkhorn_knopp_teacher(
            teacher_cls_logits_original,
            teacher_temp=teacher_temp,
            iteration=iteration,
            logger_freq=logger_freq,
            logger_loss="resize_paste_original_cls",
        )
        loss_resize_paste_resized = self._combine_dino_loss(
            self.resize_paste_loss_resized(
                student_cls_logits_resized.unsqueeze(0), teacher_r_targets.unsqueeze(0)
            )
        )
        loss_resize_paste_original = self._combine_dino_loss(
            self.resize_paste_loss(student_cls_logits_original.unsqueeze(0), teacher_o_targets.unsqueeze(0))
        )
        loss_resize_paste = loss_resize_paste_resized + loss_resize_paste_original
        return loss_resize_paste

    def _loss_patchshuffle_ibot_from_resize_shuffle(
        self,
        *,
        resize_shuffle_student: ResizeShuffleStudentOutputs,
        resize_shuffle_teacher: ResizeShuffleTeacherOutputs,
        teacher_cls_targets: Optional[Tuple[Tensor, Tensor]] = None,
        iteration: int = 0,
        logger_freq: int = 0,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        if resize_shuffle_student is None or resize_shuffle_teacher is None:
            return None

        student_patch_logits_resized = resize_shuffle_student.patchshuffle_patch_logits_resized
        student_patch_logits_original = resize_shuffle_student.patchshuffle_patch_logits_original
        teacher_patch_logits_resized = resize_shuffle_teacher.raw_patch_logits_resized
        teacher_patch_logits_original = resize_shuffle_teacher.raw_patch_logits_original
        patch_masks_resized = resize_shuffle_student.patchshuffle_masks_resized
        patch_masks_original = resize_shuffle_student.patchshuffle_masks_original
        if patch_masks_resized is None:
            patch_masks_resized = resize_shuffle_teacher.patchshuffle_masks_resized
        if patch_masks_original is None:
            patch_masks_original = resize_shuffle_teacher.patchshuffle_masks_original

        teacher_temp = self._get_teacher_temp()
        patch_losses = []
        if torch.is_tensor(patch_masks_resized) and patch_masks_resized.any():
            if student_patch_logits_resized is None or teacher_patch_logits_resized is None:
                return None
            if patch_masks_resized.device != student_patch_logits_resized.device:
                patch_masks_resized = patch_masks_resized.to(device=student_patch_logits_resized.device)
            if teacher_patch_logits_resized.shape[0] == student_patch_logits_resized.shape[0]:
                teacher_r_count = torch.full(
                    (1,),
                    teacher_patch_logits_resized.shape[0],
                    dtype=torch.long,
                    device=teacher_patch_logits_resized.device,
                )
                teacher_r_patch_targets = self.patchshuffle_patch_loss_resized.sinkhorn_knopp_teacher(
                    teacher_patch_logits_resized,
                    teacher_temp=teacher_temp,
                    n_masked_patches_tensor=teacher_r_count,
                    iteration=iteration,
                    logger_freq=logger_freq,
                    logger_loss="raw_patch_resized",
                )
                loss_resized = self._combine_ibot_loss(
                    self.patchshuffle_patch_loss_resized.forward_masked(
                        student_patch_logits_resized,
                        teacher_r_patch_targets,
                        student_masks_flat=patch_masks_resized,
                    )
                )
                patch_losses.append(loss_resized)
            else:
                logger.warning(
                    "patchshuffle resized logits length mismatch: student=%s teacher=%s",
                    student_patch_logits_resized.shape[0],
                    teacher_patch_logits_resized.shape[0],
                )

        if torch.is_tensor(patch_masks_original) and patch_masks_original.any():
            if student_patch_logits_original is None or teacher_patch_logits_original is None:
                return None
            if patch_masks_original.device != student_patch_logits_original.device:
                patch_masks_original = patch_masks_original.to(device=student_patch_logits_original.device)
            if teacher_patch_logits_original.shape[0] == student_patch_logits_original.shape[0]:
                teacher_o_count = torch.full(
                    (1,),
                    teacher_patch_logits_original.shape[0],
                    dtype=torch.long,
                    device=teacher_patch_logits_original.device,
                )
                teacher_o_patch_targets = self.patchshuffle_patch_loss.sinkhorn_knopp_teacher(
                    teacher_patch_logits_original,
                    teacher_temp=teacher_temp,
                    n_masked_patches_tensor=teacher_o_count,
                    iteration=iteration,
                    logger_freq=logger_freq,
                    logger_loss="raw_patch_original",
                )
                loss_original = self._combine_ibot_loss(
                    self.patchshuffle_patch_loss.forward_masked(
                        student_patch_logits_original,
                        teacher_o_patch_targets,
                        student_masks_flat=patch_masks_original,
                    )
                )
                patch_losses.append(loss_original)
            else:
                logger.warning(
                    "patchshuffle original logits length mismatch: student=%s teacher=%s",
                    student_patch_logits_original.shape[0],
                    teacher_patch_logits_original.shape[0],
                )

        patch_loss = None
        if patch_losses:
            patch_loss = sum(patch_losses) / len(patch_losses)

        student_cls_logits_resized = resize_shuffle_student.patchshuffle_cls_logits_resized
        student_cls_logits_original = resize_shuffle_student.patchshuffle_cls_logits_original
        if student_cls_logits_resized is None or student_cls_logits_original is None:
            return None

        if teacher_cls_targets is None:
            teacher_cls_targets = self._build_teacher_cls_targets_for_resize_shuffle(
                resize_shuffle_teacher=resize_shuffle_teacher,
                iteration=iteration,
                logger_freq=logger_freq,
            )
        if teacher_cls_targets is None:
            return None
        teacher_r_cls_targets, teacher_o_cls_targets = teacher_cls_targets

        cls_loss_resized = self._combine_dino_loss(
            self.patchshuffle_cls_loss_resized(
                student_cls_logits_resized.unsqueeze(0), teacher_r_cls_targets.unsqueeze(0)
            )
        )
        cls_loss_original = self._combine_dino_loss(
            self.patchshuffle_cls_loss(student_cls_logits_original.unsqueeze(0), teacher_o_cls_targets.unsqueeze(0))
        )
        cls_loss = cls_loss_resized + cls_loss_original
        if patch_loss is None:
            patch_loss = cls_loss.new_zeros(())
        return patch_loss, cls_loss

    def _resolve_resize_shuffle_aux_outputs(
        self,
        *,
        teacher_global: Mapping[str, Any],
        student_global: Mapping[str, Any],
    ) -> Tuple[Optional[ResizeShuffleStudentOutputs], Optional[ResizeShuffleTeacherOutputs]]:
        student_aux_outputs: Optional[ResizeShuffleStudentOutputs] = None
        teacher_aux_outputs: Optional[ResizeShuffleTeacherOutputs] = None
        if isinstance(student_global, Mapping):
            student_aux_outputs = student_outputs_from_any(student_global.get("resize_shuffle_aug"))
        if isinstance(teacher_global, Mapping):
            teacher_aux_outputs = teacher_outputs_from_any(teacher_global.get("resize_shuffle_aug"))
        if isinstance(self._resize_shuffle_outputs, dict):
            if student_aux_outputs is None:
                student_aux_outputs = student_outputs_from_any(self._resize_shuffle_outputs.get("student"))
            if teacher_aux_outputs is None:
                teacher_aux_outputs = teacher_outputs_from_any(self._resize_shuffle_outputs.get("teacher"))
        return student_aux_outputs, teacher_aux_outputs

    def _accumulate_resize_paste_loss(
        self,
        *,
        loss_acc: Tensor,
        loss_dict: Dict[str, Any],
        student_aux_outputs: Optional[ResizeShuffleStudentOutputs],
        teacher_aux_outputs: Optional[ResizeShuffleTeacherOutputs],
        resize_paste_weight: float,
        iteration: int,
        logger_freq: int,
    ) -> Tensor:
        if resize_paste_weight <= 0.0:
            return loss_acc
        loss = self._loss_resize_paste_ce_from_resize_shuffle(
            resize_shuffle_student=student_aux_outputs,
            resize_shuffle_teacher=teacher_aux_outputs,
            iteration=iteration,
            logger_freq=logger_freq,
        )
        if loss is None:
            return loss_acc
        loss_dict["aug/resize_paste_cls_loss"] = loss
        loss_dict["aug/resize_paste_cls_loss_weight"] = float(resize_paste_weight)
        return loss_acc + (resize_paste_weight * loss)

    def _build_teacher_cls_targets_if_needed(
        self,
        *,
        teacher_aux_outputs: Optional[ResizeShuffleTeacherOutputs],
        patchshuffle_weight: float,
        bridge_global_weight: float,
        bridge_patchshuffle_weight: float,
        iteration: int,
        logger_freq: int,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        need_teacher_cls_targets = (
            patchshuffle_weight > 0.0 or bridge_global_weight > 0.0 or bridge_patchshuffle_weight > 0.0
        )
        if not need_teacher_cls_targets:
            return None
        return self._build_teacher_cls_targets_for_resize_shuffle(
            resize_shuffle_teacher=teacher_aux_outputs,
            iteration=iteration,
            logger_freq=logger_freq,
        )

    def _accumulate_patchshuffle_loss(
        self,
        *,
        loss_acc: Tensor,
        loss_dict: Dict[str, Any],
        student_aux_outputs: Optional[ResizeShuffleStudentOutputs],
        teacher_aux_outputs: Optional[ResizeShuffleTeacherOutputs],
        teacher_cls_targets: Optional[Tuple[Tensor, Tensor]],
        patchshuffle_weight: float,
        patchshuffle_patch_weight: float,
        patchshuffle_cls_weight: float,
        iteration: int,
        logger_freq: int,
    ) -> Tensor:
        if patchshuffle_weight <= 0.0:
            return loss_acc
        loss = self._loss_patchshuffle_ibot_from_resize_shuffle(
            resize_shuffle_student=student_aux_outputs,
            resize_shuffle_teacher=teacher_aux_outputs,
            teacher_cls_targets=teacher_cls_targets,
            iteration=iteration,
            logger_freq=logger_freq,
        )
        if loss is None:
            return loss_acc
        patch_loss, cls_loss = loss
        loss_dict["aug/patchshuffle_patch_loss"] = patch_loss
        loss_dict["aug/patchshuffle_cls_loss"] = cls_loss
        loss_dict["aug/patchshuffle_patch_loss_weight"] = float(patchshuffle_patch_weight)
        loss_dict["aug/patchshuffle_cls_loss_weight"] = float(patchshuffle_cls_weight)
        patchshuffle_total_loss = (patch_loss * patchshuffle_patch_weight) + (cls_loss * patchshuffle_cls_weight)
        loss_dict["aug/patchshuffle_cls_patch"] = patchshuffle_total_loss
        return loss_acc + patchshuffle_total_loss

    def _accumulate_bridge_losses(
        self,
        *,
        loss_acc: Tensor,
        loss_dict: Dict[str, Any],
        student_aux_outputs: Optional[ResizeShuffleStudentOutputs],
        teacher_cls_targets: Optional[Tuple[Tensor, Tensor]],
        ibot_loss_ok: bool,
        bridge_global_weight: float,
        bridge_patchshuffle_weight: float,
    ) -> Tensor:
        if not ibot_loss_ok or teacher_cls_targets is None:
            return loss_acc

        if bridge_global_weight > 0.0:
            bridge_global_loss = self._loss_bridge_branch_from_resize_shuffle(
                student_aux_outputs=student_aux_outputs,
                teacher_cls_targets=teacher_cls_targets,
                branch_prefix="bridge_global",
            )
            if bridge_global_loss is not None:
                bridge_global_total, bridge_global_resized, bridge_global_original = bridge_global_loss
                if not torch.isfinite(bridge_global_total):
                    logger.warning("bridge_global_loss is non-finite; skipping this branch for current iteration.")
                else:
                    loss_dict["aug/bridge_global_loss"] = bridge_global_total
                    if bridge_global_resized is not None and torch.isfinite(bridge_global_resized):
                        loss_dict["aug/bridge_global_resized_loss"] = bridge_global_resized
                    if bridge_global_original is not None and torch.isfinite(bridge_global_original):
                        loss_dict["aug/bridge_global_original_loss"] = bridge_global_original
                    loss_dict["aug/bridge_global_weight"] = float(bridge_global_weight)
                    loss_acc = loss_acc + (bridge_global_weight * bridge_global_total)

        if bridge_patchshuffle_weight > 0.0:
            bridge_patchshuffle_loss = self._loss_bridge_branch_from_resize_shuffle(
                student_aux_outputs=student_aux_outputs,
                teacher_cls_targets=teacher_cls_targets,
                branch_prefix="bridge_patchshuffle",
            )
            if bridge_patchshuffle_loss is not None:
                (
                    bridge_patchshuffle_total,
                    bridge_patchshuffle_resized,
                    bridge_patchshuffle_original,
                ) = bridge_patchshuffle_loss
                if not torch.isfinite(bridge_patchshuffle_total):
                    logger.warning("bridge_patchshuffle_loss is non-finite; skipping this branch for current iteration.")
                else:
                    loss_dict["aug/bridge_patchshuffle_loss"] = bridge_patchshuffle_total * 0.5
                    if bridge_patchshuffle_resized is not None and torch.isfinite(bridge_patchshuffle_resized):
                        loss_dict["aug/bridge_patchshuffle_resized_loss"] = bridge_patchshuffle_resized
                    if bridge_patchshuffle_original is not None and torch.isfinite(bridge_patchshuffle_original):
                        loss_dict["aug/bridge_patchshuffle_original_loss"] = bridge_patchshuffle_original
                    loss_dict["aug/bridge_patchshuffle_weight"] = float(bridge_patchshuffle_weight)
                    loss_acc = loss_acc + (bridge_patchshuffle_weight * bridge_patchshuffle_total * 0.5)

        return loss_acc

    def compute_losses(
        self,
        *,
        teacher_global,
        student_global,
        student_local,
        gram_global,
        masks,
        mask_indices_list,
        masks_weight,
        iteration,
        logger_freq=0,
    ):
        loss_acc, loss_dict = super().compute_losses(
            teacher_global=teacher_global,
            student_global=student_global,
            student_local=student_local,
            gram_global=gram_global,
            masks=masks,
            mask_indices_list=mask_indices_list,
            masks_weight=masks_weight,
            iteration=iteration,
            logger_freq=logger_freq,
        )

        if self._cached_data is None:
            return loss_acc, loss_dict

        ibot_loss = loss_dict.get("ibot_loss")
        ibot_loss_ok = ibot_loss is not None and float(ibot_loss.detach().item()) < 4.0

        weights = self._get_resize_shuffle_weights()
        student_aux_outputs, teacher_aux_outputs = self._resolve_resize_shuffle_aux_outputs(
            teacher_global=teacher_global,
            student_global=student_global,
        )

        loss_acc = self._accumulate_resize_paste_loss(
            loss_acc=loss_acc,
            loss_dict=loss_dict,
            student_aux_outputs=student_aux_outputs,
            teacher_aux_outputs=teacher_aux_outputs,
            resize_paste_weight=weights["resize_paste"],
            iteration=iteration,
            logger_freq=logger_freq,
        )

        teacher_cls_targets = self._build_teacher_cls_targets_if_needed(
            teacher_aux_outputs=teacher_aux_outputs,
            patchshuffle_weight=weights["patchshuffle"],
            bridge_global_weight=weights["bridge_global"],
            bridge_patchshuffle_weight=weights["bridge_patchshuffle"],
            iteration=iteration,
            logger_freq=logger_freq,
        )

        loss_acc = self._accumulate_patchshuffle_loss(
            loss_acc=loss_acc,
            loss_dict=loss_dict,
            student_aux_outputs=student_aux_outputs,
            teacher_aux_outputs=teacher_aux_outputs,
            teacher_cls_targets=teacher_cls_targets,
            patchshuffle_weight=weights["patchshuffle"],
            patchshuffle_patch_weight=weights["patchshuffle_patch"],
            patchshuffle_cls_weight=weights["patchshuffle_cls"],
            iteration=iteration,
            logger_freq=logger_freq,
        )

        loss_acc = self._accumulate_bridge_losses(
            loss_acc=loss_acc,
            loss_dict=loss_dict,
            student_aux_outputs=student_aux_outputs,
            teacher_cls_targets=teacher_cls_targets,
            ibot_loss_ok=ibot_loss_ok,
            bridge_global_weight=weights["bridge_global"],
            bridge_patchshuffle_weight=weights["bridge_patchshuffle"],
        )

        return loss_acc, loss_dict
