from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from torch import Tensor


@dataclass
class ResizeShuffleTeacherOutputs:
    raw_cls_logits_resized: Tensor
    raw_cls_logits_original: Tensor
    raw_patch_logits_resized: Optional[Tensor] = None
    raw_patch_logits_original: Optional[Tensor] = None
    patchshuffle_masks_resized: Optional[Tensor] = None
    patchshuffle_masks_original: Optional[Tensor] = None


@dataclass
class ResizeShuffleStudentOutputs:
    resize_paste_cls_logits_resized: Optional[Tensor] = None
    resize_paste_cls_logits_original: Optional[Tensor] = None

    patchshuffle_patch_logits_resized: Optional[Tensor] = None
    patchshuffle_patch_logits_original: Optional[Tensor] = None
    patchshuffle_masks_resized: Optional[Tensor] = None
    patchshuffle_masks_original: Optional[Tensor] = None

    patchshuffle_cls_logits_resized: Optional[Tensor] = None
    patchshuffle_cls_logits_original: Optional[Tensor] = None

    bridge_global_logits_resized: Optional[Tensor] = None
    bridge_global_logits_original: Optional[Tensor] = None
    bridge_patchshuffle_logits_resized: Optional[Tensor] = None
    bridge_patchshuffle_logits_original: Optional[Tensor] = None


def teacher_outputs_from_any(value) -> Optional[ResizeShuffleTeacherOutputs]:
    if value is None:
        return None
    if isinstance(value, ResizeShuffleTeacherOutputs):
        return value
    if isinstance(value, Mapping):
        cls_resized = value.get("raw_cls_logits_resized")
        cls_original = value.get("raw_cls_logits_original")
        if cls_resized is None or cls_original is None:
            return None
        return ResizeShuffleTeacherOutputs(
            raw_cls_logits_resized=cls_resized,
            raw_cls_logits_original=cls_original,
            raw_patch_logits_resized=value.get("raw_patch_logits_resized"),
            raw_patch_logits_original=value.get("raw_patch_logits_original"),
            patchshuffle_masks_resized=value.get("patchshuffle_masks_resized"),
            patchshuffle_masks_original=value.get("patchshuffle_masks_original"),
        )
    return None


def student_outputs_from_any(value) -> Optional[ResizeShuffleStudentOutputs]:
    if value is None:
        return None
    if isinstance(value, ResizeShuffleStudentOutputs):
        return value
    if isinstance(value, Mapping):
        return ResizeShuffleStudentOutputs(
            resize_paste_cls_logits_resized=value.get("resize_paste_cls_logits_resized"),
            resize_paste_cls_logits_original=value.get("resize_paste_cls_logits_original"),
            patchshuffle_patch_logits_resized=value.get("patchshuffle_patch_logits_resized"),
            patchshuffle_patch_logits_original=value.get("patchshuffle_patch_logits_original"),
            patchshuffle_masks_resized=value.get("patchshuffle_masks_resized"),
            patchshuffle_masks_original=value.get("patchshuffle_masks_original"),
            patchshuffle_cls_logits_resized=value.get("patchshuffle_cls_logits_resized"),
            patchshuffle_cls_logits_original=value.get("patchshuffle_cls_logits_original"),
            bridge_global_logits_resized=value.get("bridge_global_logits_resized"),
            bridge_global_logits_original=value.get("bridge_global_logits_original"),
            bridge_patchshuffle_logits_resized=value.get("bridge_patchshuffle_logits_resized"),
            bridge_patchshuffle_logits_original=value.get("bridge_patchshuffle_logits_original"),
        )
    return None
