import logging
import math
import random
from typing import Optional

import numpy as np
import torch
from torch import nn
from torchvision.transforms import v2
from PIL import Image
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from dinov3.data.masking import MaskingGenerator
from dinov3.data.transforms import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    GaussianBlur,
    make_normalize_transform,
)
from dinov3.data.resize_shuffle.augmentor import Augmentor, AugmentSwitch

logger = logging.getLogger("dinov3")


def _build_resize_shuffle_augmentor(
    use_resize_shuffle_augmentor: bool,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    switch: Optional[AugmentSwitch] = None,
) -> Optional[Augmentor]:
    if not use_resize_shuffle_augmentor:
        return None
    return Augmentor(
        switches=switch or AugmentSwitch(),
        mean=mean,
        std=std,
        use_normalize=True,
    )


def _image_to_tensor01(image) -> torch.Tensor:
    """PIL RGB or torch Tensor -> float32 [C,H,W], in [0,1]."""
    if isinstance(image, torch.Tensor):
        x = image
        if x.dtype != torch.float32:
            x = v2.functional.to_dtype(x, torch.float32, scale=True)
        return x
    x = v2.functional.pil_to_tensor(image)
    x = v2.functional.to_dtype(x, torch.float32, scale=True)
    return x


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        gram_teacher_crops_size=None,
        gram_teacher_no_distortions=False,
        teacher_no_color_jitter=False,
        local_crops_subset_of_global_crops=False,
        patch_size=16,
        share_color_jitter=False,
        horizontal_flips=True,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        use_resize_shuffle_augmentor: bool = False,
        resize_shuffle_augmentor_switch: Optional[AugmentSwitch] = None,
    ):
        self.use_resize_shuffle_augmentor = use_resize_shuffle_augmentor
        self.resize_shuffle_augmentor_switch = resize_shuffle_augmentor_switch
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.teacher_no_color_jitter = teacher_no_color_jitter
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size = patch_size
        self.share_color_jitter = share_color_jitter
        self.mean = mean
        self.std = std
        self._resize_shuffle_augmentor = _build_resize_shuffle_augmentor(
            self.use_resize_shuffle_augmentor,
            mean,
            std,
            switch=self.resize_shuffle_augmentor_switch,
        )

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info("global_crops_scale: %s", global_crops_scale)
        logger.info("local_crops_scale: %s", local_crops_scale)
        logger.info("local_crops_number: %s", local_crops_number)
        logger.info("global_crops_size: %s", global_crops_size)
        logger.info("local_crops_size: %s", local_crops_size)
        logger.info("gram_crops_size: %s", gram_teacher_crops_size)
        logger.info("gram_teacher_no_distortions: %s", gram_teacher_no_distortions)
        logger.info("teacher_no_color_jitter: %s", teacher_no_color_jitter)
        logger.info("local_crops_subset_of_global_crops: %s", local_crops_subset_of_global_crops)
        logger.info("patch_size if local_crops_subset_of_global_crops: %s", patch_size)
        logger.info("share_color_jitter: %s", share_color_jitter)
        logger.info("horizontal flips: %s", horizontal_flips)
        logger.info("###################################")

        global_crop_max_size = max(global_crops_size, gram_teacher_crops_size if gram_teacher_crops_size else 0)
        self.global_crop_max_size = global_crop_max_size
        # center crop probability
        self.resize_shuffle_resize_prob = 0.5
        self.resize_shuffle_raw_direct = v2.Resize(
            (global_crop_max_size, global_crop_max_size),
            interpolation=v2.InterpolationMode.BILINEAR,
        )
        self.resize_shuffle_raw_shorter = v2.Resize(
            global_crop_max_size,
            interpolation=v2.InterpolationMode.BILINEAR,
        )
        self.resize_shuffle_raw_center_crop = v2.CenterCrop(global_crop_max_size)

        # random resized crop and flip
        self.geometric_augmentation_global = v2.Compose(
            [
                v2.RandomResizedCrop(
                    global_crop_max_size,
                    scale=global_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        resize_global = nn.Identity()
        self.resize_global_post_transf = nn.Identity()
        self.resize_gram_teacher = None
        if gram_teacher_crops_size is not None:
            if gram_teacher_no_distortions:
                resize_global = v2.Resize(
                    global_crops_size,
                    interpolation=v2.InterpolationMode.BICUBIC,
                )
            else:
                self.resize_global_post_transf = v2.Resize(
                    global_crops_size,
                    interpolation=v2.InterpolationMode.BICUBIC,
                )

            self.resize_gram_teacher = v2.Resize(
                gram_teacher_crops_size,
                interpolation=v2.InterpolationMode.BICUBIC,
            )

        self.geometric_augmentation_local = v2.Compose(
            [
                v2.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        color_jittering = v2.Compose(
            [
                v2.RandomApply(
                    [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                v2.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)
        global_transfo2_extra = v2.Compose(
            [
                GaussianBlur(p=0.1),
                v2.RandomSolarize(threshold=128, p=0.2),
            ]
        )
        local_transfo_extra = GaussianBlur(p=0.5)

        self.normalize = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                make_normalize_transform(mean=mean, std=std),
            ]
        )

        if self.share_color_jitter:
            self.color_jittering = color_jittering
            self.global_transfo1 = v2.Compose([resize_global, global_transfo1_extra, self.normalize])
            self.global_transfo2 = v2.Compose([resize_global, global_transfo2_extra, self.normalize])
            self.local_transfo = v2.Compose([local_transfo_extra, self.normalize])
            self.global_transfo1_nonorm = v2.Compose([resize_global, global_transfo1_extra])
            self.global_transfo2_nonorm = v2.Compose([resize_global, global_transfo2_extra])
            self.local_transfo_nonorm = v2.Compose([local_transfo_extra])
        else:
            self.global_transfo1 = v2.Compose(
                [resize_global, color_jittering, global_transfo1_extra, self.normalize]
            )
            self.global_transfo2 = v2.Compose(
                [resize_global, color_jittering, global_transfo2_extra, self.normalize]
            )
            self.local_transfo = v2.Compose([color_jittering, local_transfo_extra, self.normalize])
            self.global_transfo1_nonorm = v2.Compose([resize_global, color_jittering, global_transfo1_extra])
            self.global_transfo2_nonorm = v2.Compose([resize_global, color_jittering, global_transfo2_extra])
            self.local_transfo_nonorm = v2.Compose([color_jittering, local_transfo_extra])

        self.to_img = v2.ToImage()

    def __call__(self, image):
        image = self.to_img(image) if not isinstance(image, Image.Image) else image

        output = {}
        output["weak_flag"] = True

        if self.share_color_jitter:
            image = self.color_jittering(image)

        # global crops
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1_transf = self.global_transfo1(im1_base)
        global_crop_1 = self.resize_global_post_transf(global_crop_1_transf)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2_transf = self.global_transfo2(im2_base)
        global_crop_2 = self.resize_global_post_transf(global_crop_2_transf)

        output["global_crops"] = [global_crop_1, global_crop_2]

        if self.teacher_no_color_jitter:
            output["global_crops_teacher"] = [
                self.normalize(im1_base),
                self.normalize(im2_base),
            ]
        else:
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        if self.gram_teacher_crops_size is not None:
            if self.gram_teacher_no_distortions:
                gram_crop_1 = self.normalize(self.resize_gram_teacher(im1_base))
                gram_crop_2 = self.normalize(self.resize_gram_teacher(im2_base))
            else:
                gram_crop_1 = self.resize_gram_teacher(global_crop_1_transf)
                gram_crop_2 = self.resize_gram_teacher(global_crop_2_transf)
            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        # local crops
        if self.local_crops_subset_of_global_crops:
            _local_crops = [self.local_transfo(im1_base) for _ in range(self.local_crops_number // 2)] + [
                self.local_transfo(im2_base) for _ in range(self.local_crops_number // 2)
            ]

            local_crops = []
            offsets = []
            gs = self.global_crops_size
            ls = self.local_crops_size
            for img in _local_crops:
                rx, ry = np.random.randint(0, (gs - ls) // self.patch_size, 2) * self.patch_size
                local_crops.append(img[:, rx : rx + ls, ry : ry + ls])
                offsets.append((rx, ry))

            output["local_crops"] = local_crops
            output["offsets"] = offsets
        else:
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
            ]
            output["local_crops"] = local_crops
            output["offsets"] = ()

        if self._resize_shuffle_augmentor is not None:
            try:
                if np.random.rand() < 0.3:
                    im1_base_, im2_base_ = im2_base, im1_base
                else:
                    im1_base_ = self.resize_global_post_transf(self.global_transfo1_nonorm(im1_base))
                    im2_base_ = self.resize_global_post_transf(self.global_transfo2_nonorm(im2_base))

                img_ = im1_base_ if np.random.rand() < 0.5 else im2_base_
                tensor_input_resized = _image_to_tensor01(img_)

                h, w = v2.functional.get_size(image)
                image_raw = image
                if h != self.global_crop_max_size or w != self.global_crop_max_size:
                    image_raw = self._resize_resize_shuffle_raw(image)
                tensor_input_raw = _image_to_tensor01(image_raw)

                output["resize_shuffle_aug"] = self._resize_shuffle_augmentor(tensor_input_raw)
                output["resize_shuffle_aug_resized"] = self._resize_shuffle_augmentor(tensor_input_resized)
            except Exception as exc:
                logger.warning("resize_shuffle augmentation failed: %s", exc)

        return output

    def _resize_resize_shuffle_raw(self, image):
        if torch.rand(()) > self.resize_shuffle_resize_prob:
            return self.resize_shuffle_raw_direct(image)
        resized = self.resize_shuffle_raw_shorter(image)
        return self.resize_shuffle_raw_center_crop(resized)


def _make_mask_generator_for_grid(mask_generator: MaskingGenerator, grid_hw: tuple[int, int]) -> MaskingGenerator:
    base_h, base_w = mask_generator.get_shape()
    if (base_h, base_w) == grid_hw:
        return mask_generator
    base_n = base_h * base_w
    target_n = grid_hw[0] * grid_hw[1]
    if base_n <= 0 or target_n <= 0:
        return mask_generator
    max_num_patches = None
    if mask_generator.max_num_patches is not None:
        ratio = mask_generator.max_num_patches / base_n
        max_num_patches = ratio * target_n
    min_num_patches = mask_generator.min_num_patches
    if max_num_patches is not None:
        min_num_patches = min(min_num_patches, max_num_patches)
    min_aspect = math.exp(mask_generator.log_aspect_ratio[0])
    max_aspect = math.exp(mask_generator.log_aspect_ratio[1])
    return MaskingGenerator(
        input_size=grid_hw,
        max_num_patches=max_num_patches,
        min_num_patches=min_num_patches,
        min_aspect=min_aspect,
        max_aspect=max_aspect,
    )


def _build_collated_masks(
    *,
    batch_size: int,
    n_tokens: int,
    mask_ratio_tuple,
    mask_probability,
    mask_generator,
    random_circular_shift,
):
    if batch_size <= 0 or n_tokens is None or n_tokens <= 0 or mask_generator is None:
        return None
    n_samples_masked = int(batch_size * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_max = probs[i + 1]
        mask = torch.BoolTensor(mask_generator(int(n_tokens * prob_max)))
        if random_circular_shift:
            shift_x, shift_y = (
                random.randint(0, mask.shape[0] - 1),
                random.randint(0, mask.shape[1] - 1),
            )
            mask = torch.roll(mask, (shift_x, shift_y), (0, 1))
        masks_list.append(mask)
    for _ in range(n_samples_masked, batch_size):
        masks_list.append(torch.BoolTensor(mask_generator(0)))
    random.shuffle(masks_list)
    return torch.stack(masks_list).flatten(1)


def _attach_resize_shuffle_masks(
    resize_shuffle_aug_batch: dict,
    *,
    patch_hw: tuple[int, int] | None,
    mask_ratio_tuple,
    mask_probability,
    mask_generator,
    random_circular_shift,
):
    if not isinstance(resize_shuffle_aug_batch, dict) or patch_hw is None or mask_generator is None:
        return
    patch_h, patch_w = patch_hw
    if patch_h <= 0 or patch_w <= 0:
        return
    for key in ("resizepaste", "shift"):
        value = resize_shuffle_aug_batch.get(key)
        if not torch.is_tensor(value):
            continue
        H = int(value.shape[-2])
        W = int(value.shape[-1])
        if H % patch_h != 0 or W % patch_w != 0:
            continue
        grid_hw = (H // patch_h, W // patch_w)
        gen = _make_mask_generator_for_grid(mask_generator, grid_hw)
        masks = _build_collated_masks(
            batch_size=value.shape[0],
            n_tokens=grid_hw[0] * grid_hw[1],
            mask_ratio_tuple=mask_ratio_tuple,
            mask_probability=mask_probability,
            mask_generator=gen,
            random_circular_shift=random_circular_shift,
        )
        if masks is not None:
            resize_shuffle_aug_batch[f"{key}_mask"] = masks


def collate_data_and_cast(
    samples_list,
    mask_ratio_tuple,
    mask_probability,
    dtype,
    n_tokens=None,
    mask_generator=None,
    random_circular_shift=False,
    local_batch_size=None,
):
    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack(
        [s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list]
    )
    collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])
    if "gram_teacher_crops" in samples_list[0][0]:
        collated_gram_teacher_crops = torch.stack(
            [s[0]["gram_teacher_crops"][i] for i in range(n_global_crops) for s in samples_list]
        )
    else:
        collated_gram_teacher_crops = None

    if local_batch_size is not None:
        B = n_global_crops * local_batch_size
    else:
        B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_max = probs[i + 1]
        mask = torch.BoolTensor(mask_generator(int(N * prob_max)))
        if random_circular_shift:
            shift_x, shift_y = (
                random.randint(0, mask.shape[0] - 1),
                random.randint(0, mask.shape[1] - 1),
            )
            mask = torch.roll(mask, (shift_x, shift_y), (0, 1))
        masks_list.append(mask)
        upperbound += int(N * prob_max)
    for _ in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    out = {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
    if "resize_shuffle_aug" in samples_list[0][0]:
        collate_resize_shuffle_aug(samples_list=samples_list, out=out, dtype=dtype, key_="resize_shuffle_aug_resized")
        collate_resize_shuffle_aug(samples_list=samples_list, out=out, dtype=dtype, key_="resize_shuffle_aug")
        patch_hw = None
        if mask_generator is not None:
            grid_h, grid_w = mask_generator.get_shape()
            if grid_h > 0 and grid_w > 0:
                g_h = collated_global_crops.shape[-2]
                g_w = collated_global_crops.shape[-1]
                if g_h % grid_h == 0 and g_w % grid_w == 0:
                    patch_hw = (g_h // grid_h, g_w // grid_w)
        _attach_resize_shuffle_masks(
            out.get("resize_shuffle_aug_resized", {}),
            patch_hw=patch_hw,
            mask_ratio_tuple=mask_ratio_tuple,
            mask_probability=mask_probability,
            mask_generator=mask_generator,
            random_circular_shift=random_circular_shift,
        )
        _attach_resize_shuffle_masks(
            out.get("resize_shuffle_aug", {}),
            patch_hw=patch_hw,
            mask_ratio_tuple=mask_ratio_tuple,
            mask_probability=mask_probability,
            mask_generator=mask_generator,
            random_circular_shift=random_circular_shift,
        )
    if collated_gram_teacher_crops is not None:
        out["collated_gram_teacher_crops"] = collated_gram_teacher_crops.to(dtype)
    return out


def _pad_1d_tensors(values, *, pad_value=-1, dtype=None):
    if not values:
        return torch.empty((0, 0), dtype=torch.long)
    max_len = 0
    ref = None
    for v in values:
        if torch.is_tensor(v):
            if ref is None:
                ref = v
            max_len = max(max_len, int(v.numel()))
    if dtype is None:
        dtype = ref.dtype if ref is not None else torch.long
    device = ref.device if ref is not None else None
    if device is None:
        out = torch.full((len(values), max_len), pad_value, dtype=dtype)
    else:
        out = torch.full((len(values), max_len), pad_value, dtype=dtype, device=device)
    for i, v in enumerate(values):
        if not torch.is_tensor(v):
            continue
        flat = v.reshape(-1).to(dtype=dtype, device=out.device)
        if flat.numel() == 0:
            continue
        out[i, : flat.numel()] = flat
    return out


def collate_resize_shuffle_aug(samples_list, out, dtype=torch.float32, key_="resize_shuffle_aug_resized"):
    first_aug = samples_list[0][0].get(key_, None)
    if first_aug is None:
        return

    aug_dicts = [s[0].get(key_, None) for s in samples_list]

    all_keys = set()
    for d in aug_dicts:
        if isinstance(d, dict):
            all_keys.update(d.keys())

    resize_shuffle_aug_batch = {}

    for key in all_keys:
        values = []
        valid = True
        for d in aug_dicts:
            if not isinstance(d, dict) or key not in d:
                valid = False
                break
            values.append(d[key])
        if not valid:
            continue

        v0 = values[0]

        if torch.is_tensor(v0):
            resize_shuffle_aug_batch[key] = torch.stack([v.to(dtype) for v in values], dim=0)
            continue

        if isinstance(v0, dict) and key.endswith("_info"):
            if key == "shift_info":
                perm_idx = [v.get("perm_idx") if isinstance(v, dict) else None for v in values]
                resize_shuffle_aug_batch[key] = _pad_1d_tensors(perm_idx, pad_value=-1, dtype=torch.long)
                continue
            if key == "resizepaste_info":
                uncovered = [v.get("uncovered_idx") if isinstance(v, dict) else None for v in values]
                resize_shuffle_aug_batch[key] = _pad_1d_tensors(uncovered, pad_value=-1, dtype=torch.long)
                continue

        resize_shuffle_aug_batch[key] = torch.empty((len(values), 0), dtype=torch.long)

    out[key_] = resize_shuffle_aug_batch




if __name__ == "__main__":
    from functools import partial
    from pathlib import Path

    from omegaconf import OmegaConf
    from tqdm import tqdm

    from dinov3.data.resize_shuffle import create_dir_time, save_index
    from dinov3.data.masking import MaskingGenerator
    from dinov3.new_train.data.imgnet import ImageNetResizeDataset

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "ssl_default_config.yaml"
    cfg = OmegaConf.load(str(cfg_path))

    mask_generator = MaskingGenerator(
        input_size=(14, 14),
        max_num_patches=0.5 * 196,
    )
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        dtype={
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[cfg.compute_precision.param_dtype],
        n_tokens=196,
        mask_generator=mask_generator,
        random_circular_shift=cfg.ibot.mask_random_circular_shift,
        local_batch_size=None,
    )

    aug = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
        gram_teacher_crops_size=cfg.crops.gram_teacher_crops_size,
        gram_teacher_no_distortions=cfg.crops.gram_teacher_no_distortions,
        local_crops_subset_of_global_crops=cfg.crops.localcrops_subset_of_globalcrops,
        share_color_jitter=cfg.crops.share_color_jitter,
        horizontal_flips=cfg.crops.horizontal_flips,
        mean=cfg.crops.rgb_mean,
        std=cfg.crops.rgb_std,
        use_resize_shuffle_augmentor=True,
        resize_shuffle_augmentor_switch=None,
    )

    data = ImageNetResizeDataset(
        image_list_txt="/root/data/imagenet_1k/data_new/imagenet_1k.txt",
        size=224,
        transform=aug,
    )

    save_max, save_ = 1, 0
    for idx, (data_idx, _) in enumerate(data):
        save_ += 1
        out_dir = create_dir_time(
            base_dir="/mnt/work/output_dir/aug_out_110_2",
            prefix=f"{idx}_aug",
        )
        save_index(data_idx, out_dir, prefix="augVis")
        if save_ > save_max:
            break

    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=8,
        num_workers=2,
        pin_memory=False,
        drop_last=True,
        persistent_workers=False,
        collate_fn=collate_fn,
    )
    num = 10
    i = 0
    for data_ in tqdm(dataloader):
        i += 1
        if i > num:
            break
        _ = data_
