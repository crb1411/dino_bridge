from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.distributed as dist

from dinov3.configs import DinoV3SetupArgs, setup_config
from dinov3.models import build_model_for_eval

logger = logging.getLogger("dinov3.new_train.eval")


def _maybe_init_dist_for_dcp(ckpt_path: Path) -> None:
    if not ckpt_path.is_dir():
        return
    if not dist.is_available() or dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=backend, init_method="env://")
        return
    store = tempfile.NamedTemporaryFile(prefix="dinov3_dist_", suffix=".tmp", delete=False)
    store.close()
    dist.init_process_group(
        backend=backend,
        init_method=f"file://{store.name}",
        rank=0,
        world_size=1,
    )


def load_eval_model(
    config_file: str | Path,
    checkpoint_path: str | Path,
    *,
    opts: Optional[Iterable[str]] = None,
    init_distributed_if_needed: bool = True,
) -> torch.nn.Module:
    """
    Load a teacher backbone for eval from either a consolidated checkpoint or DCP dir.
    """
    config_path = Path(config_file).expanduser().resolve()
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    if init_distributed_if_needed:
        _maybe_init_dist_for_dcp(ckpt_path)
    setup_args = DinoV3SetupArgs(
        config_file=str(config_path),
        pretrained_weights=str(ckpt_path),
        output_dir=None,
        opts=list(opts or []),
    )
    cfg = setup_config(setup_args, strict_cfg=False)
    model = build_model_for_eval(cfg, setup_args.pretrained_weights)
    return model
