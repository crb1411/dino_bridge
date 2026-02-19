#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _select_eval_checkpoint(root: Path, which: str, eval_step: int | None) -> Path:
    eval_dir = root / "eval"
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"eval directory not found: {eval_dir}")
    steps = sorted([int(p.name) for p in eval_dir.iterdir() if p.is_dir() and p.name.isdigit()])
    if not steps:
        raise FileNotFoundError(f"no eval steps found under: {eval_dir}")
    step = eval_step if eval_step is not None else steps[-1]
    step_dir = eval_dir / str(step)
    if not step_dir.is_dir():
        raise FileNotFoundError(f"eval step directory not found: {step_dir}")
    ckpt_path = step_dir / f"{which}_checkpoint.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    return ckpt_path


def _load_state_dict(ckpt_path: Path) -> dict:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required to load checkpoints.") from exc
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("teacher", "student", "model", "state_dict"):
            if key in ckpt:
                return ckpt[key]
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")
    return ckpt


def _normalize_key(k: str) -> str:
    k = k.replace("._orig_mod.", ".")
    prefixes = ("module.", "backbone.", "student.", "teacher.", "model_ema.")
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p) :]
                changed = True
    return k


def _filter_backbone_state_dict(state_dict: dict) -> dict:
    allowed_prefixes = (
        "patch_embed.",
        "cls_token",
        "storage_tokens",
        "mask_token",
        "rope_embed.",
        "blocks.",
        "norm.",
        "cls_norm.",
        "local_cls_norm.",
    )
    out = {}
    for k, v in state_dict.items():
        nk = _normalize_key(k)
        if nk.startswith(allowed_prefixes) or nk in ("cls_token", "storage_tokens", "mask_token"):
            out[nk] = v
    if not out:
        raise RuntimeError("Backbone state_dict appears empty after filtering.")
    return out


def _infer_num_layers(state_dict: dict) -> int:
    block_ids = set()
    for k in state_dict.keys():
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_ids.add(int(parts[1]))
    if not block_ids:
        raise RuntimeError("No transformer blocks found in state_dict.")
    return max(block_ids) + 1


def _infer_hidden_size(state_dict: dict) -> int:
    if "cls_token" in state_dict:
        return state_dict["cls_token"].shape[-1]
    for k in ("patch_embed.proj.weight", "patch_embed.proj.bias"):
        if k in state_dict:
            return state_dict[k].shape[0]
    raise RuntimeError("Unable to infer hidden size from state_dict.")


def _infer_num_heads(state_dict: dict, hidden_size: int, fallback: int | None) -> int:
    if "rope_embed.periods" in state_dict:
        periods = state_dict["rope_embed.periods"]
        d_head = int(periods.numel() * 4)
        if d_head > 0 and hidden_size % d_head == 0:
            return hidden_size // d_head
    if fallback is not None:
        return fallback
    raise RuntimeError("Unable to infer num_heads (rope periods missing and no fallback).")


def _infer_intermediate_size(state_dict: dict) -> int:
    for k in ("blocks.0.mlp.w1.weight", "blocks.0.mlp.fc1.weight"):
        if k in state_dict:
            return state_dict[k].shape[0]
    raise RuntimeError("Unable to infer intermediate_size from state_dict.")


def _arch_fallbacks(arch: str) -> dict:
    arch_map = {
        "vit_base": dict(hidden_size=768, num_layers=12, num_heads=12, ffn_ratio=4.0),
        "vit_large": dict(hidden_size=1024, num_layers=24, num_heads=16, ffn_ratio=4.0),
        "vit_so400m": dict(hidden_size=1152, num_layers=27, num_heads=18, ffn_ratio=3.777777778),
        "vit_huge2": dict(hidden_size=1280, num_layers=32, num_heads=20, ffn_ratio=4.0),
        "vit_giant2": dict(hidden_size=1536, num_layers=40, num_heads=24, ffn_ratio=4.0),
    }
    return arch_map.get(arch, {})


def _make_config(cfg: dict, state_dict: dict):
    try:
        from transformers import DINOv3ViTConfig
    except Exception as exc:
        raise RuntimeError("transformers is required. Install with: pip install transformers") from exc

    student_cfg = cfg.get("student", {})
    crops_cfg = cfg.get("crops", {})
    arch = student_cfg.get("arch", "")
    fallback = _arch_fallbacks(arch)

    hidden_size = _infer_hidden_size(state_dict)
    num_layers = _infer_num_layers(state_dict)
    num_heads = _infer_num_heads(state_dict, hidden_size, fallback.get("num_heads"))
    intermediate_size = _infer_intermediate_size(state_dict)

    use_gated_mlp = any(k.startswith("blocks.0.mlp.w1") for k in state_dict.keys())

    config = DINOv3ViTConfig(
        patch_size=int(student_cfg.get("patch_size", 16)),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        num_hidden_layers=int(num_layers),
        num_attention_heads=int(num_heads),
        hidden_act="silu" if use_gated_mlp else "gelu",
        attention_dropout=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        rope_theta=float(student_cfg.get("pos_embed_rope_base", 100.0)),
        image_size=int(crops_cfg.get("global_crops_size", 224)),
        num_channels=int(student_cfg.get("in_chans", 3)),
        query_bias=bool(student_cfg.get("qkv_bias", True)),
        key_bias=True,
        value_bias=bool(student_cfg.get("qkv_bias", True)),
        proj_bias=bool(student_cfg.get("proj_bias", True)),
        mlp_bias=bool(student_cfg.get("ffn_bias", True)),
        layerscale_value=float(student_cfg.get("layerscale", 1.0)),
        drop_path_rate=float(student_cfg.get("drop_path_rate", 0.0)),
        use_gated_mlp=use_gated_mlp,
        num_register_tokens=int(student_cfg.get("n_storage_tokens", 0)),
        pos_embed_shift=student_cfg.get("pos_embed_rope_shift_coords", None),
        pos_embed_jitter=student_cfg.get("pos_embed_rope_jitter_coords", None),
        pos_embed_rescale=student_cfg.get("pos_embed_rope_rescale_coords", None),
    )
    return config


def _infer_block_prefix(target_keys: list[str]) -> str:
    counts = {}
    for k in target_keys:
        parts = k.split(".")
        for i, p in enumerate(parts):
            if p.isdigit():
                suffix = ".".join(parts[i + 1 :])
                if any(tok in suffix for tok in ("attn", "attention", "mlp", "ffn", "layernorm", "norm")):
                    prefix = ".".join(parts[:i])
                    counts[prefix] = counts.get(prefix, 0) + 1
    if not counts:
        raise RuntimeError("Unable to infer transformer block prefix in HF model keys.")
    return max(counts, key=counts.get)


def _find_key(available: set[str], target_state: dict, patterns: list[str], shape=None) -> str | None:
    for pat in patterns:
        for k in available:
            if pat in k and (shape is None or tuple(target_state[k].shape) == tuple(shape)):
                return k
    if shape is not None:
        for k in available:
            if tuple(target_state[k].shape) == tuple(shape):
                return k
    return None


def _map_embeddings(src_sd: dict, tgt_sd: dict, used: set[str]) -> dict:
    mapped = {}
    available = set(tgt_sd.keys()) - used
    has_hf_layout = any(k.startswith("embeddings.") for k in tgt_sd.keys())

    def assign_direct(src_key: str, target_key: str, required: bool = True):
        if src_key not in src_sd:
            if required:
                raise RuntimeError(f"Missing source key: {src_key}")
            return
        if target_key not in tgt_sd:
            if required:
                raise RuntimeError(f"Target key not found: {target_key}")
            return
        tensor = src_sd[src_key]
        tgt_shape = tgt_sd[target_key].shape
        if tensor.shape != tgt_shape:
            if src_key == "mask_token" and tensor.ndim == 2 and len(tgt_shape) == 3:
                tensor = tensor.unsqueeze(1)
            if tensor.shape != tgt_shape:
                raise RuntimeError(f"Shape mismatch for {src_key} -> {target_key}: {tensor.shape} vs {tgt_shape}")
        mapped[target_key] = tensor
        used.add(target_key)

    def assign(src_key: str, patterns: list[str], required: bool = True):
        if src_key not in src_sd:
            if required:
                raise RuntimeError(f"Missing source key: {src_key}")
            return
        tgt_key = _find_key(available, tgt_sd, patterns, shape=src_sd[src_key].shape)
        if tgt_key is None:
            if required:
                raise RuntimeError(f"Failed to map {src_key} -> {patterns}")
            return
        mapped[tgt_key] = src_sd[src_key]
        used.add(tgt_key)

    if has_hf_layout:
        assign_direct("cls_token", "embeddings.cls_token")
        assign_direct("storage_tokens", "embeddings.register_tokens", required=False)
        assign_direct("mask_token", "embeddings.mask_token", required=False)
        assign_direct("patch_embed.proj.weight", "embeddings.patch_embeddings.weight")
        assign_direct("patch_embed.proj.bias", "embeddings.patch_embeddings.bias")
    else:
        assign("cls_token", ["cls_token", "class_token"])
        assign("storage_tokens", ["register_tokens", "storage_tokens"], required=False)
        assign("mask_token", ["mask_token"], required=False)
        assign("patch_embed.proj.weight", ["patch_embeddings.projection.weight", "patch_embeddings.proj.weight"])
        assign("patch_embed.proj.bias", ["patch_embeddings.projection.bias", "patch_embeddings.proj.bias"])
        assign("patch_embed.norm.weight", ["patch_embeddings.norm.weight"], required=False)
        assign("patch_embed.norm.bias", ["patch_embeddings.norm.bias"], required=False)
    return mapped


def _map_norms(src_block: dict, tgt_block_keys: set[str], tgt_sd: dict, used: set[str]) -> dict:
    mapped = {}
    available = tgt_block_keys - used

    def assign(src_key: str, patterns: list[str]):
        if src_key not in src_block:
            return
        tgt_key = _find_key(available, tgt_sd, patterns, shape=src_block[src_key].shape)
        if tgt_key is None:
            raise RuntimeError(f"Failed to map {src_key} -> {patterns}")
        mapped[tgt_key] = src_block[src_key]
        used.add(tgt_key)

    assign("norm1.weight", ["norm1", "layernorm1", "ln1"])
    assign("norm1.bias", ["norm1", "layernorm1", "ln1"])
    assign("norm2.weight", ["norm2", "layernorm2", "ln2"])
    assign("norm2.bias", ["norm2", "layernorm2", "ln2"])
    return mapped


def _map_layerscale(src_block: dict, tgt_block_keys: set[str], tgt_sd: dict, used: set[str]) -> dict:
    mapped = {}
    available = tgt_block_keys - used

    def assign(src_key: str, patterns: list[str]):
        if src_key not in src_block:
            return
        tgt_key = _find_key(available, tgt_sd, patterns, shape=src_block[src_key].shape)
        if tgt_key is None:
            return
        mapped[tgt_key] = src_block[src_key]
        used.add(tgt_key)

    assign("ls1.gamma", ["ls1", "layer_scale_1", "layer_scale1", "layerscale1"])
    assign("ls2.gamma", ["ls2", "layer_scale_2", "layer_scale2", "layerscale2"])
    return mapped


def _map_attention(src_block: dict, tgt_block_keys: set[str], tgt_sd: dict, used: set[str]) -> dict:
    mapped = {}
    available = tgt_block_keys - used

    def assign(tgt_key: str, tensor):
        mapped[tgt_key] = tensor
        used.add(tgt_key)

    if "attn.qkv.weight" in src_block:
        w = src_block["attn.qkv.weight"]
        b = src_block.get("attn.qkv.bias")
        qkv_key = _find_key(available, tgt_sd, ["qkv"], shape=w.shape)
        if qkv_key is not None:
            assign(qkv_key, w)
            if b is not None:
                qkv_bias_key = _find_key(available, tgt_sd, ["qkv"], shape=b.shape)
                if qkv_bias_key is not None:
                    assign(qkv_bias_key, b)
        else:
            wq, wk, wv = w.chunk(3, dim=0)
            bq = bk = bv = None
            if b is not None:
                bq, bk, bv = b.chunk(3, dim=0)
            q_w = _find_key(available, tgt_sd, ["q_proj", "query"], shape=wq.shape)
            k_w = _find_key(available, tgt_sd, ["k_proj", "key"], shape=wk.shape)
            v_w = _find_key(available, tgt_sd, ["v_proj", "value"], shape=wv.shape)
            if not (q_w and k_w and v_w):
                raise RuntimeError("Unable to locate q/k/v projection weights in target model.")
            assign(q_w, wq)
            assign(k_w, wk)
            assign(v_w, wv)
            if b is not None:
                q_b = _find_key(available, tgt_sd, ["q_proj", "query"], shape=bq.shape)
                k_b = _find_key(available, tgt_sd, ["k_proj", "key"], shape=bk.shape)
                v_b = _find_key(available, tgt_sd, ["v_proj", "value"], shape=bv.shape)
                if q_b and k_b and v_b:
                    assign(q_b, bq)
                    assign(k_b, bk)
                    assign(v_b, bv)

    if "attn.proj.weight" in src_block:
        w = src_block["attn.proj.weight"]
        out_w = _find_key(available, tgt_sd, ["out_proj", "proj", "output.dense"], shape=w.shape)
        if out_w is None:
            raise RuntimeError("Unable to locate attention output projection weight in target model.")
        assign(out_w, w)
    if "attn.proj.bias" in src_block:
        b = src_block["attn.proj.bias"]
        out_b = _find_key(available, tgt_sd, ["out_proj", "proj", "output.dense"], shape=b.shape)
        if out_b is not None:
            assign(out_b, b)
    return mapped


def _map_mlp(src_block: dict, tgt_block_keys: set[str], tgt_sd: dict, used: set[str]) -> dict:
    mapped = {}
    available = tgt_block_keys - used

    def assign(src_key: str, patterns: list[str]):
        if src_key not in src_block:
            return
        tgt_key = _find_key(available, tgt_sd, patterns, shape=src_block[src_key].shape)
        if tgt_key is None:
            raise RuntimeError(f"Failed to map {src_key} -> {patterns}")
        mapped[tgt_key] = src_block[src_key]
        used.add(tgt_key)

    if "mlp.w1.weight" in src_block:
        if _find_key(available, tgt_sd, ["gate_proj"], shape=src_block["mlp.w1.weight"].shape):
            assign("mlp.w1.weight", ["gate_proj"])
            assign("mlp.w2.weight", ["up_proj"])
            assign("mlp.w3.weight", ["down_proj"])
            if "mlp.w1.bias" in src_block:
                assign("mlp.w1.bias", ["gate_proj"])
                assign("mlp.w2.bias", ["up_proj"])
                assign("mlp.w3.bias", ["down_proj"])
        elif _find_key(available, tgt_sd, ["w1"], shape=src_block["mlp.w1.weight"].shape):
            assign("mlp.w1.weight", ["w1"])
            assign("mlp.w2.weight", ["w2"])
            assign("mlp.w3.weight", ["w3"])
            if "mlp.w1.bias" in src_block:
                assign("mlp.w1.bias", ["w1"])
                assign("mlp.w2.bias", ["w2"])
                assign("mlp.w3.bias", ["w3"])
        else:
            raise RuntimeError("Target MLP layout not compatible with SwiGLU source weights.")
    else:
        assign("mlp.fc1.weight", ["fc1"])
        assign("mlp.fc2.weight", ["fc2"])
        if "mlp.fc1.bias" in src_block:
            assign("mlp.fc1.bias", ["fc1"])
        if "mlp.fc2.bias" in src_block:
            assign("mlp.fc2.bias", ["fc2"])
    return mapped


def _map_blocks(src_sd: dict, tgt_sd: dict, used: set[str], num_layers: int) -> dict:
    mapped = {}
    tgt_keys = list(tgt_sd.keys())
    has_hf_layout = any(k.startswith("layer.") for k in tgt_keys)

    def assign_direct(src_key: str, tgt_key: str, required: bool = True):
        if src_key not in src_sd:
            if required:
                raise RuntimeError(f"Missing source key: {src_key}")
            return
        if tgt_key not in tgt_sd:
            if required:
                raise RuntimeError(f"Target key not found: {tgt_key}")
            return
        mapped[tgt_key] = src_sd[src_key]
        used.add(tgt_key)

    if has_hf_layout:
        for i in range(num_layers):
            src_prefix = f"blocks.{i}."
            src_block = {k[len(src_prefix) :]: v for k, v in src_sd.items() if k.startswith(src_prefix)}
            if not src_block:
                continue
            prefix = f"layer.{i}."
            # Norms
            assign_direct(f"{src_prefix}norm1.weight", f"{prefix}norm1.weight")
            assign_direct(f"{src_prefix}norm1.bias", f"{prefix}norm1.bias")
            assign_direct(f"{src_prefix}norm2.weight", f"{prefix}norm2.weight")
            assign_direct(f"{src_prefix}norm2.bias", f"{prefix}norm2.bias")

            # Attention
            if f"{src_prefix}attn.qkv.weight" in src_sd:
                w = src_sd[f"{src_prefix}attn.qkv.weight"]
                wq, wk, wv = w.chunk(3, dim=0)
                mapped[f"{prefix}attention.q_proj.weight"] = wq
                mapped[f"{prefix}attention.k_proj.weight"] = wk
                mapped[f"{prefix}attention.v_proj.weight"] = wv
                used.update(
                    {
                        f"{prefix}attention.q_proj.weight",
                        f"{prefix}attention.k_proj.weight",
                        f"{prefix}attention.v_proj.weight",
                    }
                )
                if f"{src_prefix}attn.qkv.bias" in src_sd:
                    b = src_sd[f"{src_prefix}attn.qkv.bias"]
                    bq, bk, bv = b.chunk(3, dim=0)
                    mapped[f"{prefix}attention.q_proj.bias"] = bq
                    mapped[f"{prefix}attention.k_proj.bias"] = bk
                    mapped[f"{prefix}attention.v_proj.bias"] = bv
                    used.update(
                        {
                            f"{prefix}attention.q_proj.bias",
                            f"{prefix}attention.k_proj.bias",
                            f"{prefix}attention.v_proj.bias",
                        }
                    )
            assign_direct(f"{src_prefix}attn.proj.weight", f"{prefix}attention.o_proj.weight")
            assign_direct(f"{src_prefix}attn.proj.bias", f"{prefix}attention.o_proj.bias", required=False)

            # MLP
            if f"{src_prefix}mlp.w1.weight" in src_sd:
                assign_direct(f"{src_prefix}mlp.w1.weight", f"{prefix}mlp.gate_proj.weight")
                assign_direct(f"{src_prefix}mlp.w2.weight", f"{prefix}mlp.up_proj.weight")
                assign_direct(f"{src_prefix}mlp.w3.weight", f"{prefix}mlp.down_proj.weight")
                if f"{src_prefix}mlp.w1.bias" in src_sd:
                    assign_direct(f"{src_prefix}mlp.w1.bias", f"{prefix}mlp.gate_proj.bias")
                    assign_direct(f"{src_prefix}mlp.w2.bias", f"{prefix}mlp.up_proj.bias")
                    assign_direct(f"{src_prefix}mlp.w3.bias", f"{prefix}mlp.down_proj.bias")
            else:
                assign_direct(f"{src_prefix}mlp.fc1.weight", f"{prefix}mlp.up_proj.weight")
                assign_direct(f"{src_prefix}mlp.fc2.weight", f"{prefix}mlp.down_proj.weight")
                if f"{src_prefix}mlp.fc1.bias" in src_sd:
                    assign_direct(f"{src_prefix}mlp.fc1.bias", f"{prefix}mlp.up_proj.bias")
                    assign_direct(f"{src_prefix}mlp.fc2.bias", f"{prefix}mlp.down_proj.bias")

            # LayerScale
            if f"{src_prefix}ls1.gamma" in src_sd:
                assign_direct(f"{src_prefix}ls1.gamma", f"{prefix}layer_scale1.lambda1")
            if f"{src_prefix}ls2.gamma" in src_sd:
                assign_direct(f"{src_prefix}ls2.gamma", f"{prefix}layer_scale2.lambda1")
        return mapped

    block_prefix = _infer_block_prefix(tgt_keys)
    for i in range(num_layers):
        src_prefix = f"blocks.{i}."
        src_block = {k[len(src_prefix) :]: v for k, v in src_sd.items() if k.startswith(src_prefix)}
        if not src_block:
            continue
        tgt_block_keys = {k for k in tgt_keys if k.startswith(f"{block_prefix}.{i}.")}
        mapped.update(_map_norms(src_block, tgt_block_keys, tgt_sd, used))
        mapped.update(_map_attention(src_block, tgt_block_keys, tgt_sd, used))
        mapped.update(_map_mlp(src_block, tgt_block_keys, tgt_sd, used))
        mapped.update(_map_layerscale(src_block, tgt_block_keys, tgt_sd, used))
    return mapped


def _map_final_norms(src_sd: dict, tgt_sd: dict, used: set[str]) -> dict:
    mapped = {}
    available = set(tgt_sd.keys()) - used
    has_hf_norm = "norm.weight" in tgt_sd

    def assign(src_key: str, patterns: list[str]):
        if src_key not in src_sd:
            return
        tgt_key = _find_key(available, tgt_sd, patterns, shape=src_sd[src_key].shape)
        if tgt_key is None:
            return
        mapped[tgt_key] = src_sd[src_key]
        used.add(tgt_key)

    if has_hf_norm and "norm.weight" in src_sd:
        mapped["norm.weight"] = src_sd["norm.weight"]
        used.add("norm.weight")
    else:
        assign("norm.weight", ["layernorm", "final_layer_norm", "norm"])
    if has_hf_norm and "norm.bias" in src_sd:
        mapped["norm.bias"] = src_sd["norm.bias"]
        used.add("norm.bias")
    else:
        assign("norm.bias", ["layernorm", "final_layer_norm", "norm"])
    return mapped


def _write_preprocessor(output_dir: Path, cfg: dict):
    crops_cfg = cfg.get("crops", {})
    mean = crops_cfg.get("rgb_mean", [0.485, 0.456, 0.406])
    std = crops_cfg.get("rgb_std", [0.229, 0.224, 0.225])
    size = int(crops_cfg.get("global_crops_size", 224))
    data = {
        "image_processor_type": "DINOv3ImageProcessor",
        "do_resize": True,
        "size": {"height": size, "width": size},
        "do_normalize": True,
        "image_mean": mean,
        "image_std": std,
    }
    with (output_dir / "preprocessor_config.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert DINOv3 training checkpoints to Transformers format.")
    parser.add_argument("input_dir", type=str, help="Training log directory containing config.yaml and eval/ checkpoints.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for HF model.")
    parser.add_argument("--which", choices=("teacher", "student"), default="teacher", help="Which checkpoint to use.")
    parser.add_argument("--eval-step", type=int, default=None, help="Eval step to convert (default: latest).")
    args = parser.parse_args()

    root = Path(args.input_dir)
    cfg_path = root / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config.yaml not found: {cfg_path}")
    cfg = _load_yaml(cfg_path)

    ckpt_path = _select_eval_checkpoint(root, args.which, args.eval_step)
    state_dict = _load_state_dict(ckpt_path)
    state_dict = _filter_backbone_state_dict(state_dict)

    config = _make_config(cfg, state_dict)
    from transformers import DINOv3ViTModel

    hf_model = DINOv3ViTModel(config)
    target_state = hf_model.state_dict()
    used = set()
    mapped = {}
    mapped.update(_map_embeddings(state_dict, target_state, used))
    mapped.update(_map_blocks(state_dict, target_state, used, config.num_hidden_layers))
    mapped.update(_map_final_norms(state_dict, target_state, used))

    missing, unexpected = hf_model.load_state_dict(mapped, strict=False)
    param_keys = set(dict(hf_model.named_parameters()).keys())
    missing_params = [k for k in missing if k in param_keys]
    if missing_params:
        raise RuntimeError(f"Missing parameters after mapping: {missing_params[:20]}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading HF model: {unexpected[:20]}")

    output_dir = Path(args.output_dir) if args.output_dir else root / "transformers"
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(output_dir)
    _write_preprocessor(output_dir, cfg)
    with (output_dir / "conversion_info.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": str(root),
                "checkpoint": str(ckpt_path),
                "which": args.which,
                "eval_step": args.eval_step,
            },
            f,
            indent=2,
        )
    print(f"Saved transformers model to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Example: load and run inference with transformers
#
# Requires:
#   pip install transformers pillow
#
# Usage:
#   python scripts/convert_dinov3_to_transformers.py /path/to/run --output-dir /tmp/dino
#   python scripts/convert_dinov3_to_transformers.py \
#   /mnt/data/train_ssl/imagenet_1k/dino_neg/logs_train/log_20260218_2014 \
#   --output-dir /tmp/dino_neg_transformers
"""
    import torch
    from PIL import Image
    from transformers import DINOv3ImageProcessor, DINOv3ViTModel

    model_dir = "/tmp/dino_neg_transformers"
    image_path = "/path/to/image.jpg"

    processor = DINOv3ImageProcessor.from_pretrained(model_dir)
    model = DINOv3ViTModel.from_pretrained(model_dir)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    print(outputs.last_hidden_state.shape)

"""
