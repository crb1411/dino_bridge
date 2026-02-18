# dino_bridge

`dino_bridge` 是基于 DINOv3 的 CRB 分支，目标是在受限算力下稳定训练高质量视觉压缩表征（`D=1024`），并增强对细粒度结构的可迁移能力。

原始官方文档已完整保留为：`README_dinov3.md`。

## 1. 方法摘要（基于论文内容）

本分支把 SSL 视为“视觉压缩器”训练问题：把高维图像压缩到紧凑向量，同时保持下游任务可用信息。

核心设计可以概括为 3+1：

- `DTCH-SK`（Dual-Temperature Cumulative History Sinkhorn）
  - 在更平滑的 soft 空间做历史均衡（balanced）。
  - 在后处理用幂次恢复单样本尖锐性（crisp）。
  - 解决大原型字典（如 `K=65536`）+ 小 batch 下“均衡与尖锐冲突”的训练动态问题。
- `Invariant Learning`（正交不变性）
  - `Crop-Resize`: 强化尺度变化下的全局语义一致性（主对齐 CLS）。
  - `PatchRoll`: 强化温和重排下的全局 + 局部鲁棒性（CLS + 稀疏 patch 对齐）。
- `Patch-to-CLS Bridging`
  - 用 Inverse Patch Embedding 聚合 patch token，构造 patch-global 表征。
  - 在原型空间对齐 teacher CLS，把 patch 细节显式蒸馏进最终 CLS 压缩向量。
- `DINO/iBOT` 原有结构保留并协同
  - DINO local/global + iBOT patch + KoLeo 与上述模块可组合训练。

## 2. 当前分支关键工程点

- 训练入口：`dinov3/new_train/train/train_img.py`
- 自动设备选择：`dinov3/new_train/utils/auto_device.py`
  - 统一支持 `cuda / npu / mps / xpu / cpu`。
- FSDP+Compile：`dinov3/fsdp/ac_compile_parallelize.py`
  - 当配置 `train.compile=true` 且设备不是 CUDA（如 NPU）时，会自动跳过 `torch.compile`，不阻塞训练。
- DTCH 实现：`dinov3/new_train/dtch/dtch_blance.py`
- Bridge 聚合器：`dinov3/new_train/models/inverse_patch.py`
- Resize/Shuffle 元架构：`dinov3/new_train/train/ssl_resize_shuffle.py`

## 3. 训练启动（示例）

当前使用脚本为 `run/run_ssl_debug.sh`，核心命令等价于：

```bash
torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=40028 \
  -m dinov3.new_train.train.train_img \
  --config-file /mnt/seek/ssl/dinov3/run/dinov3_vitlarge_pretrain.yaml \
  --checkpoint_dir <ckpt_path> \
  --output-dir <output_dir>
```

NPU 场景建议保留：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=40028
```

## 4. `run/dinov3_vitlarge_pretrain.yaml` 逐行说明

说明：下面按实际文件行号（`nl -ba run/dinov3_vitlarge_pretrain.yaml`）逐行解释。

- L1 `MODEL:` 模型级配置根节点。
- L2 `META_ARCHITECTURE: SSLResizeShuffle` 选择训练元架构为 Resize+Shuffle 分支。
- L3 `DEVICE: cuda` 配置中的目标设备标识；实际运行仍由 `auto_device.py` 自动检测。
- L4 `WEIGHTS: ''` 初始权重路径，空表示不额外加载。
- L5 `DTYPE: float32` 模型默认计算 dtype。
- L6 `compute_precision:` 混合精度/FSDP 精度策略根节点。
- L7 `param_dtype: bf16` 参数前向/反向主精度设为 bf16。
- L8 `reduce_dtype: fp32` 梯度规约精度设为 fp32（更稳）。
- L9 `sharding_strategy: SHARD_GRAD_OP` FSDP 分片策略。
- L10 `dino:` DINO 分支配置根节点。
- L11 `loss_weight: 1.0` DINO 损失权重。
- L12 `global_ignore_diagonal: true` 计算全局对齐时忽略对角项（同位配对）。
- L13 `head_n_prototypes: 65536` DINO head 原型数 `K`。
- L14 `head_bottleneck_dim: 256` DINO head bottleneck 维度。
- L15 `head_norm_last_layer: false` DINO 最后一层不做 norm/weight-norm 约束。
- L16 `head_nlayers: 3` DINO head MLP 层数。
- L17 `head_hidden_dim: 2048` DINO head 隐层维度。
- L18 `koleo_cls_loss_weight: 0.1` CLS KoLeo 正则权重。
- L19 `koleo_patch_loss_weight: 0.1` Patch KoLeo 正则权重。
- L20 `koleo_cls_gate_enabled: true` 启用 CLS KoLeo 门控。
- L21 `koleo_patch_gate_enabled: true` 启用 Patch KoLeo 门控。
- L22 `koleo_cls_gate_threshold: 0.0` CLS KoLeo 门控阈值。
- L23 `koleo_patch_gate_threshold: 0.0` Patch KoLeo 门控阈值。
- L24 `koleo_loss_distributed: false` KoLeo 不做跨卡分布式聚合。
- L25 `koleo_topk: 1` KoLeo 邻居选择 top-k=1。
- L26 `koleo_distributed_replicas: 0` 分布式 KoLeo 副本数（0 表示关闭）。
- L27 `koleo_distributed_loss_group_size: null` 分布式 KoLeo 分组大小，空表示默认。
- L28 `force_weight_norm: false` 不强制对 head 使用 weight norm。
- L29 `ibot:` iBOT 分支配置根节点。
- L30 `loss_weight: 1.0` iBOT 损失权重。
- L31 `mask_sample_probability: 0.5` 每张图像进入 mask 机制的概率。
- L32 `mask_ratio_min_max:` mask 比例范围定义。
- L33 `- 0.1` mask 最小比例 10%。
- L34 `- 0.5` mask 最大比例 50%。
- L35 `mask_random_circular_shift: false` 不启用 mask 随机环移。
- L36 `force_masking_even_with_zero_weight: false` 即使 loss=0 也强制 mask 的开关（这里关闭）。
- L37 `separate_head: true` iBOT 使用独立 head。
- L38 `head_n_prototypes: 65536` iBOT head 原型数。
- L39 `head_bottleneck_dim: 256` iBOT head bottleneck 维度。
- L40 `head_norm_last_layer: false` iBOT head 最后一层不加 norm。
- L41 `head_nlayers: 3` iBOT head 层数。
- L42 `head_hidden_dim: 2048` iBOT head 隐层维度。
- L43 `gram:` Gram 分支配置根节点。
- L44 `use_loss: false` 不启用 Gram loss。
- L45 `compute_stats: false` 不计算 Gram 统计。
- L46 `train:` 训练流程主配置根节点。
- L47 `batch_size_per_gpu: 64` 单卡 batch size。
- L48 `dataset_path: null` 数据集路径留空（通常由 `data.*` 子项给出）。
- L49 `saveckp_freq: 20` checkpoint 保存频率（按训练内部计数单位）。
- L50 `seed: 0` 随机种子。
- L51 `num_workers: 4` DataLoader worker 数。
- L52 `OFFICIAL_EPOCH_LENGTH: 1000` 每个 epoch 对应的迭代步数定义。
- L53 `monitor_gradient_norm: false` 关闭梯度范数监控。
- L54 `chunk_schedule: []` 不启用 chunk 调度。
- L55 `cache_dataset: true` 启用数据集缓存。
- L56 `use_teacher_head: true` student 使用 teacher head 目标进行蒸馏。
- L57 `learn_from_teacher_tokens: false` 不直接从 teacher token 学习。
- L58 `reshuffle_sampler_perm: true` 每轮重排 sampler 顺序。
- L59 `centering: sinkhorn_knopp` 使用 Sinkhorn-Knopp 类中心化/分配。
- L60 `checkpointing: true` 启用激活检查点。
- L61 `checkpointing_full: false` 不做 full checkpointing，使用选择性策略。
- L62 `compile: true` 请求 `torch.compile`；在非 CUDA 设备（如 NPU）会自动跳过。
- L63 `cudagraphs: false` 不启用 CUDA Graphs。
- L64 `cell_augmentation: false` 关闭细胞增强分支。
- L65 `cell_augmentation_type: hpa` 细胞增强类型占位配置（当前未启用）。
- L66 `sharded_eval_checkpoint: false` 评估时不使用分片 checkpoint。
- L67 `student:` student backbone 配置根节点。
- L68 `arch: vit_large` student 架构为 ViT-L。
- L69 `patch_size: 16` patch 大小 16。
- L70 `drop_path_rate: 0.2` stochastic depth 比例。
- L71 `layerscale: 1.0e-05` LayerScale 初始化系数。
- L72 `patch_drop: 0.0` 不丢弃 patch token。
- L73 `pretrained_weights: ''` student 预训练权重路径为空。
- L74 `ffn_layer: swiglu64` FFN 类型为 swiglu64。
- L75 `ffn_ratio: 4` FFN 扩展倍率。
- L76 `resume_from_teacher_chkpt: ''` 不从 teacher checkpoint 恢复。
- L77 `qkv_bias: true` QKV 线性层启用 bias。
- L78 `proj_bias: true` attention 输出投影启用 bias。
- L79 `ffn_bias: true` FFN 线性层启用 bias。
- L80 `norm_layer: layernormbf16` 归一化层实现为 bf16 友好版本。
- L81 `n_storage_tokens: 4` 额外 storage token 数量。
- L82 `untie_cls_and_patch_norms: false` CLS 与 patch norm 不拆分。
- L83 `untie_global_and_local_cls_norm: true` global/local CLS norm 拆分。
- L84 `mask_k_bias: true` mask 相关 K 分支使用 bias。
- L85 `in_chans: 3` 输入通道数为 RGB 3。
- L86 `pos_embed_type: rope` 位置编码采用 RoPE。
- L87 `pos_embed_rope_base: 100` RoPE base 参数。
- L88 `pos_embed_rope_min_period: null` RoPE 最小周期留空（默认策略）。
- L89 `pos_embed_rope_max_period: null` RoPE 最大周期留空（默认策略）。
- L90 `pos_embed_rope_normalize_coords: separate` 坐标归一化策略为 separate。
- L91 `pos_embed_rope_shift_coords: null` 不启用坐标平移扰动。
- L92 `pos_embed_rope_jitter_coords: null` 不启用坐标 jitter。
- L93 `pos_embed_rope_rescale_coords: 2` RoPE 坐标缩放系数为 2。
- L94 `pos_embed_rope_dtype: bf16` RoPE 计算 dtype 为 bf16。
- L95 `fp8_enabled: false` 不启用 FP8。
- L96 `fp8_filter: blocks` FP8 过滤目标（当前无效，因为 FP8 关闭）。
- L97 `teacher:` teacher 配置根节点。
- L98 `momentum_teacher: null` teacher 动量由 `schedules.momentum` 接管。
- L99 `final_momentum_teacher: null` 终止动量由 schedule 控制。
- L100 `warmup_teacher_temp: null` warmup 温度由 schedule 控制。
- L101 `teacher_temp: null` teacher 温度由 schedule 控制。
- L102 `warmup_teacher_temp_epochs: null` warmup 轮次由 schedule 控制。
- L103 `in_chans: 3` teacher 输入通道数。
- L104 `distillation:` 额外蒸馏配置根节点。
- L105 `enabled: false` 关闭单模型蒸馏。
- L106 `full_cfg_path: ''` 蒸馏配置路径为空。
- L107 `checkpoint_path: ''` 蒸馏权重路径为空。
- L108 `multidistillation:` 多蒸馏配置根节点。
- L109 `enabled: false` 关闭多蒸馏。
- L110 `hrft:` HRFT 配置根节点。
- L111 `enabled: false` 关闭 HRFT。
- L112 `checkpoint_path: ''` HRFT 权重路径为空。
- L113 `optim:` 优化器配置根节点。
- L114 `epochs: 1000` 总训练 epoch。
- L115 `optimizer: adamw` 优化器为 AdamW。
- L116 `weight_decay: null` 旧版 WD 配置留空（使用 `schedules.weight_decay`）。
- L117 `weight_decay_end: null` 旧版 WD 终值留空。
- L118 `lr: null` 旧版 LR 配置留空（使用 `schedules.lr`）。
- L119 `warmup_epochs: null` 旧版 warmup 配置留空。
- L120 `min_lr: null` 旧版最小 LR 留空。
- L121 `schedule_trunc_extra: null` 不使用旧版截断调度。
- L122 `clip_grad: 30.0` 梯度裁剪阈值。
- L123 `freeze_last_layer_epochs: null` 旧版 last-layer freeze 留空（使用 `schedules.lr.freeze_last_layer_epochs`）。
- L124 `scaling_rule: sqrt_wrt_1024` 按全局 batch 相对 1024 的平方根缩放 LR。
- L125 `patch_embed_lr_mult: 0.2` patch-embed 层 LR 乘子。
- L126 `dino_head_wd_multiplier: 1.0` DINO head 的 WD 乘子。
- L127 `dino_head_lr_multiplier: 0.8` DINO head 的 LR 乘子。
- L128 `layerwise_decay: 0.98` 按层衰减 LR。
- L129 `multi_tensor_optim: true` 启用 multi-tensor 优化路径。
- L130 `dump_fsdp_weights_path: ''` FSDP 权重导出路径为空。
- L131 `adamw_beta1: 0.9` AdamW beta1。
- L132 `adamw_beta2: 0.99` AdamW beta2。
- L133 `crops:` 数据增强裁剪配置根节点。
- L134 `global_crops_scale:` global crop 尺度范围定义。
- L135 `- 0.32` global crop 最小面积比例。
- L136 `- 1.0` global crop 最大面积比例。
- L137 `local_crops_number: 8` local crop 数量。
- L138 `local_crops_scale:` local crop 尺度范围定义。
- L139 `- 0.05` local crop 最小面积比例。
- L140 `- 0.32` local crop 最大面积比例。
- L141 `global_crops_size: 224` global crop 输出尺寸。
- L142 `local_crops_size: 96` local crop 输出尺寸。
- L143 `localcrops_subset_of_globalcrops: false` local crop 不限制为 global 子集。
- L144 `share_color_jitter: false` 不共享 color jitter 参数。
- L145 `horizontal_flips: false` 关闭水平翻转。
- L146 `rgb_mean:` 归一化均值列表。
- L147 `- 0.485` R 通道均值。
- L148 `- 0.456` G 通道均值。
- L149 `- 0.406` B 通道均值。
- L150 `rgb_std:` 归一化标准差列表。
- L151 `- 0.229` R 通道标准差。
- L152 `- 0.224` G 通道标准差。
- L153 `- 0.225` B 通道标准差。
- L154 `use_resize_shuffle_augmentor: true` 启用 resize-shuffle 增强。
- L155 `resize_shuffle_augmentor_switch: null` 无额外增强切换策略（默认）。
- L156 `checkpointing:` checkpoint 保留策略根节点。
- L157 `period: 1000` 保存周期。
- L158 `max_to_keep: 10` 最多保留最近 10 个。
- L159 `keep_every: 50000` 每隔 50000 step 做长期保留。
- L160 `save_student: true` checkpoint 中保存 student。
- L161 `schedules:` 新版调度器配置根节点。
- L162 `lr:` 学习率调度配置根节点。
- L163 `start: 0` LR 初值。
- L164 `peak: 5.0e-05` LR 峰值。
- L165 `end: 5.0e-06` LR 终值。
- L166 `warmup_epochs: 1` LR warmup 轮数。
- L167 `freeze_last_layer_epochs: 1` 末层冻结轮数。
- L168 `weight_decay:` WD 调度配置根节点。
- L169 `start: 0.04` WD 初值。
- L170 `peak: 0.04` WD 峰值。
- L171 `end: 0.04` WD 终值。
- L172 `warmup_epochs: 0` WD warmup 轮数。
- L173 `teacher_temp:` teacher 温度调度根节点。
- L174 `start: 0.04` teacher 温度初值。
- L175 `peak: 0.07` teacher 温度峰值。
- L176 `end: 0.07` teacher 温度终值。
- L177 `warmup_epochs: 100` teacher 温度 warmup 轮数。
- L178 `momentum:` EMA 动量调度根节点。
- L179 `start: 0.994` 动量初值。
- L180 `peak: 0.994` 动量峰值。
- L181 `end: 0.994` 动量终值。
- L182 `warmup_epochs: 0` 动量 warmup 轮数。
- L183 `data:` 数据源配置根节点。
- L184 `imagenet_1k:` Imagenet-1K 数据集子配置。
- L185 `txt_path: /root/data/imagenet_1k/data_new/imagenet_1k.txt` 样本列表文件路径。
- L186 `dataset_ratio: 1.0` 数据采样比例（1.0 表示全量）。
- L187 `resize_shuffle_augmentor:` ResizeShuffle 额外损失配置根节点。
- L188 `resize_paste_loss_weight:` resize-paste loss 权重调度根节点。
- L189 `start: 0.0` resize-paste loss 初值。
- L190 `peak: 0.5` resize-paste loss 峰值。
- L191 `end: 0.5` resize-paste loss 终值。
- L192 `zero_epochs: 50` 前 50 轮置零。
- L193 `warmup_epochs: 50` 后续 50 轮 warmup。
- L194 `patch_shuffle_loss_weight:` patch-shuffle loss 权重调度根节点。
- L195 `start: 0.0` patch-shuffle loss 初值。
- L196 `peak: 1.0` patch-shuffle loss 峰值。
- L197 `end: 1.0` patch-shuffle loss 终值。
- L198 `zero_epochs: 50` patch-shuffle 前 50 轮置零。
- L199 `warmup_epochs: 50` patch-shuffle warmup 轮数。
- L200 `patch_shuffle_patch_weight: 1.0` patch 分支权重。
- L201 `patch_shuffle_cls_weight: 1.0` CLS 分支权重。
- L202 `patch_shuffle_patch_probability: 0.5` 进入 patch-shuffle 的概率。
- L203 `patch_shuffle_patch_min_max:` patch-shuffle 比例范围定义。
- L204 `- 0.05` patch-shuffle 最小比例。
- L205 `- 0.15` patch-shuffle 最大比例。
- L206 `use_all_shift_mask: true` 使用全量 shift mask 策略。
- L207 `bridge_patchshuffle_weight:` bridge 的 patchshuffle 权重调度根节点。
- L208 `start: 0.0` bridge-patchshuffle 初值。
- L209 `peak: 0.25` bridge-patchshuffle 峰值。
- L210 `end: 0.25` bridge-patchshuffle 终值。
- L211 `zero_epochs: 50` bridge-patchshuffle 前 50 轮置零。
- L212 `warmup_epochs: 50` bridge-patchshuffle warmup 轮数。
- L213 `bridge_global_weight:` bridge 的 global 权重调度根节点。
- L214 `start: 0.0` bridge-global 初值。
- L215 `peak: 0.25` bridge-global 峰值。
- L216 `end: 0.25` bridge-global 终值。
- L217 `zero_epochs: 50` bridge-global 前 50 轮置零。
- L218 `warmup_epochs: 50` bridge-global warmup 轮数。
- L219 `patch_size: 16` resize-shuffle/bridge 分支使用的 patch 大小。
- L220 `dtch:` DTCH 配置根节点。
- L221 `enabled: true` 启用 DTCH 机制。
- L222 `patch_hist_cache: 40000` patch 历史缓存统计容量。
- L223 `cls_hist_cache: 40000` CLS 历史缓存统计容量。
