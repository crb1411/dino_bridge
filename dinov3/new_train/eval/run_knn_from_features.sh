#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-/mnt/work/ckpt/ckpt578999}"
FEATURE_DIR="${FEATURE_DIR:-${OUTPUT_DIR}/feature_cache}"
TRAIN_TRY="${TRAIN_TRY:-0}"

BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
KS="${KS:-10,20,100,200}"
TEMPERATURE="${TEMPERATURE:-0.07}"
METRIC="${METRIC:-mean_accuracy}"
DEVICE="${DEVICE:-auto}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPU_VISIBLE_DEVICES="${NPU_VISIBLE_DEVICES:-0}"
TRAIN_CHUNK_SIZE="${TRAIN_CHUNK_SIZE:-0}"
SKIP_FIRST_NN="${SKIP_FIRST_NN:-false}"
NORMALIZE="${NORMALIZE:-true}"

detect_device() {
  python - <<'PY'
import torch
if torch.cuda.is_available():
    print("cuda")
elif hasattr(torch, "npu") and torch.npu.is_available():
    print("npu")
else:
    print("cpu")
PY
}

if [[ "${DEVICE}" == "auto" ]]; then
  DEVICE="$(detect_device)"
fi

case "${DEVICE}" in
  cuda)
    export CUDA_VISIBLE_DEVICES
    export NPU_VISIBLE_DEVICES=""
    ;;
  npu)
    export NPU_VISIBLE_DEVICES
    export CUDA_VISIBLE_DEVICES=""
    ;;
  cpu)
    export NPU_VISIBLE_DEVICES=""
    export CUDA_VISIBLE_DEVICES=""
    ;;
  *)
    echo "Invalid DEVICE value: ${DEVICE} (expected auto/cuda/npu/cpu)" >&2
    exit 2
    ;;
esac

ARGS=(
  --feature-dir "${FEATURE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --train-try "${TRAIN_TRY}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --ks "${KS}"
  --temperature "${TEMPERATURE}"
  --metric "${METRIC}"
  --device "${DEVICE}"
  --train-chunk-size "${TRAIN_CHUNK_SIZE}"
)

if [[ "${SKIP_FIRST_NN}" == "true" ]]; then
  ARGS+=(--skip-first-nn)
fi
if [[ "${NORMALIZE}" == "false" ]]; then
  ARGS+=(--no-normalize)
fi

PYTHONPATH="${REPO_ROOT}" \
python "${REPO_ROOT}/dinov3/new_train/eval/knn_from_features.py" \
  "${ARGS[@]}"
