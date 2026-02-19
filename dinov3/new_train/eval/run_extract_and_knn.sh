#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   WEIGHTS=/path/to/checkpoint OUTPUT_DIR=/path/to/output \
#     bash dinov3/new_train/eval/run_extract_and_knn.sh
#
# Required env:
#   WEIGHTS: checkpoint path (file or sharded checkpoint directory)
#   OUTPUT_DIR: directory for feature cache and evaluation outputs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}}"

WEIGHTS="${WEIGHTS:-/mnt/data/train_ssl/imagenet_1k/train_imagenet_1k_22/logs_out/log_20260207_0905/eval/160999/teacher_checkpoint.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/work/ckpt/ckpt578999}"

if [[ -z "${WEIGHTS}" ]]; then
  echo "Missing required env: WEIGHTS" >&2
  exit 2
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "Missing required env: OUTPUT_DIR" >&2
  exit 2
fi

CONFIG="${CONFIG:-/mnt/data/train_ssl/run/dinov3_vitlarge_pretrain.yaml}"
DATASET_ROOT="${DATASET_ROOT:-/root/data/imagenet_1k}"

# Feature extraction (logreg with run_eval=false)
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-40029}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPU_VISIBLE_DEVICES="${NPU_VISIBLE_DEVICES:-0}"

# kNN options (keep aligned with run_knn_from_features.sh defaults)
FEATURE_DIR="${FEATURE_DIR:-${OUTPUT_DIR}/feature_cache}"
TRAIN_TRY="${TRAIN_TRY:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
KS="${KS:-10,20,100,200}"
TEMPERATURE="${TEMPERATURE:-0.07}"
METRIC="${METRIC:-mean_accuracy}"
DEVICE="${DEVICE:-auto}" # auto/cuda/npu/cpu
TRAIN_CHUNK_SIZE="${TRAIN_CHUNK_SIZE:-0}"
SKIP_FIRST_NN="${SKIP_FIRST_NN:-false}"
NORMALIZE="${NORMALIZE:-true}"

LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs_eval}"

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

abs_path() {
  local p="${1:-}"
  if [ -z "${p}" ]; then
    echo ""
    return
  fi
  p="${p#"${p%%[![:space:]]*}"}"
  p="${p%"${p##*[![:space:]]}"}"
  if command -v realpath >/dev/null 2>&1; then
    realpath -m "${p}"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    python - <<PY
import os
print(os.path.abspath("${p}"))
PY
    return
  fi
  echo "${p}"
}

CONFIG="$(abs_path "${CONFIG}")"
WEIGHTS="$(abs_path "${WEIGHTS}")"
OUTPUT_DIR="$(abs_path "${OUTPUT_DIR}")"
DATASET_ROOT="$(abs_path "${DATASET_ROOT}")"
WORK_DIR="$(abs_path "${WORK_DIR}")"
FEATURE_DIR="$(abs_path "${FEATURE_DIR}")"
LOG_DIR="$(abs_path "${LOG_DIR}")"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
FEATURE_LOG="${LOG_DIR}/extract_features_${TIMESTAMP}.log"
KNN_LOG="${LOG_DIR}/knn_${TIMESTAMP}.log"

if [[ "${DEVICE}" == "auto" ]]; then
  DEVICE="$(detect_device)"
fi

case "${DEVICE}" in
  cuda|npu|cpu) ;;
  *)
    echo "Invalid DEVICE value: ${DEVICE} (expected auto/cuda/npu/cpu)" >&2
    exit 2
    ;;
esac

cd "${WORK_DIR}"
if [[ "${DEVICE}" == "cuda" ]]; then
  export CUDA_VISIBLE_DEVICES
  export NPU_VISIBLE_DEVICES=""
elif [[ "${DEVICE}" == "npu" ]]; then
  export NPU_VISIBLE_DEVICES
  export CUDA_VISIBLE_DEVICES=""
else
  export NPU_VISIBLE_DEVICES=""
  export CUDA_VISIBLE_DEVICES=""
fi
export MASTER_ADDR
export MASTER_PORT
export HOSTNAME="$(hostname)"

echo "[1/2] Start feature extraction (run_eval=false)"
echo "  WEIGHTS=${WEIGHTS}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  FEATURE_DIR=${FEATURE_DIR}"
echo "  DEVICE=${DEVICE}"

PYTHONPATH="${REPO_ROOT}" \
torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/dinov3/eval/log_regression.py" \
  run_eval=false \
  model.config_file="${CONFIG}" \
  model.pretrained_weights="${WEIGHTS}" \
  output_dir="${OUTPUT_DIR}" \
  train.dataset="ImageNet:split=TRAIN:root=${DATASET_ROOT}:extra=${DATASET_ROOT}" \
  train.val_dataset="ImageNet:split=VAL:root=${DATASET_ROOT}:extra=${DATASET_ROOT}" \
  eval.test_dataset="ImageNet:split=VAL:root=${DATASET_ROOT}:extra=${DATASET_ROOT}" \
  2>&1 | tee "${FEATURE_LOG}"

if [[ ! -f "${FEATURE_DIR}/train_features_try${TRAIN_TRY}.pt" ]]; then
  echo "Feature extraction failed: missing ${FEATURE_DIR}/train_features_try${TRAIN_TRY}.pt" >&2
  exit 3
fi
if [[ ! -f "${FEATURE_DIR}/train_labels_try${TRAIN_TRY}.pt" ]]; then
  echo "Feature extraction failed: missing ${FEATURE_DIR}/train_labels_try${TRAIN_TRY}.pt" >&2
  exit 3
fi
if [[ ! -f "${FEATURE_DIR}/test_features.pt" && ! -f "${FEATURE_DIR}/val_features.pt" ]]; then
  echo "Feature extraction failed: missing test/val features under ${FEATURE_DIR}" >&2
  exit 3
fi
if [[ ! -f "${FEATURE_DIR}/test_labels.pt" && ! -f "${FEATURE_DIR}/val_labels.pt" ]]; then
  echo "Feature extraction failed: missing test/val labels under ${FEATURE_DIR}" >&2
  exit 3
fi

echo "[2/2] Start kNN evaluation"

KNN_ARGS=(
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
  KNN_ARGS+=(--skip-first-nn)
fi
if [[ "${NORMALIZE}" == "false" ]]; then
  KNN_ARGS+=(--no-normalize)
fi

PYTHONPATH="${REPO_ROOT}" \
python "${REPO_ROOT}/dinov3/new_train/eval/knn_from_features.py" \
  "${KNN_ARGS[@]}" \
  2>&1 | tee "${KNN_LOG}"

echo "Done."
echo "Feature log: ${FEATURE_LOG}"
echo "kNN log: ${KNN_LOG}"
echo "kNN results: ${OUTPUT_DIR}/results-knn.csv"
