#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_log_regression.sh [RUN_EVAL]
#   CONFIG=... WEIGHTS=... OUTPUT_DIR=... DATASET_ROOT=... bash run_log_regression.sh [RUN_EVAL]
#   RUN_EVAL: true/false (default: true). When false, only extracts and caches features.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}}"

CONFIG="${CONFIG:-/mnt/data/train_ssl/run/dinov3_vitlarge_pretrain.yaml}"
# WEIGHTS="${WEIGHTS:-/mnt/data/train/crb/train_out/train_imagenet_1k_120/logs_out/log_20260126_0701/eval/training_578999/teacher_checkpoint.pth}"
WEIGHTS="${WEIGHTS:-/mnt/data/train_ssl/imagenet_1k/train_imagenet_1k_22/logs_out/log_20260202_2039/eval/training_92999/sharded_teacher_checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/work/ckpt/ckpt_92999}"
DATASET_ROOT="${DATASET_ROOT:-/root/data/imagenet_1k}"
RUN_EVAL="${1:-${RUN_EVAL:-true}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-40029}"
DEVICE="${DEVICE:-auto}" # auto/cuda/npu/cpu
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPU_VISIBLE_DEVICES="${NPU_VISIBLE_DEVICES:-0}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs_eval}"
# CPU parallelism controls
CPU_CORES="${CPU_CORES:-150}"
DEFAULT_NUM_WORKERS=$(( CPU_CORES / 3 ))
if (( DEFAULT_NUM_WORKERS < 8 )); then
  DEFAULT_NUM_WORKERS=8
fi
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-${DEFAULT_NUM_WORKERS}}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-${DEFAULT_NUM_WORKERS}}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CPU_CORES}}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-${CPU_CORES}}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${CPU_CORES}}"
NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${CPU_CORES}}"

abs_path() {
  local p="${1:-}"
  if [ -z "${p}" ]; then
    echo ""
    return
  fi
  # Trim leading/trailing whitespace to avoid accidental " /abs/path" issues.
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
LOG_DIR="$(abs_path "${LOG_DIR}")"

case "${RUN_EVAL}" in
  1|true|TRUE|True|yes|YES) RUN_EVAL="true" ;;
  0|false|FALSE|False|no|NO) RUN_EVAL="false" ;;
  *)
    echo "Invalid RUN_EVAL value: ${RUN_EVAL} (expected true/false/1/0/yes/no)" >&2
    exit 2
    ;;
esac

mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/logreg_${TIMESTAMP}_node${NNODES}_${NODE_RANK}.log"

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
export OMP_NUM_THREADS
export MKL_NUM_THREADS
export OPENBLAS_NUM_THREADS
export NUMEXPR_NUM_THREADS

echo "Using DEVICE=${DEVICE}"
nohup env PYTHONPATH="${REPO_ROOT}" \
  torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${REPO_ROOT}/dinov3/eval/log_regression.py" \
    run_eval="${RUN_EVAL}" \
    model.config_file="${CONFIG}" \
    model.pretrained_weights="${WEIGHTS}" \
    output_dir="${OUTPUT_DIR}" \
    train.num_workers="${TRAIN_NUM_WORKERS}" \
    eval.num_workers="${EVAL_NUM_WORKERS}" \
    train.dataset="ImageNet:split=TRAIN:root=${DATASET_ROOT}:extra=${DATASET_ROOT}" \
    train.val_dataset="ImageNet:split=VAL:root=${DATASET_ROOT}:extra=${DATASET_ROOT}" \
    eval.test_dataset="ImageNet:split=VAL:root=${DATASET_ROOT}:extra=${DATASET_ROOT}" \
  > "${LOG_FILE}" 2>&1 &

echo "评测任务已在后台启动，PID: $!"
echo "日志文件: ${LOG_FILE}"
echo "查看日志: tail -f ${LOG_FILE}"
