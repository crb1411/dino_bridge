#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Prepare ImageNet-1k dataset for dinov3 log_regression evaluation.

Usage:
  bash prepare_imagenet1k_for_logreg.sh [options]

Options:
  --root PATH              ImageNet root. Default: /root/data/imagenet_1k
  --devkit-root PATH       Devkit root. Default: <root>/ILSVRC2012_devkit_t12
  --train-src PATH         Train source dir. Default: <root>/ILSVRC2012_img_train
  --val-src PATH           Val source dir. Default: <root>/ILSVRC2012_img_val
  --extra PATH             Extra output dir for npy metadata. Default: <root>
  --python BIN             Python executable. Default: python
  --skip-train-extract     Do not extract *.tar inside train source
  --force-labels           Force overwrite labels.txt
  -h, --help               Show this help

This script will:
  1) Ensure <root>/train and <root>/val exist (symlink to source dirs if needed)
  2) Optionally extract <train-src>/*.tar into per-class folders if class dirs are missing
  3) Run prepare_imagenet_extra.py to generate labels.txt + entries/class metadata
  4) Check that required metadata files exist
EOF
}

ROOT="/root/data/imagenet_1k"
PYTHON_BIN="python"
SKIP_TRAIN_EXTRACT="false"
FORCE_LABELS="false"
DEVKIT_ROOT=""
TRAIN_SRC=""
VAL_SRC=""
EXTRA=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"
      shift 2
      ;;
    --devkit-root)
      DEVKIT_ROOT="$2"
      shift 2
      ;;
    --train-src)
      TRAIN_SRC="$2"
      shift 2
      ;;
    --val-src)
      VAL_SRC="$2"
      shift 2
      ;;
    --extra)
      EXTRA="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --skip-train-extract)
      SKIP_TRAIN_EXTRACT="true"
      shift
      ;;
    --force-labels)
      FORCE_LABELS="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

ROOT="$(readlink -f "${ROOT}")"
TRAIN_SRC="${TRAIN_SRC:-${ROOT}/ILSVRC2012_img_train}"
VAL_SRC="${VAL_SRC:-${ROOT}/ILSVRC2012_img_val}"
DEVKIT_ROOT="${DEVKIT_ROOT:-${ROOT}/ILSVRC2012_devkit_t12}"
EXTRA="${EXTRA:-${ROOT}}"

TRAIN_SRC="$(readlink -f "${TRAIN_SRC}")"
VAL_SRC="$(readlink -f "${VAL_SRC}")"
DEVKIT_ROOT="$(readlink -f "${DEVKIT_ROOT}")"
EXTRA="$(readlink -f "${EXTRA}")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPARE_SCRIPT="${SCRIPT_DIR}/prepare_imagenet_extra.py"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

require_dir() {
  local path="$1"
  local name="$2"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] ${name} not found or not a directory: ${path}" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  local name="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] ${name} not found: ${path}" >&2
    exit 1
  fi
}

require_dir "${ROOT}" "root"
require_dir "${TRAIN_SRC}" "train source"
require_dir "${VAL_SRC}" "val source"
require_dir "${DEVKIT_ROOT}" "devkit root"
require_file "${PREPARE_SCRIPT}" "prepare script"

mkdir -p "${ROOT}"
mkdir -p "${EXTRA}"

if [[ ! -e "${ROOT}/train" ]]; then
  ln -s "${TRAIN_SRC}" "${ROOT}/train"
  echo "[INFO] created symlink: ${ROOT}/train -> ${TRAIN_SRC}"
fi
if [[ ! -e "${ROOT}/val" ]]; then
  ln -s "${VAL_SRC}" "${ROOT}/val"
  echo "[INFO] created symlink: ${ROOT}/val -> ${VAL_SRC}"
fi

if [[ "${SKIP_TRAIN_EXTRACT}" != "true" ]]; then
  class_dir_count="$(find "${TRAIN_SRC}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  shopt -s nullglob
  tar_files=( "${TRAIN_SRC}"/*.tar )
  shopt -u nullglob

  if (( class_dir_count < 1000 )) && (( ${#tar_files[@]} > 0 )); then
    echo "[INFO] class dirs in train source: ${class_dir_count}, tar files: ${#tar_files[@]}"
    echo "[INFO] extracting class tar files under: ${TRAIN_SRC}"
    for tar_path in "${tar_files[@]}"; do
      class_name="$(basename "${tar_path}" .tar)"
      out_dir="${TRAIN_SRC}/${class_name}"
      if [[ -d "${out_dir}" ]] && find "${out_dir}" -mindepth 1 -maxdepth 1 -type f | grep -q .; then
        continue
      fi
      mkdir -p "${out_dir}"
      tar -xf "${tar_path}" -C "${out_dir}"
    done
    class_dir_count="$(find "${TRAIN_SRC}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    echo "[INFO] class dirs in train source after extraction: ${class_dir_count}"
  else
    echo "[INFO] skip train extraction (class dirs: ${class_dir_count}, tar files: ${#tar_files[@]})"
  fi
fi

FORCE_LABELS_ARGS=()
if [[ "${FORCE_LABELS}" == "true" ]]; then
  FORCE_LABELS_ARGS+=( "--force-labels" )
fi

echo "[INFO] running prepare_imagenet_extra.py ..."
PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
  --root "${ROOT}" \
  --extra "${EXTRA}" \
  --devkit-root "${DEVKIT_ROOT}" \
  --prepare-val \
  --val-src "${VAL_SRC}" \
  "${FORCE_LABELS_ARGS[@]}"

required_files=(
  "${ROOT}/labels.txt"
  "${EXTRA}/entries-TRAIN.npy"
  "${EXTRA}/entries-VAL.npy"
  "${EXTRA}/class-ids-TRAIN.npy"
  "${EXTRA}/class-ids-VAL.npy"
  "${EXTRA}/class-names-TRAIN.npy"
  "${EXTRA}/class-names-VAL.npy"
)

for f in "${required_files[@]}"; do
  require_file "${f}" "required output file"
done

val_class_dirs="$(find "${ROOT}/val" -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "[INFO] validation class directories: ${val_class_dirs}"
echo "[DONE] ImageNet-1k is ready for run_log_regression.sh"
