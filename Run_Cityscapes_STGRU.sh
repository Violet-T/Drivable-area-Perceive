#!/usr/bin/env bash
set -euo pipefail

# Cityscapes STGRU workflow:
# 1. Convert gtFine key-frame labels to binary road/free-space masks for inspection.
# 2. Precompute real YOLOP + SEA-RAFT samples from leftImg8bit_sequence pairs.
# 3. Train STGRU with precomputed current mask, warped history mask, and gtFine supervision.

MODE="${1:-all}"

DEFAULT_MAX_SAMPLES_PER_SPLIT="0"
DEFAULT_EPOCHS="20"
DEFAULT_BATCH_SIZE="2"
DEFAULT_NUM_WORKERS="2"
DEFAULT_PRECOMPUTE_ROOT="data/stgru_samples_cityscapes"
DEFAULT_STGRU_WEIGHT_DIR="weights/STGRU_Cityscapes"
if [[ "${MODE}" == "smoke" ]]; then
  DEFAULT_MAX_SAMPLES_PER_SPLIT="2"
  DEFAULT_EPOCHS="1"
  DEFAULT_BATCH_SIZE="1"
  DEFAULT_NUM_WORKERS="0"
  DEFAULT_PRECOMPUTE_ROOT="data/stgru_samples_cityscapes_smoke"
  DEFAULT_STGRU_WEIGHT_DIR="weights/STGRU_Cityscapes_smoke"
fi

CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-data/cityscapes}"
DATA_ROOT="${DATA_ROOT:-data}"
BINARY_ROOT="${BINARY_ROOT:-data/cityscapes_binary}"
PRECOMPUTE_ROOT="${PRECOMPUTE_ROOT:-${DEFAULT_PRECOMPUTE_ROOT}}"
STGRU_WEIGHT_DIR="${STGRU_WEIGHT_DIR:-${DEFAULT_STGRU_WEIGHT_DIR}}"

SPLITS="${SPLITS:-train,val,test}"
FREE_LABEL_IDS="${FREE_LABEL_IDS:-7}"
PREVIOUS_FRAME_OFFSET="${PREVIOUS_FRAME_OFFSET:-1}"
REQUIRE_SEQUENCE="${REQUIRE_SEQUENCE:-1}"
MAX_SAMPLES_PER_SPLIT="${MAX_SAMPLES_PER_SPLIT:-${DEFAULT_MAX_SAMPLES_PER_SPLIT}}"
OVERWRITE="${OVERWRITE:-1}"

DEVICE="${DEVICE:-cuda}"
IMAGE_WIDTH="${IMAGE_WIDTH:-960}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-540}"
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
EPOCHS="${EPOCHS:-${DEFAULT_EPOCHS}}"
NUM_WORKERS="${NUM_WORKERS:-${DEFAULT_NUM_WORKERS}}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-16}"
AMP="${AMP:-1}"

YOLOP_REPO="${YOLOP_REPO:-src/YOLOP/external/YOLOP}"
YOLOP_CHECKPOINT="${YOLOP_CHECKPOINT:-weights/YOLOP/End-to-end.pth}"
YOLOP_IMG_SIZE="${YOLOP_IMG_SIZE:-640}"
SEA_RAFT_REPO="${SEA_RAFT_REPO:-src/SEA_RAFT/external/SEA-RAFT}"
SEA_RAFT_CONFIG="${SEA_RAFT_CONFIG:-src/SEA_RAFT/external/SEA-RAFT/config/eval/spring-M.json}"
SEA_RAFT_CHECKPOINT="${SEA_RAFT_CHECKPOINT:-weights/SEA-RAFT/Tartan-C-T-TSKH-spring540x960-M.pth}"
SEA_RAFT_URL="${SEA_RAFT_URL:-}"

require_sequence_args=()
if [[ "${REQUIRE_SEQUENCE}" == "1" ]]; then
  require_sequence_args+=(--require-sequence)
fi

overwrite_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

amp_args=()
if [[ "${AMP}" == "1" ]]; then
  amp_args+=(--amp)
fi

check_cityscapes_layout() {
  if [[ ! -d "${CITYSCAPES_ROOT}/gtFine" ]]; then
    echo "Missing Cityscapes gtFine directory: ${CITYSCAPES_ROOT}/gtFine" >&2
    exit 1
  fi
  if [[ ! -d "${CITYSCAPES_ROOT}/leftImg8bit_sequence" ]]; then
    echo "Missing Cityscapes sequence directory: ${CITYSCAPES_ROOT}/leftImg8bit_sequence" >&2
    echo "Download and extract leftImg8bit_sequence_trainvaltest.zip first." >&2
    exit 1
  fi
}

prepare_binary() {
  python3 src/STGRU/prepare_cityscapes_binary.py \
    --cityscapes-root "${CITYSCAPES_ROOT}" \
    --output-root "${BINARY_ROOT}" \
    --splits "${SPLITS}" \
    --free-label-ids "${FREE_LABEL_IDS}" \
    "${overwrite_args[@]}"
}

precompute() {
  check_cityscapes_layout
  python3 src/STGRU/precompute_stgru_samples.py \
    --cityscapes-root "${CITYSCAPES_ROOT}" \
    --output-root "${PRECOMPUTE_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --splits "${SPLITS}" \
    --max-samples-per-split "${MAX_SAMPLES_PER_SPLIT}" \
    "${require_sequence_args[@]}" \
    --previous-frame-offset "${PREVIOUS_FRAME_OFFSET}" \
    --image-width "${IMAGE_WIDTH}" \
    --image-height "${IMAGE_HEIGHT}" \
    --free-label-ids "${FREE_LABEL_IDS}" \
    --device "${DEVICE}" \
    --yolop-repo "${YOLOP_REPO}" \
    --yolop-checkpoint "${YOLOP_CHECKPOINT}" \
    --yolop-img-size "${YOLOP_IMG_SIZE}" \
    --sea-raft-repo "${SEA_RAFT_REPO}" \
    --sea-raft-config "${SEA_RAFT_CONFIG}" \
    --sea-raft-checkpoint "${SEA_RAFT_CHECKPOINT}" \
    --sea-raft-url "${SEA_RAFT_URL}" \
    "${overwrite_args[@]}"
}

train_stgru() {
  python3 src/STGRU/train_stgru.py \
    --data-root "${DATA_ROOT}" \
    --sample-list "${PRECOMPUTE_ROOT}/train.csv" \
    --val-sample-list "${PRECOMPUTE_ROOT}/val.csv" \
    --test-sample-list "${PRECOMPUTE_ROOT}/test.csv" \
    --output-dir "${STGRU_WEIGHT_DIR}" \
    --image-width "${IMAGE_WIDTH}" \
    --image-height "${IMAGE_HEIGHT}" \
    --free-label-ids "${FREE_LABEL_IDS}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --hidden-channels "${HIDDEN_CHANNELS}" \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    "${amp_args[@]}"
}

smoke() {
  prepare_binary
  precompute
  train_stgru
}

case "${MODE}" in
  binary)
    prepare_binary
    ;;
  precompute)
    precompute
    ;;
  train)
    train_stgru
    ;;
  all)
    prepare_binary
    precompute
    train_stgru
    ;;
  smoke)
    smoke
    ;;
  *)
    echo "Usage: $0 [binary|precompute|train|all|smoke]" >&2
    exit 1
    ;;
esac
