#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"

NUM_SCENES="${NUM_SCENES:-100}"
TRAIN_COUNT="${TRAIN_COUNT:-80}"
VAL_COUNT="${VAL_COUNT:-10}"
TEST_COUNT="${TEST_COUNT:-10}"
RANDOM_SEED="${RANDOM_SEED:-42}"
BDD_SPLIT="${BDD_SPLIT:-train}"
MAX_TOTAL_SIZE="${MAX_TOTAL_SIZE:-200G}"
DEVICE="${DEVICE:-cuda}"
IMAGE_WIDTH="${IMAGE_WIDTH:-960}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-540}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-20}"

BDD_OUTPUT_DIR="${BDD_OUTPUT_DIR:-data/bdd100k_video_scenes}"
BDD_STGRU_ROOT="${BDD_STGRU_ROOT:-data/bdd100k_stgru}"
BDD_PRECOMPUTE_ROOT="${BDD_PRECOMPUTE_ROOT:-data/stgru_samples_bdd100k}"
STGRU_WEIGHT_DIR="${STGRU_WEIGHT_DIR:-weights/STGRU_BDD100K}"
BDD_DRIVABLE_ROOT="${BDD_DRIVABLE_ROOT:-data/bdd100k_drivable_maps}"
BDD_DRIVABLE_VALUES="${BDD_DRIVABLE_VALUES:-1,2}"
SKIP_REMOTE_LABEL_SELECTION="${SKIP_REMOTE_LABEL_SELECTION:-1}"
YOLOP_CHECKPOINT="${YOLOP_CHECKPOINT:-weights/YOLOP/End-to-end.pth}"
SEA_RAFT_CONFIG="${SEA_RAFT_CONFIG:-src/SEA_RAFT/external/SEA-RAFT/config/eval/spring-M.json}"
SEA_RAFT_CHECKPOINT="${SEA_RAFT_CHECKPOINT:-weights/SEA-RAFT/Tartan-C-T-TSKH-spring540x960-M.pth}"
SEA_RAFT_URL="${SEA_RAFT_URL:-}"

# Examples:
#   COOKIE_ARGS="--cookie-file /workspace/.bdd100k_cookies.txt"
#   URL_ARGS="--video-url <video_zip_url> --label-url <image_label_zip_url> --label-url <drivable_maps_zip_url>"
COOKIE_ARGS="${COOKIE_ARGS:-}"
URL_ARGS="${URL_ARGS:-}"
LABEL_PATTERN_ARGS="${LABEL_PATTERN_ARGS:-}"

download_prepare() {
  local remote_label_args=()
  if [[ "${SKIP_REMOTE_LABEL_SELECTION}" == "1" ]]; then
    remote_label_args+=(--skip-remote-label-selection)
  fi

  python3 download_bdd100k_video_scenes.py \
    ${URL_ARGS} \
    ${COOKIE_ARGS} \
    ${LABEL_PATTERN_ARGS} \
    "${remote_label_args[@]}" \
    --output-dir "${BDD_OUTPUT_DIR}" \
    --num-scenes "${NUM_SCENES}" \
    --split "${BDD_SPLIT}" \
    --max-total-size "${MAX_TOTAL_SIZE}" \
    --selection-mode stratified \
    --random-seed "${RANDOM_SEED}" \
    --prepare-stgru \
    --stgru-output-root "${BDD_STGRU_ROOT}" \
    --local-drivable-root "${BDD_DRIVABLE_ROOT}" \
    --bdd-drivable-values "${BDD_DRIVABLE_VALUES}" \
    --stgru-train-count "${TRAIN_COUNT}" \
    --stgru-val-count "${VAL_COUNT}" \
    --stgru-test-count "${TEST_COUNT}" \
    --stgru-fps 30 \
    --stgru-clip-start 9 \
    --stgru-clip-duration 3 \
    --stgru-center-second 10 \
    --stgru-context-frames 10
}

precompute() {
  python3 src/STGRU/precompute_bdd100k_stgru_samples.py \
    --bdd-stgru-root "${BDD_STGRU_ROOT}" \
    --output-root "${BDD_PRECOMPUTE_ROOT}" \
    --data-root data \
    --splits train,val,test \
    --image-width "${IMAGE_WIDTH}" \
    --image-height "${IMAGE_HEIGHT}" \
    --yolop-checkpoint "${YOLOP_CHECKPOINT}" \
    --sea-raft-config "${SEA_RAFT_CONFIG}" \
    --sea-raft-checkpoint "${SEA_RAFT_CHECKPOINT}" \
    --sea-raft-url "${SEA_RAFT_URL}" \
    --device "${DEVICE}" \
    --overwrite
}

train_stgru() {
  python3 src/STGRU/train_stgru.py \
    --data-root data \
    --sample-list "${BDD_PRECOMPUTE_ROOT}/train.csv" \
    --val-sample-list "${BDD_PRECOMPUTE_ROOT}/val.csv" \
    --test-sample-list "${BDD_PRECOMPUTE_ROOT}/test.csv" \
    --output-dir "${STGRU_WEIGHT_DIR}" \
    --image-width "${IMAGE_WIDTH}" \
    --image-height "${IMAGE_HEIGHT}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --amp
}

case "${MODE}" in
  download|prepare)
    download_prepare
    ;;
  precompute)
    precompute
    ;;
  train)
    train_stgru
    ;;
  all)
    download_prepare
    precompute
    train_stgru
    ;;
  *)
    echo "Usage: $0 [download|prepare|precompute|train|all]"
    exit 1
    ;;
esac
