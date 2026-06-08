#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
CONTAINER_NAME="${CONTAINER_NAME:-perceive}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' is not running."
  echo "Start it first, for example:"
  echo "docker run --gpus all -it --name ${CONTAINER_NAME} -v \$(pwd):/workspace freespace_temporal:latest"
  exit 1
fi

case "${MODE}" in
  smoke)
    OUTPUT_DIR="/workspace/weights/STGRU_smoke"
    EPOCHS="${EPOCHS:-1}"
    IMAGE_WIDTH="${IMAGE_WIDTH:-480}"
    IMAGE_HEIGHT="${IMAGE_HEIGHT:-270}"
    MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-64}"
    MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-32}"
    ;;
  train)
    OUTPUT_DIR="/workspace/weights/STGRU"
    EPOCHS="${EPOCHS:-20}"
    IMAGE_WIDTH="${IMAGE_WIDTH:-960}"
    IMAGE_HEIGHT="${IMAGE_HEIGHT:-540}"
    MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
    MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"
    ;;
  precompute-smoke)
    docker exec "${CONTAINER_NAME}" bash -lc "
cd /workspace && rm -rf /workspace/data/stgru_samples_smoke && \
python3 src/STGRU/precompute_stgru_samples.py \
  --cityscapes-root /workspace/data/cityscapes \
  --output-root /workspace/data/stgru_samples_smoke \
  --data-root /workspace/data \
  --splits train \
  --max-samples-per-split ${MAX_TRAIN_SAMPLES:-2} \
  --require-sequence \
  --image-width ${IMAGE_WIDTH:-960} \
  --image-height ${IMAGE_HEIGHT:-540} \
  --device ${DEVICE} \
  --overwrite && \
python3 src/STGRU/precompute_stgru_samples.py \
  --cityscapes-root /workspace/data/cityscapes \
  --output-root /workspace/data/stgru_samples_smoke \
  --data-root /workspace/data \
  --splits val,test \
  --max-samples-per-split ${MAX_VAL_SAMPLES:-1} \
  --image-width ${IMAGE_WIDTH:-960} \
  --image-height ${IMAGE_HEIGHT:-540} \
  --device ${DEVICE} \
  --overwrite
"
    docker exec "${CONTAINER_NAME}" chown -R "$(id -u):$(id -g)" /workspace/data/stgru_samples_smoke 2>/dev/null || true
    exit 0
    ;;
  precompute)
    docker exec "${CONTAINER_NAME}" bash -lc "
cd /workspace && python3 src/STGRU/precompute_stgru_samples.py \
  --cityscapes-root /workspace/data/cityscapes \
  --output-root /workspace/data/stgru_samples \
  --data-root /workspace/data \
  --splits train \
  --require-sequence \
  --image-width ${IMAGE_WIDTH:-960} \
  --image-height ${IMAGE_HEIGHT:-540} \
  --device ${DEVICE} \
  --overwrite && \
python3 src/STGRU/precompute_stgru_samples.py \
  --cityscapes-root /workspace/data/cityscapes \
  --output-root /workspace/data/stgru_samples \
  --data-root /workspace/data \
  --splits val,test \
  --image-width ${IMAGE_WIDTH:-960} \
  --image-height ${IMAGE_HEIGHT:-540} \
  --device ${DEVICE} \
  --overwrite
"
    docker exec "${CONTAINER_NAME}" chown -R "$(id -u):$(id -g)" /workspace/data/stgru_samples 2>/dev/null || true
    exit 0
    ;;
  train-precomputed)
    docker exec "${CONTAINER_NAME}" bash -lc "
cd /workspace && python3 src/STGRU/train_stgru.py \
  --data-root /workspace/data \
  --sample-list /workspace/data/stgru_samples/train.csv \
  --val-sample-list /workspace/data/stgru_samples/val.csv \
  --test-sample-list /workspace/data/stgru_samples/test.csv \
  --output-dir /workspace/weights/STGRU \
  --image-width ${IMAGE_WIDTH:-960} \
  --image-height ${IMAGE_HEIGHT:-540} \
  --epochs ${EPOCHS:-20} \
  --batch-size ${BATCH_SIZE} \
  --num-workers ${NUM_WORKERS} \
  --device ${DEVICE} \
  --amp
"
    docker exec "${CONTAINER_NAME}" chown -R "$(id -u):$(id -g)" /workspace/weights/STGRU 2>/dev/null || true
    exit 0
    ;;
  train-precomputed-smoke)
    docker exec "${CONTAINER_NAME}" bash -lc "
cd /workspace && rm -rf /workspace/weights/STGRU_precompute_smoke && python3 src/STGRU/train_stgru.py \
  --data-root /workspace/data \
  --sample-list /workspace/data/stgru_samples_smoke/train.csv \
  --val-sample-list /workspace/data/stgru_samples_smoke/val.csv \
  --test-sample-list /workspace/data/stgru_samples_smoke/test.csv \
  --output-dir /workspace/weights/STGRU_precompute_smoke \
  --image-width ${IMAGE_WIDTH:-960} \
  --image-height ${IMAGE_HEIGHT:-540} \
  --epochs ${EPOCHS:-1} \
  --batch-size ${BATCH_SIZE:-1} \
  --num-workers 0 \
  --device ${DEVICE} \
  --amp
"
    docker exec "${CONTAINER_NAME}" chown -R "$(id -u):$(id -g)" /workspace/weights/STGRU_precompute_smoke 2>/dev/null || true
    exit 0
    ;;
  video)
    docker exec "${CONTAINER_NAME}" bash -lc "
cd /workspace && python3 src/STGRU/run_stgru_video.py \
  --input-video ${INPUT_VIDEO:-/workspace/data/demo/1.mp4} \
  --output-dir ${OUTPUT_DIR:-/workspace/output/stgru_video} \
  --stgru-checkpoint ${STGRU_CHECKPOINT:-/workspace/weights/STGRU/stgru_best.pth} \
  --target-width ${IMAGE_WIDTH:-960} \
  --target-height ${IMAGE_HEIGHT:-540} \
  --device ${DEVICE} \
  --overwrite
"
    docker exec "${CONTAINER_NAME}" chown -R "$(id -u):$(id -g)" "${OUTPUT_DIR:-/workspace/output/stgru_video}" 2>/dev/null || true
    exit 0
    ;;
  *)
    echo "Usage: $0 [smoke|train|precompute-smoke|precompute|train-precomputed-smoke|train-precomputed|video]"
    exit 1
    ;;
esac

docker exec "${CONTAINER_NAME}" bash -lc "
cd /workspace && python3 src/STGRU/train_stgru.py \
  --cityscapes-root /workspace/data/cityscapes \
  --output-dir ${OUTPUT_DIR} \
  --image-width ${IMAGE_WIDTH} \
  --image-height ${IMAGE_HEIGHT} \
  --free-label-ids 7 \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --num-workers ${NUM_WORKERS} \
  --max-train-samples ${MAX_TRAIN_SAMPLES} \
  --max-val-samples ${MAX_VAL_SAMPLES} \
  --device ${DEVICE} \
  --amp
"

docker exec "${CONTAINER_NAME}" chown -R "$(id -u):$(id -g)" /workspace/weights/STGRU /workspace/weights/STGRU_smoke 2>/dev/null || true
