# Temporal Free-Space Perception

This project studies temporal consistency enhancement for monocular free-space perception.

Current pipeline:

```text
Camera image sequence
→ YOLOP single-frame drivable/free-space mask
→ SEA-RAFT optical flow
→ historical mask warping
→ temporal fusion
→ OpenCV visualization
→ temporal consistency metrics
```

Out of current scope:

```text
BEV / IPM
OccupancyGrid
planning-ready occupancy mapping
RViz-first visualization
mandatory ROS2 integration
```

## Repository Layout

```text
.
├── src/
│   ├── YOLOP/           project wrapper plus YOLOP official repository
│   ├── SEA_RAFT/        SEA-RAFT wrapper plus SEA-RAFT official repository
│   ├── STGRU/           future trainable temporal fusion module
│   └── utils/           temporal ops, visualization, datasets, helper scripts, legacy code
├── weights/             local model checkpoints, ignored by Git
├── data/                local datasets, ignored by Git
├── output/              generated experiment outputs, ignored by Git
├── docs/
├── Dockerfile
└── Run.sh
```

`SEA_RAFT` uses an underscore because Python package names cannot contain `-`.

## Docker

Build on the host:

```bash
docker build -t freespace_temporal:latest .
```

Run on the host:

```bash
docker run --gpus all -it --rm \
  --name perceive \
  -v $(pwd):/workspace \
  freespace_temporal:latest
```

All project commands should run inside the container:

```bash
cd /workspace
```

If generated files become owned by root on the host, fix ownership from the host with:

```bash
docker exec perceive chown -R $(id -u):$(id -g) /workspace/output /workspace/data /workspace/weights
```

## YOLOP Single-Frame Baseline

```bash
python3 src/YOLOP/run_yolop_inference.py \
  --input-video /workspace/data/demo/eg.mp4 \
  --checkpoint /workspace/weights/YOLOP/End-to-end.pth \
  --output-dir /workspace/output/yolop_eg_baseline \
  --mask-mode probability \
  --save-frames \
  --vis-threshold 0.5 \
  --vis-binary \
  --device cuda \
  --overwrite
```

## YOLOP + SEA-RAFT No-STGRU Baseline

```bash
python3 src/SEA_RAFT/run_yolop_temporal_fusion.py \
  --input-video /workspace/data/demo/eg.mp4 \
  --output-dir /workspace/output/yolop_eg_temporal \
  --alpha 0.7 \
  --non-free-threshold 0.2 \
  --history-size 3 \
  --history-decay 0.6 \
  --device cuda \
  --yolop-checkpoint /workspace/weights/YOLOP/End-to-end.pth \
  --sea-raft-config /workspace/src/SEA_RAFT/external/SEA-RAFT/config/eval/spring-S.json \
  --sea-raft-url MemorySlices/Tartan-C-T-TSKH-spring540x960-S \
  --save-frames \
  --save-arrays \
  --vis-threshold 0.5 \
  --vis-binary \
  --overwrite
```

The current temporal fusion is still an alpha-based baseline. The paper reproduction direction is to replace this hand-designed fusion with a trainable STGRU/GRFP module while keeping:

```text
YOLOP = single-frame drivable-area predictor
SEA-RAFT = optical-flow estimator
warp = temporal alignment operator
STGRU = trainable temporal fusion module
```

## BDD100K STGRU Training

Remote SSH deployment and training steps are documented in:

```text
docs/remote_training_ssh.md
```

Small smoke run inside Docker:

```bash
NUM_SCENES=5 \
TRAIN_COUNT=3 \
VAL_COUNT=1 \
TEST_COUNT=1 \
MAX_TOTAL_SIZE=20G \
EPOCHS=1 \
BATCH_SIZE=2 \
./Run_BDD100K_STGRU.sh all
```

Formal 100-scene run:

```bash
NUM_SCENES=100 \
TRAIN_COUNT=80 \
VAL_COUNT=10 \
TEST_COUNT=10 \
MAX_TOTAL_SIZE=200G \
EPOCHS=20 \
BATCH_SIZE=2 \
./Run_BDD100K_STGRU.sh all
```

Required local assets:

```text
data/bdd100k_drivable_maps/
```

The repository includes the project-required YOLOP and SEA-RAFT `.pth` checkpoints:

```text
weights/YOLOP/End-to-end.pth
weights/SEA-RAFT/Tartan-C-T-TSKH-spring540x960-M.pth
```

By default, BDD100K precompute uses the Hugging Face SEA-RAFT `spring-S` model URL. To force the local `spring-M` checkpoint:

```bash
SEA_RAFT_CONFIG=src/SEA_RAFT/external/SEA-RAFT/config/eval/spring-M.json \
SEA_RAFT_CHECKPOINT=weights/SEA-RAFT/Tartan-C-T-TSKH-spring540x960-M.pth \
SEA_RAFT_URL= \
./Run_BDD100K_STGRU.sh precompute
```
