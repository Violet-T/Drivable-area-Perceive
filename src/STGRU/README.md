# STGRU

This module contains the trainable temporal fusion layer for the project.

The implemented structure follows the paper idea from **Semantic Video
Segmentation by Gated Recurrent Flow Propagation**:

```text
current YOLOP probability
+ SEA-RAFT warped historical probability
+ photometric error after image warp
→ reset gate
→ candidate hidden state
→ update gate
→ fused free-space probability
```

Current status:

- `STGRUCell` is implemented as a PyTorch module.
- `STGRUFusionModule` exposes a numpy-facing wrapper for the pipeline.
- `train_stgru.py` provides a train entry for Cityscapes sequence bootstrap
  training and precomputed YOLOP + SEA-RAFT samples.
- If no checkpoint is provided, the STGRU weights are untrained and should only
  be used for structure smoke tests.
- For quantitative experiments, train STGRU and load its checkpoint with the
  pipeline `--stgru-checkpoint` argument.

Recommended training data layout:

```text
data/cityscapes/
├── leftImg8bit_sequence/
│   ├── train/
│   └── val/
└── gtFine/
    ├── train/
    └── val/
```

Bootstrap training command:

```bash
python3 src/STGRU/train_stgru.py \
  --cityscapes-root data/cityscapes \
  --output-dir weights/STGRU \
  --image-width 960 \
  --image-height 540 \
  --free-label-ids 7 \
  --epochs 20 \
  --batch-size 2 \
  --device cuda \
  --amp
```

Final experiment training should use precomputed real pipeline samples:

```bash
python3 src/STGRU/train_stgru.py \
  --data-root data \
  --sample-list data/stgru_samples/train.csv \
  --val-sample-list data/stgru_samples/val.csv \
  --output-dir weights/STGRU \
  --target-is-cityscapes-label \
  --device cuda
```

CSV columns for precomputed samples:

```text
current_mask,warped_mask,target_mask,current_image,warped_previous_image,photometric_error
```

Required columns are `current_mask`, `warped_mask`, and `target_mask`.
`photometric_error` can be provided directly. Otherwise the script computes it
from `current_image` and `warped_previous_image`; if neither is available, it
falls back to the mask difference.

Paper-aligned workflow:

```bash
# 1. Convert Cityscapes multi-class labelIds into binary road/free-space masks.
python3 src/STGRU/prepare_cityscapes_binary.py \
  --cityscapes-root data/cityscapes \
  --output-root data/cityscapes_binary \
  --free-label-ids 7

# 2. Precompute real YOLOP + SEA-RAFT inputs for STGRU.
python3 src/STGRU/precompute_stgru_samples.py \
  --cityscapes-root data/cityscapes \
  --output-root data/stgru_samples \
  --data-root data \
  --splits train,val,test \
  --image-width 960 \
  --image-height 540 \
  --device cuda

# 3. Train STGRU and evaluate val/test metrics.
python3 src/STGRU/train_stgru.py \
  --data-root data \
  --sample-list data/stgru_samples/train.csv \
  --val-sample-list data/stgru_samples/val.csv \
  --test-sample-list data/stgru_samples/test.csv \
  --output-dir weights/STGRU \
  --image-width 960 \
  --image-height 540 \
  --device cuda \
  --amp
```

Convenience wrapper:

```bash
./Run_STGRU.sh precompute-smoke
./Run_STGRU.sh train-precomputed-smoke

./Run_STGRU.sh precompute
./Run_STGRU.sh train-precomputed

STGRU_CHECKPOINT=/workspace/weights/STGRU/stgru_best.pth \
INPUT_VIDEO=/workspace/data/demo/1.mp4 \
./Run_STGRU.sh video
```

The precomputed mode is the recommended paper-aligned path because STGRU sees
real YOLOP masks, SEA-RAFT warped history, and binary Cityscapes supervision.
