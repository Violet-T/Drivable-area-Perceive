# AGENTS.md

## Project Overview

This project focuses on **temporal consistency enhancement for monocular free-space perception** in dynamic traffic scenes.

The project scope has been narrowed down to:

```text
Single-frame free-space perception
+
Optical-flow-based temporal alignment
+
Temporal fusion
+
OpenCV visualization
+
Temporal consistency evaluation
```

The project no longer includes:

```text
BEV conversion
IPM projection
OccupancyGrid mapping
3D occupancy prediction
LiDAR / radar fusion
trajectory planning
vehicle control
RViz visualization
mandatory ROS2 integration
```

Core pipeline:

```text
Camera Image Sequence
→ YOLOP Single-frame Drivable / Free-space Perception
→ Raw Free-space Mask / Probability Map
→ SEA-RAFT Optical Flow Estimation
→ Historical Mask Warping
→ Temporal Fusion
→ Stable Free-space Mask
→ OpenCV Visualization
→ Temporal Consistency Evaluation
```

Core research question:

```text
Can SEA-RAFT-based temporal alignment and fusion improve the frame-to-frame continuity,
boundary stability, and flicker suppression of monocular free-space segmentation results
in dynamic traffic scenes?
```

Final project claim:

```text
Compared with single-frame YOLOP free-space perception, adding SEA-RAFT-based temporal alignment
and temporal fusion improves frame-to-frame consistency, reduces boundary jitter, and suppresses
short-term mask flicker in dynamic traffic scenes.
```

Do NOT claim:

```text
This project solves BEV perception.
This project replaces multi-sensor fusion.
This project outputs planning-ready occupancy maps.
This project solves all adverse-weather perception problems.
```

---

## Environment Rules

The project must run inside a self-built Docker environment that is compatible with the whole project.

Required principles:

- Use Ubuntu 22.04 as the base system.
- Use Python as the primary implementation language.
- Use PyTorch with CUDA support for YOLOP and SEA-RAFT.
- Use OpenCV and NumPy for mask processing, warping, visualization, video export, and metric computation.
- Do NOT use conda.
- Do NOT rely on host-machine Python packages.
- Build, run, debug, train, and evaluate inside the Docker container.
- Keep dependencies lightweight and reproducible.
- Record the Docker image name, CUDA version, PyTorch version, and major dependency versions in `docs/dev_log.md`.

The Docker environment should support:

```text
YOLOP training / inference
SEA-RAFT inference
mask warping
temporal fusion
OpenCV visualization
metric calculation
video export
```

Preferred workflow:

```bash
docker build -t freespace_temporal:latest .

docker run --gpus all -it --rm \
    -v <project_path>:/workspace \
    freespace_temporal:latest
```

All project commands should be executed inside the container.

---

## Project Structure Rules

The project must remain modular and easy to debug.

Suggested structure:

```text
src/
├── YOLOP/
│   ├── external/
│   │   └── YOLOP/
│   ├── yolop_wrapper.py
│   └── run_yolop_inference.py
├── SEA_RAFT/
│   ├── external/
│   │   └── SEA-RAFT/
│   ├── sea_raft_wrapper.py
│   └── run_yolop_temporal_fusion.py
├── STGRU/
│   └── README.md
├── utils/
│   ├── temporal/
│   │   ├── warp.py
│   │   ├── fusion.py
│   │   └── mask_buffer.py
│   ├── visualization/
│   │   └── overlay.py
│   ├── datasets/
│   ├── scripts/
│   └── legacy/
│       └── external/
│           └── PIDNet/
```

Top-level layout:

```text
AGENTS.md
README.md
Dockerfile
src/
docs/
data/
weights/
output/
```

Do NOT place temporary experiment files, raw datasets, or large model weights directly in the repository root.

Recommended paths:

```text
data/            local dataset path, ignored by Git except directory notes
weights/         local pretrained checkpoint path, ignored by Git except directory notes
output/          generated videos, curves, tables, and logs
docs/dev_log.md continuous development log
```

---

## External Model Rules

External model repositories should be cloned under the module that owns them:

```text
src/YOLOP/external/YOLOP/
src/SEA_RAFT/external/SEA-RAFT/
```

Legacy external repositories that are no longer part of the main pipeline should live under:

```text
src/utils/legacy/external/
```

Required external models:

```text
YOLOP
SEA-RAFT
```

Rules:

- Do NOT rewrite YOLOP or SEA-RAFT internals unless explicitly requested.
- Local code should mainly provide wrappers, preprocessing, postprocessing, temporal fusion, visualization, and evaluation.
- Record the repository URL, commit hash, pretrained weight name, and any local patch in `docs/dev_log.md`.
- If a pretrained checkpoint is used, record the source, task, dataset, and output class definition.

---

## YOLOP Rules

YOLOP is responsible only for **single-frame drivable / free-space perception** in this project.

Input:

```text
Current RGB image frame
```

Output:

```text
Raw free-space mask or probability map
```

Recommended output format:

```text
float32
shape: H × W
value range: [0, 1]

1 = free-space / drivable
0 = non-free / background / obstacle
```

YOLOP must NOT handle:

```text
temporal information
optical flow estimation
mask warping
temporal fusion
BEV projection
planning
control
```

YOLOP may run detection and lane heads internally because the pretrained checkpoint is multi-task, but project-owned wrappers must expose only the drivable-area segmentation output.

Project visualization must not draw YOLOP detection boxes unless explicitly requested for a separate diagnostic.

### YOLOP Dataset Rules

Preferred training dataset:

```text
BDD100K images
+ drivable-area segmentation labels
+ optional lane-line labels
+ optional detection labels
```

For binary free-space training:

```text
num_classes = 2
0 = non-free
1 = free-space
```

If using the pretrained YOLOP BDD100K checkpoint directly:

```text
YOLOP drivable area segmentation head
→ extract drivable/free-space probability
→ convert to project free-space mask
```

If fine-tuning:

```text
load pretrained YOLOP checkpoint
train or fine-tune drivable-area segmentation branch
keep detection/lane branches optional
export only drivable/free-space mask for this project
```

The current project should prioritize binary free-space perception.

Do NOT expand to full multi-class semantic segmentation unless explicitly requested.

---

## SEA-RAFT Rules

SEA-RAFT is responsible only for **optical flow estimation**.

Input:

```text
Image_{t-1}
Image_t
```

Output:

```text
Flow_{t-1→t}
```

Flow format:

```text
float32
shape: H × W × 2

flow[..., 0] = dx
flow[..., 1] = dy
```

SEA-RAFT must NOT output semantic classes.

SEA-RAFT must NOT decide whether an area is drivable.

SEA-RAFT must NOT perform:

```text
semantic segmentation
prediction
planning
control
```

SEA-RAFT is only used to estimate pixel-level motion between consecutive frames so that historical masks can be aligned to the current frame.

---

## Warp Rules

Warp is used for temporal alignment.

Input:

```text
previous_mask = Mask_{t-1}
flow = Flow_{t-1→t}
```

Output:

```text
warped_mask_t = Warp(Mask_{t-1}, Flow_{t-1→t})
```

Purpose:

```text
Align historical free-space perception result to the current frame.
```

Preferred implementation:

```text
cv2.remap
or
torch.grid_sample
```

For probability masks:

```text
bilinear interpolation is allowed
```

For discrete label maps:

```text
nearest-neighbor interpolation is recommended
```

Warp does NOT create new semantic information.

Warp only propagates historical information according to estimated pixel motion.

---

## Temporal Fusion Rules

Temporal fusion combines:

```text
current_mask = current YOLOP free-space result
warped_mask = previous mask aligned to current frame
```

Default fusion:

```text
fused_mask = alpha * current_mask + (1 - alpha) * warped_mask
```

Default parameter:

```text
alpha = 0.7
```

The fusion weight must be configurable.

### Dynamic Obstacle Safety Rule

Historical free-space must not override current high-confidence non-free predictions.

Recommended rule:

```text
if current_mask < non_free_threshold:
    fused_mask = current_mask
else:
    fused_mask = alpha * current_mask + (1 - alpha) * warped_mask
```

Default:

```text
non_free_threshold = 0.2
```

Purpose:

```text
Prevent historical road/free-space regions from being incorrectly propagated onto newly appearing vehicles,
pedestrians, or other obstacles.
```

---

## Visualization Rules

Visualization should be implemented with **OpenCV**, not RViz.

The purpose of visualization is to show whether temporal fusion reduces:

```text
frame-to-frame flicker
boundary jitter
local mask holes
unnecessary region changes
```

Required visualization outputs:

```text
output/<sequence_name>/videos/raw_overlay.mp4
output/<sequence_name>/videos/fused_overlay.mp4
output/<sequence_name>/videos/raw_vs_fused_comparison.mp4
output/<sequence_name>/videos/temporal_panel.mp4
output/<sequence_name>/videos/diff_heatmap_comparison.mp4
```

### Recommended Main Visualization Panel

The main visualization video should use this layout:

```text
┌──────────────────────────┬──────────────────────────┐
│ Current RGB Image         │ Raw YOLOP Mask Overlay   │
├──────────────────────────┼──────────────────────────┤
│ Warped Historical Mask    │ Fused Mask Overlay       │
├──────────────────────────┼──────────────────────────┤
│ Raw Frame Difference      │ Fused Frame Difference   │
└──────────────────────────┴──────────────────────────┘
```

Overlay rules:

```text
green transparent region = free-space
yellow contour = free-space boundary
red heatmap = unstable / changed region
```

OpenCV visualization code should be placed in:

```text
src/utils/visualization/
```

Visualization must be exportable as videos or images.

Do NOT make visualization depend on ROS, RViz, or GUI-only tools.

---

## Quantitative Evaluation Rules

The project evaluates **temporal consistency**, not BEV accuracy.

Baseline:

```text
YOLOP single-frame output
```

Proposed method:

```text
YOLOP + SEA-RAFT + temporal fusion
```

The evaluation must compare:

```text
raw masks
vs
fused masks
```

Required metrics:

```text
1. Temporal IoU
2. Frame Difference
3. Boundary Jitter
4. Flicker Rate
5. Accuracy Preservation Metric, if ground truth is available
```

---

### Metric 1: Temporal IoU

Definition:

```text
TemporalIoU_t = IoU(M_t, Warp(M_{t-1}, Flow_{t-1→t}))
```

Compare:

```text
TemporalIoU_raw
TemporalIoU_fused
```

Expected result:

```text
TemporalIoU_fused > TemporalIoU_raw
```

Purpose:

```text
Measure motion-compensated frame-to-frame consistency.
```

---

### Metric 2: Frame Difference

Definition:

```text
FrameDiff_t = mean(|M_t - Warp(M_{t-1}, Flow_{t-1→t})|)
```

Compare:

```text
FrameDiff_raw
FrameDiff_fused
```

Expected result:

```text
FrameDiff_fused < FrameDiff_raw
```

Purpose:

```text
Measure unnecessary frame-to-frame mask variation.
```

---

### Metric 3: Boundary Jitter

Simplified definition:

```text
BoundaryJitter_t = mean(|Edge(M_t) - Edge(Warp(M_{t-1}, Flow_{t-1→t}))|)
```

Alternative implementations may use:

```text
Chamfer Distance
Hausdorff Distance
Boundary F-score fluctuation
```

Compare:

```text
BoundaryJitter_raw
BoundaryJitter_fused
```

Expected result:

```text
BoundaryJitter_fused < BoundaryJitter_raw
```

Purpose:

```text
Measure free-space boundary stability.
```

---

### Metric 4: Flicker Rate

Definition:

```text
FlickerRate = number_of_pixels_with_repeated_state_switches / total_pixels
```

Repeated state switch examples:

```text
free → non-free → free
non-free → free → non-free
```

Use a 3-frame or 5-frame temporal window.

Compare:

```text
FlickerRate_raw
FlickerRate_fused
```

Expected result:

```text
FlickerRate_fused < FlickerRate_raw
```

Purpose:

```text
Measure short-term segmentation flicker.
```

---

### Metric 5: Accuracy Preservation

If ground-truth labels are available, report:

```text
mIoU
drivable IoU
F1-score
Precision
Recall
```

Purpose:

```text
Verify that temporal smoothing improves stability without significantly damaging segmentation accuracy.
```

Expected result:

```text
Temporal stability improves,
while mIoU / drivable IoU remains stable or decreases only slightly.
```

The temporal layer must not simply over-smooth all results.

---

## Evaluation Output Rules

Every evaluated video sequence should generate:

```text
output/<sequence_name>/
├── videos/
│   ├── raw_overlay.mp4
│   ├── fused_overlay.mp4
│   ├── raw_vs_fused_comparison.mp4
│   ├── temporal_panel.mp4
│   └── diff_heatmap_comparison.mp4
├── metrics/
│   ├── temporal_iou.csv
│   ├── frame_difference.csv
│   ├── boundary_jitter.csv
│   ├── flicker_rate.csv
│   └── summary.csv
└── figures/
    ├── temporal_iou_curve.png
    ├── frame_difference_curve.png
    ├── boundary_jitter_curve.png
    └── flicker_rate_curve.png
```

Summary tables should compare:

```text
YOLOP raw
YOLOP + SEA-RAFT fused
```

Recommended scene categories:

```text
normal road
front vehicle occlusion
pedestrian crossing
vehicle cut-in
shadow / illumination change
rain reflection
snow / low-contrast boundary
nighttime
```

---

## Development Strategy

Always implement incrementally.

Recommended order:

```text
1. Load video sequence and export frames.
2. Run YOLOP single-frame inference and save raw masks.
3. Run SEA-RAFT on consecutive RGB frames and save flow fields.
4. Warp previous raw mask to the current frame.
5. Implement alpha-based temporal fusion.
6. Add dynamic obstacle safety rule.
7. Export raw vs fused overlay videos.
8. Export raw vs fused difference heatmaps.
9. Compute Temporal IoU and Frame Difference.
10. Compute Boundary Jitter and Flicker Rate.
11. Generate metric curves and summary tables.
12. Write experiment conclusions in docs/dev_log.md.
```

Never attempt full implementation in one step.

---

## Coding Rules

- Explain modifications before coding.
- Keep files small and modular.
- Add comments for tensor/image shapes.
- Add logging and exception handling.
- Prefer readability over excessive abstraction.
- Do not refactor external model code unless explicitly requested.
- Do not introduce BEV, IPM, OccupancyGrid, planning, or control modules.
- Intermediate tensors and images must have clear shape comments.
- All scripts must support command-line arguments for input path, output path, checkpoint path, and config path.

Recommended CLI style:

```bash
python scripts/run_temporal_fusion.py \
    --input_video data/demo.mp4 \
    --yolop_ckpt /weights/yolop_end_to_end.pth \
    --sea_raft_ckpt /weights/sea_raft.pt \
    --output_dir output/demo_sequence \
    --alpha 0.7
```

---

## Development Logging Rules

The project must maintain a continuous development log.

Required file:

```text
docs/dev_log.md
```

The development log must be written primarily in Chinese.

After every major modification or experiment, append a new entry.

Each entry should include:

```text
## Date

## Modified Module
Examples:
- YOLOP
- SEA-RAFT
- Warp
- Temporal Fusion
- Visualization
- Metrics

## Changes
Describe what algorithm step was implemented or modified.

## Reason
Explain why the change is needed and which temporal consistency problem it addresses.

## Result
Describe the expected or observed effect.

## Known Issues
Describe unresolved problems or future optimization directions.
```

Logging style:

```text
concise
engineering-oriented
focused on algorithm progress, validation, and debugging
avoid excessive theory
append-only chronological order
```

---

## Language Rules

Use Chinese for development logs, comments explaining project-specific logic, and experiment conclusions.

Allowed English terms:

```text
YOLOP
SEA-RAFT
warp
flow
mask
fused_mask
Temporal IoU
Frame Difference
Boundary Jitter
Flicker Rate
OpenCV
PyTorch
CUDA
```

---

## Important Constraints

This project is not a full autonomous driving stack.

This project is a temporal-consistency enhancement module for the visual free-space perception branch.

Do NOT add:

```text
BEV
IPM
OccupancyGrid
LiDAR / radar fusion
trajectory planning
vehicle control
RViz dependency
ROS2 mandatory dependency
```

unless explicitly requested by the project owner.
