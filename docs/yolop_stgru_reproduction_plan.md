# YOLOP 与 STGRU 复现路线

## 1. 当前项目主线

当前主线调整为：

```text
RGB video
→ YOLOP drivable-area mask
→ SEA-RAFT optical flow
→ warp historical mask / hidden state
→ temporal fusion
→ temporal consistency evaluation
```

当前 alpha fusion 只是 baseline。论文复现目标是把手工融合替换为可训练的 STGRU / GRFP。

## 2. YOLOP 需要什么训练数据

YOLOP 官方训练使用 BDD100K 风格多任务数据：

```text
dataset_root/
├── images/
│   ├── train/
│   └── val/
├── det_annotations/
│   ├── train/
│   └── val/
├── da_seg_annotations/
│   ├── train/
│   └── val/
└── ll_seg_annotations/
    ├── train/
    └── val/
```

对应 YOLOP 配置项在：

```text
src/YOLOP/external/YOLOP/lib/config/default.py
```

关键字段：

```text
DATASET.DATAROOT   = images
DATASET.LABELROOT  = det_annotations
DATASET.MASKROOT   = da_seg_annotations
DATASET.LANEROOT   = ll_seg_annotations
DATASET.TRAIN_SET  = train
DATASET.TEST_SET   = val
```

本项目只关心 drivable/free-space，因此最小训练目标是：

```text
RGB image
+ drivable-area binary mask
```

但 YOLOP 原始训练脚本仍会读取 detection label 和 lane label。若不改 YOLOP 外部代码，最省事做法是保留四类目录：

```text
det_annotations: 可以为空框 json，但格式必须满足读取
da_seg_annotations: 必须有可通行区域 mask
ll_seg_annotations: 可以为空 lane mask
```

更干净的做法是后续新增项目内 fine-tune wrapper，只训练 YOLOP drivable 分支，跳过 detection/lane loss。

## 3. 自制数据集如何标注

推荐标注目标：

```text
0 = non-free
1 = drivable / free-space
```

标注原则：

```text
只标当前车辆能够通行的地面区域。
人、车、锥桶、路沿、草地、人行道等都标为 non-free。
被车辆遮挡但语义上是路面的区域，不标为 free-space。
```

推荐流程：

```text
1. 从视频抽帧，优先覆盖逆光、雨雪、遮挡、路口、行人穿越等情况。
2. 用 CVAT 或 Label Studio 做 polygon / brush mask 标注。
3. 用 SAM / SAM2 辅助生成初始 mask，再人工修边。
4. 导出 PNG mask，确保与 RGB 图像同尺寸。
5. 划分 train / val，避免同一连续视频片段同时出现在 train 和 val。
```

推荐工具：

```text
CVAT
Label Studio
SAM / SAM2 辅助标注
```

资料：

```text
CVAT automatic annotation:
https://docs.cvat.ai/docs/annotation/auto-annotation/

CVAT SAM2 Tracker:
https://docs.cvat.ai/docs/annotation/auto-annotation/segment-anything-2-tracker/

Label Studio SAM integration:
https://labelstud.io/integrations/computer-vision/segment-anything-model/

Meta SAM2:
https://ai.meta.com/sam2/
```

## 4. YOLOP 官方如何训练

官方入口：

```bash
cd /workspace/src/YOLOP/external/YOLOP
python3 tools/train.py
```

官方训练配置在：

```text
src/YOLOP/external/YOLOP/lib/config/default.py
```

与 drivable branch 相关的配置：

```text
TRAIN.DRIVABLE_ONLY = False
LOSS.DA_SEG_GAIN = 0.2
num_seg_class = 2
```

如果只想训练 drivable branch，官方代码中有：

```text
TRAIN.DRIVABLE_ONLY = True
```

但该模式依然依赖 dataset loader 能正常返回 detection、drivable、lane 三类 target。实际工程上更建议新增项目内训练脚本，显式只训练：

```text
YOLOP image encoder
+ drivable-area segmentation head
```

## 5. 论文 STGRU 如何训练

论文：`Semantic Video Segmentation by Gated Recurrent Flow Propagation`

论文中的 STGRU / GRFP 训练思想：

```text
single-frame semantic CNN 输出 x_t
previous hidden segmentation h_{t-1}
optical flow f_{t-1→t}
warp(h_{t-1}, f_{t-1→t}) = w_t
STGRU(x_t, w_t, flow confidence) = h_t
```

它不是固定 alpha。它训练卷积门控参数，让网络自己决定当前帧预测和历史传播结果各占多少权重。

论文训练数据：

```text
Cityscapes video snippets
CamVid video sequences
```

Cityscapes 训练方式：

```text
每个 labeled frame 位于 30 帧 video snippet 的第 20 帧。
使用 labeled frame 附近的未标注连续帧。
常用 forward model 使用 5 帧训练，T=4。
loss 只施加在有 ground truth 的 labeled frame 上。
未标注帧通过光流和 STGRU 参与时序传播。
```

论文中的优化设置：

```text
STGRU: Adam
beta1 = 0.95
beta2 = 0.99
learning rate = 2e-5

static segmentation network refinement:
SGD + momentum 0.95
learning rate = 2e-11
```

论文中的光流：

```text
训练时预计算 labeled frame 附近帧之间的 forward/backward flow。
论文使用过 FullFlow、DIS、FlowNet / FlowNet2。
```

本项目对齐方式：

```text
static segmentation network: YOLOP drivable head
optical flow: SEA-RAFT
warp: torch.grid_sample 或 cv2.remap
STGRU input channels: binary/probability free-space maps
loss: binary cross entropy / dice / IoU loss
training label: sparse drivable-area ground truth at target frame
```

## 6. 我们能否效仿论文

可以，但要分阶段：

```text
阶段 1:
YOLOP frozen，只输出 current_mask。
SEA-RAFT frozen，只输出 flow。
训练 STGRU，只学习 temporal fusion。

阶段 2:
YOLOP 可选 fine-tune drivable head。
SEA-RAFT 仍 frozen。

阶段 3:
尝试端到端训练 YOLOP + STGRU。
不建议早期训练 SEA-RAFT，因为语义 loss 对 optical flow 的监督很弱。
```

训练集最低要求：

```text
连续 RGB 帧: I_{t-4}, ..., I_t
目标帧 drivable mask GT: M_t
相邻帧 optical flow: 可由 SEA-RAFT 预计算
```

推荐数据来源：

```text
Cityscapes leftImg8bit_sequence + gtFine
BDD100K videos + drivable labels
自采视频 + 稀疏标注 drivable mask
```

自采视频可以用，但必须至少标注部分关键帧的 drivable mask。没有 mask GT 时，只能做推理演示和 temporal consistency 指标，不能真正训练 STGRU。
