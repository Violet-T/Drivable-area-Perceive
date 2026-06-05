# 在线推理流程调整说明

## 项目边界

当前项目主线调整为：

```text
YOLOP 单帧 drivable/free-space perception
+ SEA-RAFT optical flow
+ warp temporal alignment
+ temporal fusion / STGRU 复现
+ OpenCV visualization
+ temporal consistency evaluation
```

不再把 BEV、IPM、OccupancyGrid、RViz2 或 ROS2 作为主线目标。

## 单帧输出规则

YOLOP 官方模型是多任务网络：

```text
object detection
+ drivable area segmentation
+ lane line segmentation
```

本项目只使用：

```text
YOLOP drivable area segmentation head
→ raw free-space mask
```

输出格式：

```text
float32
shape: HxW
value range: [0, 1]
1 = drivable / free-space
0 = non-free
```

注意：

```text
YOLOP 不负责 temporal fusion。
YOLOP 不输出项目需要的检测框。
SEA-RAFT 的输入不是 YOLOP mask。
SEA-RAFT 的输入必须是 Image_{t-1}, Image_t。
YOLOP mask 是 warp / fusion / STGRU 的输入。
```

## 离线文件式流程

当前 YOLOP 离线 baseline：

```text
video or image sequence
→ run_yolop_inference.py
→ raw_masks/*.npy
→ manifest.csv
→ 后续 SEA-RAFT / warp / fusion 脚本读取文件
```

文件只用于：

```text
最终视频导出
最终指标 CSV
实验日志
可选 debug dump
```

## 在线推理目标

在线推理不应通过中间文件在模块之间传递结果，而应该在同一个 Python 进程中维护内存数据流：

```text
VideoSequenceLoader
→ current_rgb
→ YOLOPFreeSpaceWrapper.infer_drivable_mask(current_rgb)
→ current_mask

previous_rgb + current_rgb
→ SEARAFTWrapper.infer_flow(previous_rgb, current_rgb)
→ flow_{t-1→t}

previous_fused_mask + flow_{t-1→t}
→ warp_mask_with_flow(previous_fused_mask, flow)
→ warped_mask

current_mask + warped_mask
→ temporal_fusion(alpha, non_free_threshold)
→ fused_mask
```

后续复现论文时，把最后一步替换为：

```text
current_mask + warped_hidden/mask + flow confidence
→ STGRU / GRFP
→ fused_mask
```

## 当前入口

单帧 YOLOP baseline：

```bash
python3 src/YOLOP/run_yolop_inference.py \
  --input-video /workspace/data/demo/eg.mp4 \
  --output-dir /workspace/output/yolop_eg_baseline \
  --device cuda \
  --overwrite
```

YOLOP + SEA-RAFT temporal baseline：

```bash
python3 src/SEA_RAFT/run_yolop_temporal_fusion.py \
  --input-video /workspace/data/demo/eg.mp4 \
  --output-dir /workspace/output/yolop_eg_temporal \
  --alpha 0.7 \
  --non-free-threshold 0.2 \
  --device cuda \
  --overwrite
```

## 推荐实现顺序

1. 固定 YOLOP drivable mask 输出格式。
2. 固定 SEA-RAFT flow 输出格式。
3. 保留 alpha fusion 作为 temporal baseline。
4. 补齐 temporal metrics，对比 raw YOLOP vs YOLOP+SEA-RAFT fusion。
5. 实现 STGRU 层，用训练得到的门控替换手工 alpha。
6. 使用稀疏标注视频序列训练 STGRU。
7. 对比 raw YOLOP、alpha fusion、STGRU fusion 三组结果。

## 关键设计判断

- YOLOP 只负责单帧 drivable/free-space mask。
- SEA-RAFT 只负责相邻 RGB 帧之间的 optical flow。
- mask warp 和 temporal fusion 是独立模块。
- OpenCV video / metrics 是实验验证输出，不是感知模块输入。
- 不在当前阶段引入 BEV、OccupancyGrid 或规划相关结论。
