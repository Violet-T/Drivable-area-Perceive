# 开发日志

## 2026-06-01

### Modified Module

- Docker Environment
- Project Structure
- PIDNet
- RAFT
- Temporal Fusion
- OccupancyGrid
- RViz2

### Changes

- 建立初始 ROS2 Humble workspace 结构，源码统一放入 `src/`。
- 新增四个 ROS2 Python 节点包：
  - `pidnet_node`：PIDNet 单帧 free-space mask 包装节点占位。
  - `raft_node`：RAFT optical flow 包装节点占位。
  - `temporal_fusion_node`：声明 `alpha=0.7`，记录默认融合方程 `fused = alpha * current_mask + (1 - alpha) * warped_mask`。
  - `occupancy_grid_node`：声明局部 OccupancyGrid 默认配置，`frame_id=base_link`，`forward_range=20m`，`side_range=10m`，`resolution=0.1m/cell`。
- 新增共享工具包 `freespace_utils`，包含：
  - `warp_mask_with_flow(previous_mask, flow)`：使用 `cv2.remap` 对历史 mask 做时序对齐，输入约定为 `previous_mask: HxW float32`，`flow: HxWx2 float32`。
  - `fuse_masks(current_mask, warped_mask, alpha)`：实现默认 mask 融合并裁剪到 `[0, 1]`。
  - `occupancy_grid_size()`：根据范围和分辨率计算 OccupancyGrid 尺寸。
- 新增 `src/launch/initial_pipeline.launch.py`，用于启动四个占位节点，后续作为模块接线基础；支持 `use_rviz:=true` 在显示转发可用时启动 RViz2。
- 新增 `src/rviz/freespace_perception.rviz`，包含 camera image、PIDNet mask、fused mask、OccupancyGrid 的 RViz2 显示项。
- 新增 Docker 环境文件：
  - Base image: `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04`
  - ROS distribution: `ROS2 Humble`
  - 包含 `colcon`、`rviz2`、`cv_bridge`、OpenCV、NumPy、PyTorch/CUDA 等基础依赖。
- 克隆外部模型源码到 `src/external/`：
  - PIDNet: `https://github.com/XuJiacong/PIDNet.git`
    - commit: `4c158cf24ce432f0a8cb43364fae38d93cee0dc3`
  - RAFT: `https://github.com/princeton-vl/RAFT.git`
    - commit: `2888e15a51fa41140771d3f498ed8023cff098d1`
- 在 `src/external/` 添加 `COLCON_IGNORE`，避免 `colcon` 初期构建时误构建外部研究代码或 RAFT CUDA extension。

### Reason

- 当前阶段目标是建立 algorithm-level baseline 和工程骨架，不进行端到端深度模型集成。
- PIDNet、RAFT、Temporal Fusion、OccupancyGrid 保持模块分离，便于后续逐步验证输入/输出定义、ROS2 通信和 RViz 可视化。
- 外部模型源码先以 Git 仓库形式接入，后续只在本项目中编写 wrapper、参数处理、消息转换和日志逻辑，避免手写或改写模型内部。
- Docker 环境用于保证后续 build、launch、PIDNet/RAFT inference/training 都在一致环境中执行。

### Result

- 项目已经具备初始目录结构、Docker 构建入口、ROS2 Python 节点包骨架、launch 文件和 RViz 配置。
- 当前 PIDNet 与 RAFT 节点仍为 placeholder，只声明参数和算法 I/O 约定，尚未加载真实权重或执行推理。
- Temporal Fusion 工具函数已经实现基础 `warp + alpha fusion` 方法，但节点尚未接入 ROS topic 数据流。
- OccupancyGrid 节点目前只声明地图语义和默认参数，尚未实现 BEV/IPM 投影和 grid 发布。

### Known Issues

- 尚未在 Docker 容器内执行 `colcon build` 验证；下一步应先 build 占位节点，再验证 `ros2 launch /workspace/src/launch/initial_pipeline.launch.py`。
- RViz 配置是初始占位，topic 类型和显示参数需要等真实 publisher 接入后再调试。
- PIDNet/RAFT 权重下载、模型初始化、CUDA 推理、消息格式转换尚未完成。
- BEV/IPM 相机标定参数、投影矩阵和 OccupancyGrid 写入规则尚未定义。

## 2026-06-01

### Modified Module

- SEA-RAFT
- Docker Environment
- Temporal Fusion

### Changes

- 将 optical flow 外部实现从原始 RAFT 切换为 SEA-RAFT。
- 移除旧外部源码目录 `src/external/RAFT`。
- 将已有 SEA-RAFT 仓库整理到 `src/external/SEA-RAFT`，符合外部算法仓库统一放入 `src/external/` 的结构要求。
- SEA-RAFT Git 信息：
  - URL: `https://github.com/princeton-vl/SEA-RAFT.git`
  - commit: `9137517ba24e628442aec097d3afe71d03503b75`
- 保留 ROS2 package 名称 `raft` 和 executable `raft_node`，用于兼容已有 pipeline 命名；节点内部语义更新为 SEA-RAFT wrapper。
- `raft_node` 默认 `model_repo` 更新为 `/workspace/src/external/SEA-RAFT`。
- optical flow 默认 topic 从 `/raft/flow` 调整为 `/searaft/flow`，`temporal_fusion_node` 同步订阅该 topic。
- Docker base image 调整为 `nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04`，PyTorch 固定为 `2.2.0` 系列，更贴近 SEA-RAFT README 中的开发环境。
- `requirements.txt` 增加 `h5py`，补齐 SEA-RAFT requirements 中的基础依赖。

### Reason

- SEA-RAFT 是更高效的 RAFT 系光流实现，后续可作为动态 free-space temporal alignment 的 optical flow backbone。
- 保留 `raft_node` 名称可以减少 launch、fusion、后续报告和已有 AGENTS 节点约束的连锁改动，同时实际算法实现切换到 SEA-RAFT。
- 将外部仓库放入 `src/external/` 并由 `COLCON_IGNORE` 排除，可以避免 `colcon build` 误构建研究代码。

### Result

- 当前工程中的 optical flow 目标实现已切换为 SEA-RAFT。
- `raft_node` 仍为 placeholder，只声明 SEA-RAFT 仓库路径、权重路径和 flow 输出定义，尚未加载真实 SEA-RAFT 权重。
- Temporal Fusion 的 flow 输入 topic 已与 SEA-RAFT 默认输出 topic 对齐。

### Known Issues

- 尚未在 Docker 容器内重新 build 和 launch 验证。
- SEA-RAFT 权重下载、`config/eval/*.json` 选择、模型初始化和 ROS2 Image/flow message 转换尚未完成。
- Docker 镜像尚未实际构建验证；如宿主 GPU driver 不支持 CUDA 12.2 runtime，需要回退 base image 或调整 PyTorch CUDA wheel。

## 2026-06-01

### Modified Module

- Docker Environment

### Changes

- 在宿主机执行 Docker 操作，构建项目镜像 `freespace_perception:humble`。
- 启动长期运行容器 `perceive`，用于 VSCode Docker / Dev Containers 插件 attach。
- 容器运行参数包含：
  - `--gpus all`
  - `--net=host`
  - `--ipc=host`
  - `/home/fripr/perceive:/workspace`
  - `/tmp/.X11-unix:/tmp/.X11-unix`
  - `DISPLAY` 和 `QT_X11_NO_MITSHM`
- 在容器内验证 ROS2 Humble、`rclpy`、PyTorch CUDA 和 GPU 可见性。

### Reason

- 后续 `colcon build`、ROS2 launch、PIDNet/SEA-RAFT inference、RViz2 调试都必须在项目 Docker 环境中执行。
- 使用固定容器名 `perceive` 可以便于 VSCode 插件识别并 attach 到一致的运行环境。

### Result

- Docker 镜像 `freespace_perception:humble` 构建成功。
- 容器 `perceive` 已运行，工作目录为 `/workspace`。
- 容器内 ROS 环境验证结果：
  - `ROS_DISTRO=humble`
  - `rclpy` 可导入
- 容器内 PyTorch/GPU 验证结果：
  - `torch 2.2.0+cu121`
  - `cuda_available=True`
  - GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`

### Known Issues

- 当前只完成 Docker 环境运行验证，尚未在容器内执行 `colcon build`。
- RViz2 显示仍依赖宿主机 X11 权限；如后续无法打开窗口，需要执行 `xhost +local:docker` 或按系统显示配置进一步调整。

## 2026-06-01

### Modified Module

- PIDNet
- Docker Environment

### Changes

- 在 Docker 容器 `perceive` 内补齐 PIDNet 评估依赖：
  - `yacs`
  - `gdown`
- 将依赖同步记录到 `requirements.txt`，保证后续镜像重建时环境可复现。
- 按 PIDNet 官方目录约定放置权重：
  - `pretrained_models/cityscapes/PIDNet_S_Cityscapes_test.pt`
  - `pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt`
  - `pretrained_models/imagenet/PIDNet_S_ImageNet.pth.tar`
- 为 Cityscapes 标注目录建立软链接：
  - `data/cityscapes/gtFine -> gtFine_trainvaltest/gtFine`
- 使用 PIDNet-S Cityscapes test 权重运行仓库自带 `samples/` 图片推理 demo。
- 尝试运行官方 Cityscapes val 评估命令：
  - `python3 tools/eval.py --cfg configs/cityscapes/pidnet_small_cityscapes.yaml TEST.MODEL_FILE pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt`

### Reason

- PIDNet 是当前 free-space pipeline 的单帧语义分割 backbone，需要先验证其原仓库推理链路、CUDA 环境、权重加载和基础数据路径。
- sample 推理用于确认模型可执行；Cityscapes val 评估用于确认后续 mIoU 指标验证路径。

### Result

- PIDNet-S sample 推理成功，生成语义分割可视化结果：
  - `samples/outputs/frankfurt_000000_002196_leftImg8bit.png`
  - `samples/outputs/frankfurt_000000_003025_leftImg8bit.png`
- 推理日志显示：
  - `Loaded 453 parameters`
- 官方 val 评估已成功完成模型构建、ImageNet 初始化权重加载和 Cityscapes val 权重加载。
- 正式 mIoU 评估尚未完成，当前停止在 DataLoader 读取第一张 Cityscapes val 图像阶段。

### Known Issues

- 当前工作区缺少 `data/cityscapes/leftImg8bit/` 或 `leftImg8bit_trainvaltest.zip`，导致 `cv2.imread` 返回空图像并触发：
  - `AttributeError: 'NoneType' object has no attribute 'shape'`
- 需要将 Cityscapes leftImg8bit 图像数据放置或解压到：
  - `src/external/PIDNet/data/cityscapes/leftImg8bit/`
- PIDNet 原仓库 `tools/custom.py` 不是交互式可视化界面；当前可视化方式为生成 PNG 文件。

## 2026-06-01

### Modified Module

- PIDNet

### Changes

- 确认 Cityscapes `leftImg8bit` 数据已解压，并通过软链接对齐 PIDNet list 文件期望路径：
  - `data/cityscapes/leftImg8bit -> leftImg8bit_trainvaltest/leftImg8bit`
- 对 PIDNet 原仓库做 NumPy 1.26 兼容补丁，将旧别名 `np.int` 替换为内置 `int`：
  - `tools/eval.py`
  - `tools/train.py`
  - `utils/utils.py`
  - `datasets/base_dataset.py`
- 在 Docker 容器 `perceive` 内运行 PIDNet-S Cityscapes val 官方评估：
  - `python3 tools/eval.py --cfg configs/cityscapes/pidnet_small_cityscapes.yaml TEST.MODEL_FILE pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt`

### Reason

- PIDNet 原仓库代码使用的 `np.int` 在当前 NumPy 1.26 环境中已移除，必须进行兼容修复后才能完成全量评估。
- 完整 Cityscapes val 评估用于验证 PIDNet-S 权重、数据路径、CUDA 推理和指标计算链路是否可复现。

### Result

- Cityscapes val 共 500 张图像评估完成。
- 评估结果：
  - `MeanIU: 0.7876`
  - `Pixel_Acc: 0.9619`
  - `Mean_Acc: 0.8619`
- 结果与 PIDNet README 中 PIDNet-S Cityscapes val 约 `78.8% mIoU` 基本一致。
- 评估日志保存于：
  - `src/external/PIDNet/output/cityscapes/pidnet_small_cityscapes/pidnet_s_cityscapes_val_manual_2026-06-01_rerun.log`

### Known Issues

- PIDNet 原仓库没有直接的视频播放或 GUI 可视化工具；现有 `tools/custom.py` 支持对图片文件夹批量推理并输出分割 PNG。
- 若需要连续视频效果，需要使用视频抽帧、批量推理、再用 `ffmpeg` 或 OpenCV 合成为 MP4；后续可封装为项目内 PIDNet video demo 工具。

## 2026-06-01

### Modified Module

- PIDNet
- Visualization

### Changes

- 新增项目内 Cityscapes test 批量推理与视频化 demo：
  - `src/pidnet/pidnet/cityscapes_video_demo.py`
- 在 `src/pidnet/setup.py` 中加入命令入口：
  - `pidnet_cityscapes_video_demo`
- demo 默认读取 PIDNet 原仓库的 `data/list/cityscapes/test.lst`，加载：
  - `pretrained_models/cityscapes/PIDNet_S_Cityscapes_test.pt`
- 对 Cityscapes test 全量图片执行 PIDNet-S 语义分割推理，并输出：
  - 分割 PNG 帧序列
  - `frame_index.csv`
  - MP4 视频

### Reason

- 需要将 PIDNet 单帧语义分割结果转化为连续可观察的视频形式，便于快速检查模型在 Cityscapes test 图像上的整体视觉效果。
- 该步骤为后续 free-space mask、temporal fusion 和 RViz2 可视化建立基础可视验证手段。

### Result

- 在 Docker 容器 `perceive` 内完成全量推理。
- 模型权重加载正常：
  - `Loaded 453 parameters`
- Cityscapes test 共生成 1525 张分割结果图：
  - `src/external/PIDNet/output/cityscapes/test_video_demo/segmentation_frames/`
- 生成 10 FPS 视频：
  - `src/external/PIDNet/output/cityscapes/test_video_demo/pidnet_pidnet-s_cityscapes_test_10fps.mp4`
- 视频参数：
  - 分辨率：2048x1024
  - 帧数：1525
  - 时长：152.5s
  - 大小：约 40MB

### Known Issues

- Cityscapes `test.lst` 是数据集列表顺序，不等价于真实连续视频序列；合成视频会在不同城市或片段之间跳变。
- 当前输出是纯语义分割彩色图，不包含原图叠加显示；后续可增加 overlay 或 side-by-side 模式。

## 2026-06-01

### Modified Module

- PIDNet
- Free-space Mask
- Visualization

### Changes

- 在 `src/pidnet/pidnet/cityscapes_video_demo.py` 中新增 `free-space` 渲染模式。
- 将 Cityscapes `road` 类定义为默认可通行区域：
  - `road -> 1.0`
  - 其他类别 -> `0.0`
- 新增保守障碍物排除后处理：
  - 默认障碍类别：`person,rider,car,truck,bus,train,motorcycle,bicycle`
  - `block_dilate_px`：对障碍物 mask 做膨胀，扩大不可通行边界
  - `block_shadow_px`：将障碍物 mask 向图像下方扩展，覆盖车底或障碍物下方容易被误判为 road 的区域
- `free-space` 模式输出绿色可通行区域、黑色不可通行区域的视频帧。

### Reason

- 下游 temporal fusion、BEV/IPM 和 OccupancyGrid 不应直接使用 19 类语义图，而应使用项目定义的二值 free-space mask。
- 仅将 `road` 作为 free-space 可以避免 `sidewalk`、`terrain` 等区域被误认为可通行。
- 车辆底部可见道路虽然语义上可能是 `road`，但对规划任务并非安全可通行区域，需要用障碍物类别的保守膨胀和向下遮挡进行排除。

### Result

- 小样本验证命令已在 Docker 容器 `perceive` 内运行通过：
  - `python3 src/pidnet/pidnet/cityscapes_video_demo.py --render-mode free-space --max-frames 3 --fps 10 --overwrite`
- 权重加载正常：
  - `Loaded 453 parameters`
- 生成测试输出：
  - `src/external/PIDNet/output/cityscapes/test_video_demo/free_space_frames/`
  - `src/external/PIDNet/output/cityscapes/test_video_demo/pidnet_pidnet-s_cityscapes_test_free_space_10fps.mp4`

### Known Issues

- 当前车底排除仍是图像空间启发式后处理，不等价于真实 3D 占据推理。
- 后续进入 BEV/IPM 与 OccupancyGrid 阶段时，需要结合相机标定、地面假设、障碍物 footprint inflation 或深度信息进一步约束可通行区域。

## 2026-06-01

### Modified Module

- PIDNet
- Free-space Mask

### Changes

- 按实验基线需求，移除 `cityscapes_video_demo.py` 中针对障碍物的图像空间扩张后处理：
  - 移除 obstacle dilation
  - 移除 obstacle shadow / 向下扩展遮挡
  - 移除 `block_classes`、`block_dilate_px`、`block_shadow_px` 相关参数
- `free-space` 模式回到 road-only 定义：
  - `mask = 1.0`：PIDNet 输出类别属于 `free_space_classes`，默认仅 `road`
  - `mask = 0.0`：其他所有类别

### Reason

- 需要保留一个不做障碍物修正的基础实验版本，用于观察 PIDNet 原始 road mask 直接进入 SEA-RAFT、temporal fusion、BEV/IPM 和 OccupancyGrid 后的影响。
- 该版本便于对比后续保守障碍物排除、BEV footprint inflation 或 3D 约束方法带来的差异。

### Result

- 小样本验证命令已在 Docker 容器 `perceive` 内运行通过：
  - `python3 src/pidnet/pidnet/cityscapes_video_demo.py --render-mode free-space --max-frames 3 --fps 10 --overwrite`
- 当前输出中，可通行区域完全由 `road` 类决定，不再主动扣除车辆或行人下方区域。

### Known Issues

- 该 road-only baseline 会保留车底或障碍物附近被 PIDNet 判为 `road` 的像素，后续 OccupancyGrid 可能因此产生过于乐观的 free cell。

## 2026-06-01

### Modified Module

- PIDNet
- Dataset / Batch Inference
- Project Workflow

### Changes

- 根据新版 `AGENTS.md` 调整项目主线：
  - 聚焦 PIDNet 单帧 free-space、SEA-RAFT optical flow、warp、temporal fusion、OpenCV visualization 和 temporal consistency evaluation。
  - BEV、IPM、OccupancyGrid、RViz2 和强制 ROS2 集成不再作为当前阶段主线。
- 新增 PIDNet 推理封装：
  - `src/pidnet/pidnet/pidnet_wrapper.py`
- 新增批量 PIDNet free-space 输出脚本：
  - `src/scripts/run_pidnet_inference.py`
- 批量输出从“视频可视化结果”升级为真实数据接口：
  - `raw_masks/*.npy`：`float32 HxW`，范围 `[0,1]`
  - `manifest.csv`：单帧 RGB 与 raw mask 对齐
  - `sea_raft_pairs.csv`：相邻 RGB 图像对与对应 mask 路径
  - `previews/*.png`：可选 debug 可视化
- 分类规则写入 PIDNet batch inference：
  - 19 类 Cityscapes PIDNet 输出
  - 默认提取 `road`
  - 转换为 binary free-space mask

### Reason

- 之前的视频 demo 只改变了可视化颜色，不能作为后续 SEA-RAFT、warp 和 fusion 的稳定数据接口。
- SEA-RAFT 的输入应为相邻 RGB 帧，而不是 PIDNet mask；因此需要同时输出 RGB pair manifest 和 mask manifest，使后续模块可以在同一帧索引体系下工作。
- 新版项目目标是时序一致性增强，模块间应优先传递内存数组或明确的 `.npy/.csv` 实验数据，而不是依赖 PNG 视频帧作为算法输入。

### Result

- Docker 容器 `perceive` 内完成 3 帧 smoke test：
  - `python3 src/scripts/run_pidnet_inference.py --output-dir /workspace/outputs/pidnet_smoke --max-frames 3 --save-preview --overwrite`
- 验证结果：
  - PIDNet 权重加载：`Loaded 453 PIDNet parameters`
  - `raw_mask` shape：`(1024, 2048)`
  - `raw_mask` dtype：`float32`
  - `raw_mask` value：`0.0 / 1.0`
  - `sea_raft_pairs.csv` 正确生成 2 个相邻帧 pair
- 新增在线推理流程说明：
  - `docs/online_inference_plan.md`
- 更新 README，使项目说明与新版 `AGENTS.md` 保持一致。

### Known Issues

- 当前 PIDNet batch 输出仍是离线 `.npy/.csv` 数据接口；真正在线版需要继续实现 `SEA-RAFTWrapper`、`run_online_temporal_demo.py` 和内存中的 mask/flow/fusion pipeline。
- 当前 `test.lst` 仍不是严格连续视频序列；时序实验应切换到 Cityscapes sequence 或真实视频抽帧。

## 2026-06-01

### Modified Module

- SEA-RAFT
- Warp
- Temporal Fusion
- Visualization

### Changes

- 新增 SEA-RAFT 推理封装：
  - `src/sea_raft/sea_raft_wrapper.py`
  - 输出格式：`float32 HxWx2`
  - `flow[...,0] = dx`
  - `flow[...,1] = dy`
- 新增 temporal 模块：
  - `src/temporal/warp.py`
  - `src/temporal/fusion.py`
- 新增 OpenCV 可视化辅助：
  - `src/visualization/overlay.py`
- 新增短序列在线 smoke test 脚本：
  - `src/scripts/run_online_temporal_smoke.py`
- 流程为内存串联：
  - 连续 RGB 帧
  - PIDNet 输出 raw free-space mask
  - SEA-RAFT 对相邻 RGB 帧估计 optical flow
  - 使用 flow warp 上一帧 fused mask
  - 使用 `alpha=0.7` 融合当前 raw mask 与 warped historical mask

### Reason

- 需要验证核心研究链路是否能从单帧 PIDNet 推理推进到有时序层的 free-space mask。
- SEA-RAFT 不接收 mask，而是接收相邻 RGB 图像；其输出 flow 用于对齐上一帧 mask。
- temporal fusion 的目标是让输出从单帧二值 mask 变为具有历史约束的连续概率 mask，用于后续稳定性可视化和指标评估。

### Result

- 在 Docker 容器 `perceive` 内完成 8 帧 Leverkusen 连续样例 smoke test：
  - `python3 src/scripts/run_online_temporal_smoke.py --max-frames 8 --resize-width 512 --output-dir /workspace/outputs/temporal_smoke --fps 5`
- 模型加载：
  - PIDNet：`Loaded 453 PIDNet parameters`
  - SEA-RAFT：`MemorySlices/Tartan-C-T-TSKH-spring540x960-S`
- 输出：
  - `outputs/temporal_smoke/raw_masks/`
  - `outputs/temporal_smoke/flows/`
  - `outputs/temporal_smoke/warped_masks/`
  - `outputs/temporal_smoke/fused_masks/`
  - `outputs/temporal_smoke/videos/raw_mask.mp4`
  - `outputs/temporal_smoke/videos/fused_mask.mp4`
  - `outputs/temporal_smoke/videos/temporal_panel.mp4`
- 验证结果：
  - raw mask：`float32 (256,512)`，取值 `0/1`
  - flow：`float32 (256,512,2)`
  - fused mask：`float32 (256,512)`，取值 `[0,1]`，包含连续概率值
  - temporal panel 视频：8 帧，5 FPS，分辨率 `1024x512`

### Known Issues

- 当前为 smoke test，分辨率下采样到宽度 512；完整实验需要评估更高分辨率和更长连续序列。
- 使用 PIDNet samples 中的 Leverkusen 样例作为短序列验证；正式时序实验仍建议使用 Cityscapes sequence 或真实视频抽帧。
- 当前只验证了链路可运行，尚未计算 Temporal IoU、Frame Difference、Boundary Jitter 和 Flicker Rate。

## 2026-06-01

### Modified Module

- SEA-RAFT
- Dataset / Sequence Selection
- Experiment Interpretation

### Changes

- 明确上一轮 Leverkusen / Cityscapes test smoke test 的局限：
  - `leverkusen_000000_000019`
  - `leverkusen_000001_000019`
  - `berlin_000000_000019`
  - `berlin_000001_000019`
- 这些文件名中的中间编号不是相邻时间帧编号，而更接近不同 snippet / scene id；最后的 `000019` 才是固定采样帧。
- 因此将 PIDNet samples 或 Cityscapes `test.lst` 直接按列表顺序作为连续视频序列是不正确的。

### Reason

- SEA-RAFT 估计的是相邻时间帧之间的 optical flow。如果输入不是同一场景连续帧，flow 会失真，warp 后的 historical mask 会错误对齐，fused mask 质量会明显变差。
- SEA-RAFT 不需要支持 Cityscapes mask；它只需要 RGB 图像对。mask 由 PIDNet 输出，之后通过 SEA-RAFT 的 flow 做 temporal alignment。

### Result

- 当前 smoke test 只能证明代码链路可运行，不能作为时序融合效果的有效实验结果。
- 后续必须使用真实连续帧数据：
  - Cityscapes `leftImg8bit_sequence`
  - Cityscapes `demoVideo`
  - 真实道路视频抽帧
  - 或具有 optical flow ground truth 的 KITTI / Spring / Sintel 等数据集

### Known Issues

- 当前本地未检测到 `leftImg8bit_sequence` 目录。
- 若要训练或微调 SEA-RAFT，需要 optical flow ground truth 或自监督/pseudo-label 方案；Cityscapes 标准语义分割标签不能直接作为 SEA-RAFT 的训练标签。

## 2026-06-02

### Modified Module

- PIDNet
- Dataset / Video Frame Extraction

### Changes

- 本轮调试不调用 SEA-RAFT、warp 或 temporal fusion，仅验证 PIDNet 单帧 free-space 推理。
- 将工作区根目录视频 `eg.mp4` 按 15 FPS 抽帧：
  - 输入视频：`/workspace/eg.mp4`
  - 输出帧目录：`outputs/eg_pidnet_15fps/frames/`
  - 帧列表：`outputs/eg_pidnet_15fps/frame_list.txt`
- 使用 PIDNet-S Cityscapes checkpoint 对抽帧图像批量推理：
  - `free_space_classes = road`
  - `mask_mode = binary`
- 输出：
  - `outputs/eg_pidnet_15fps/raw_masks/`
  - `outputs/eg_pidnet_15fps/previews/`
  - `outputs/eg_pidnet_15fps/manifest.csv`
  - `outputs/eg_pidnet_15fps/eg_pidnet_mask_15fps.mp4`

### Reason

- 先独立验证 PIDNet 对自有视频抽帧的 road/free-space 单帧输出质量，避免时序层影响问题定位。
- 该结果可作为后续 SEA-RAFT temporal fusion 的 raw PIDNet baseline。

### Result

- `eg.mp4` 信息：
  - 分辨率：`1280x720`
  - 时长：约 `16.323s`
  - 原视频主流帧率：`30 FPS`
- 抽帧结果：
  - 15 FPS
  - 244 张 PNG
- PIDNet 推理结果：
  - 权重加载：`Loaded 453 PIDNet parameters`
  - `raw_mask` shape：`(720,1280)`
  - `raw_mask` dtype：`float32`
  - `raw_mask` value：`0.0 / 1.0`
  - 共生成 244 个 raw mask 与 244 个 preview mask
- mask 预览视频：
  - `outputs/eg_pidnet_15fps/eg_pidnet_mask_15fps.mp4`
  - 15 FPS
  - 244 帧
  - 分辨率：`1280x720`

### Known Issues

- 当前输出仅为单帧 PIDNet binary mask，未进行 temporal smoothing。
- 由于输入视频不一定与 Cityscapes 域完全一致，PIDNet 输出质量需要通过预览视频人工检查；必要时后续再考虑 fine-tune PIDNet 或调整 free-space 定义。

## 2026-06-02

### Modified Module

- SEA-RAFT
- Warp
- Temporal Fusion
- Visualization

### Changes

- 新增从已有 PIDNet batch 输出进入时序层的脚本：
  - `src/scripts/run_temporal_fusion_from_pidnet.py`
- 输入：
  - `outputs/eg_pidnet_15fps/manifest.csv`
  - `outputs/eg_pidnet_15fps/frames/`
  - `outputs/eg_pidnet_15fps/raw_masks/`
- 流程：
  - 读取当前帧 RGB 与 PIDNet raw mask
  - 使用 SEA-RAFT 对相邻 RGB 帧估计 `Flow_{t-1→t}`
  - 使用 flow warp 上一帧 fused mask
  - 使用 `alpha=0.7` 和 `non_free_threshold=0.2` 融合当前 raw mask 与 warped mask
- 输出：
  - `outputs/eg_temporal_15fps/flows/`
  - `outputs/eg_temporal_15fps/warped_masks/`
  - `outputs/eg_temporal_15fps/fused_masks/`
  - `outputs/eg_temporal_15fps/videos/raw_mask.mp4`
  - `outputs/eg_temporal_15fps/videos/warped_mask.mp4`
  - `outputs/eg_temporal_15fps/videos/fused_mask.mp4`
  - `outputs/eg_temporal_15fps/videos/temporal_panel.mp4`

### Reason

- 用户要求将刚刚得到的 PIDNet 输出放入时序层进行融合，因此本步骤不重新运行 PIDNet，而是复用已有 raw mask。
- 该流程用于验证自有视频 `eg.mp4` 上的 PIDNet raw mask 是否能通过 SEA-RAFT flow alignment 和 temporal fusion 获得时序层输出。

### Result

- 在 Docker 容器 `perceive` 内完成 244 帧时序融合：
  - `python3 src/scripts/run_temporal_fusion_from_pidnet.py --pidnet-manifest /workspace/outputs/eg_pidnet_15fps/manifest.csv --output-dir /workspace/outputs/eg_temporal_15fps --fps 15 --alpha 0.7 --non-free-threshold 0.2`
- SEA-RAFT 模型：
  - `MemorySlices/Tartan-C-T-TSKH-spring540x960-S`
- 输出数据验证：
  - `flow` shape：`(720,1280,2)`，dtype：`float32`
  - `warped_mask` shape：`(720,1280)`，dtype：`float32`
  - `fused_mask` shape：`(720,1280)`，dtype：`float32`
  - 244 个 flow、244 个 warped mask、244 个 fused mask
- 输出视频：
  - raw / warped / fused：`1280x720`，15 FPS，244 帧
  - temporal panel：`2560x1440`，15 FPS，244 帧
- raw 与 fused 的平均绝对差异：
  - 平均：约 `0.00699`
  - 最小：约 `0.00027`
  - 最大：约 `0.12036`

### Known Issues

- 当前仅完成可视化和基础数据验证，尚未计算正式 Temporal IoU、Frame Difference、Boundary Jitter、Flicker Rate。
- 由于 `non_free_threshold=0.2` 会阻止历史 free-space 覆盖当前 non-free 区域，融合结果整体偏保守，raw/fused 差异不会特别大。

## 2026-06-02

### Modified Module

- Dataset Download
- PIDNet
- Development Workflow

### Changes

- 撤销上一轮 PIDNet mask 后处理实验：
  - 删除 `src/pidnet/pidnet/mask_postprocess.py`
  - 删除 `src/scripts/postprocess_pidnet_masks.py`
  - 删除 `outputs/eg_pidnet_15fps_postprocessed/`
  - 删除 `outputs/eg_temporal_15fps_postprocessed/`
- 保留原始 PIDNet batch 输出和原始 SEA-RAFT temporal fusion 输出。
- 新增 Cityscapes sequence 子集下载脚本：
  - `src/scripts/download_cityscapes_sequence_subset.py`
- 脚本功能：
  - 使用 Cityscapes 官方账号登录。
  - 使用 HTTP Range 读取 ZIP EOCD 和中央目录。
  - 根据 `scene` / `split` / `city` / `seq` 筛选一两个连续场景。
  - 只下载目标 PNG 文件对应的 ZIP 压缩数据块并解压落盘。
  - 生成 `cityscapes_sequence_subset_manifest.csv`。

### Reason

- 后处理会引入人工图像空间先验，可能掩盖 PIDNet 原始模型在复杂道路纹理、反光和遮挡情况下的真实鲁棒性问题。
- 当前研究目标应优先通过更合适的连续 Cityscapes sequence 数据验证 PIDNet + SEA-RAFT 的时序一致性，而不是直接修补 PIDNet 输出。
- Cityscapes `leftImg8bit_sequence_trainvaltest.zip` 体积很大，完整下载成本高，因此需要支持只获取少量连续场景。

### Result

- 已确认后处理相关代码和 postprocessed 输出目录被移除。
- 新增下载脚本通过 Python 语法检查：
  - `PYTHONPYCACHEPREFIX=/tmp/perceive_pycache python3 -m py_compile src/scripts/download_cityscapes_sequence_subset.py`
- 该脚本尚未实际下载数据，因为需要用户提供 Cityscapes 合法账号和目标场景。

### Known Issues

- 该脚本依赖 Cityscapes 官方下载服务支持 HTTP Range；如果服务器或登录会话不返回 HTTP 206，则无法只下载 ZIP 内部子文件。
- `packageID=14` 当前作为 `leftImg8bit_sequence_trainvaltest.zip` 的默认值；若官网下载列表更新，应通过 `--package-url` 显式传入最新链接。
- 该脚本只下载图像 sequence，不下载语义标签；如果后续需要监督 fine-tune 或 accuracy 指标，需要另外下载 `gtFine_trainvaltest.zip`。

## 2026-06-02

### Modified Module

- Dataset Download

### Changes

- 更新 `src/scripts/download_cityscapes_sequence_subset.py`。
- 在 `--list-scenes` 输出中增加每个场景的压缩体积和解压后 PNG 体积估计。
- 在选择场景后输出总文件数、场景数、帧范围和体积统计。
- 新增 `--max-download-gib` 参数，默认限制为 `10.0 GiB`。
- `--dry-run` 模式下先显示选择结果和体积，不实际下载。

### Reason

- 用户希望只下载 10 GiB 以下的 1 至 2 个 Cityscapes sequence 场景。
- 需要在正式下载前确认选择范围，避免误选过多场景导致下载完整大包或超过磁盘预算。

### Result

- 脚本通过语法检查：
  - `PYTHONPYCACHEPREFIX=/tmp/perceive_pycache python3 -m py_compile src/scripts/download_cityscapes_sequence_subset.py`
- 当前默认会在选中数据超过 `10 GiB` 时主动中止。

### Known Issues

- 实际可下载性仍依赖 Cityscapes 官方登录会话和服务器是否支持 HTTP Range。

## 2026-06-02

### Modified Module

- Dataset Download
- Dataset Validation
- Development Workflow

### Changes

- 使用 Cityscapes 官方账号登录下载 `leftImg8bit_sequence_trainvaltest.zip` 中的子场景。
- 根据 `--list-scenes` 输出选择：
  - `train/bochum/bochum_000000`
- 更新 `src/scripts/download_cityscapes_sequence_subset.py`：
  - 增加 Range 请求重试参数。
  - 增加中央目录分块读取，降低官方服务器超时影响。
  - 增加 `--batch-range` 批量下载模式，将相邻 ZIP entry 合并成较大的 Range 请求。
- 新增 `.gitignore`：
  - 忽略 `.cityscapes_cookies.txt`
  - 忽略 `datasets/`
- 生成连续片段索引：
  - `datasets/cityscapes/bochum_000000_clips.csv`

### Reason

- 用户希望只下载 10 GiB 以下的 1 至 2 个 Cityscapes sequence 场景。
- `train/bochum/bochum_000000` 体积约 `6.686 GiB`，低于 10 GiB，且包含多个连续 30 帧片段，适合用于 PIDNet + SEA-RAFT temporal consistency 实验。
- 初始逐文件 Range 下载约 `4 s/file`，完整 2880 张预计约 3 小时，因此需要批量 Range 优化。

### Result

- 下载路径：
  - `datasets/cityscapes/leftImg8bit_sequence/train/bochum/`
- Manifest：
  - `datasets/cityscapes/cityscapes_sequence_subset_manifest.csv`
- 连续片段索引：
  - `datasets/cityscapes/bochum_000000_clips.csv`
- 下载结果：
  - `2880` 张 PNG
  - 落盘约 `6.7G`
  - 单张图像尺寸：`2048x1024`
  - 图像模式：`RGB`
- 连续性分析：
  - 共 `96` 个连续片段
  - 每个片段 `30` 帧
  - 第一个片段：`000294-000323`
  - 最后一个片段：`038131-038160`
- 批量下载结果：
  - `2880` 个文件合并为 `59` 个 batch
  - 跳过已存在文件 `11` 个
  - 下载与解压总耗时约 `994.6 s`

### Known Issues

- `bochum_000000` 不是一条无断点的 2880 帧长视频，而是 96 个连续 30 帧片段；后续时序实验必须按 `bochum_000000_clips.csv` 分片运行，不能跨片段连接。
- 当前只下载了 leftImg8bit sequence 图像，没有下载 gtFine 标签。

## 2026-06-02

### Modified Module

- Visualization
- PIDNet
- SEA-RAFT
- Warp
- Metrics
- Development Workflow

### Changes

- 中断旧的 `bochum_000000_clip_panels` 批处理后，按用户新要求从 clip 77 继续，不从头重跑。
- 更新 `src/scripts/run_cityscapes_sequence_clip_panels.py` 的四宫格布局：
  - 左上：当前帧图像 + 当前 PIDNet road/free-space mask
  - 左下：当前 PIDNet mask
  - 右上：当前帧图像 + SEA-RAFT warp 后的历史 mask
  - 右下：warp 后的历史 mask
- 新增内存态干扰模拟参数：
  - `--weather-mode none|glare|fog|rain|snow|mixed`
  - `--weather-strength`
- 干扰只作用于本次推理输入和可视化，不写回、不修改原始 Cityscapes sequence 数据。
- 每个 clip 输出目录新增 `performance_matrix.csv`，记录每帧：
  - PIDNet 推理耗时
  - SEA-RAFT 推理耗时
  - warp 耗时
  - panel 写入耗时
  - Temporal IoU(raw vs warped)
  - Frame Difference(raw vs warped)
  - mask 均值统计
- 每个 clip 只保留合成视频和性能矩阵，不单独输出 mask、flow 或原图像文件。

### Reason

- 旧四宫格展示 fused mask，不能直接回答“SEA-RAFT warp 对历史 mask 对齐是否有效”的问题。
- 在简单交通场景中，单帧 PIDNet 已较稳定，SEA-RAFT 的可见增益不明显；加入强光、雾、雨、雪等受控干扰，用于观察 motion-compensated historical mask 在短时扰动下的对齐效果。
- 每个场景需要独立性能矩阵，便于后续比较不同 clip、不同干扰模式和不同强度下的速度与稳定性指标。

### Result

- 新输出目录：
  - `outputs/bochum_000000_weather_warp_panels/`
- 从 clip 77 继续处理到 clip 95：
  - 生成 `19` 个 `temporal_panel_h264.mp4`
  - 生成 `19` 个 clip 级 `performance_matrix.csv`
  - 生成 `summary.csv`
- 视频格式检查：
  - H.264 / `avc1`
  - `yuv420p`
  - `2048x1024`
  - `17 FPS`
  - 每个 clip `30` 帧
- 示例 clip 77 指标：
  - `mean_frame_total_time_s ≈ 0.261`
  - `total_clip_compute_time_s ≈ 11.8 s`
  - `mean_temporal_iou_raw_vs_warped ≈ 0.977`
  - `mean_frame_difference_raw_vs_warped ≈ 0.0209`

### Known Issues

- `outputs/bochum_000000_clip_panels/` 中 clip 0-76 是旧布局；本次按要求没有从头重跑。若最终报告需要全量统一四宫格，需要之后单独重跑 clip 0-76。
- 当前速度仍是研究验证级别，全分辨率 `2048x1024` 下每个 30 帧 clip 约 10-13 秒，无法满足实时工程要求。
- 主要瓶颈来自 PIDNet 与 SEA-RAFT 串行全分辨率推理，后续需要加入 profiler、降低输入分辨率、FP16/AMP、TensorRT/ONNX、异步解码/推理/编码，以及降低 optical flow 调用频率。

## 2026-06-02

### Modified Module

- SEA-RAFT
- Visualization
- Metrics

### Changes

- 使用本地官方 spring 训练权重加载 SEA-RAFT：
  - checkpoint：`src/external/SEA-RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth`
  - config：`src/external/SEA-RAFT/config/eval/spring-M.json`
- 在 `src/scripts/run_cityscapes_sequence_clip_panels.py` 中增加 `sudden_glare` 干扰模式：
  - 默认从 clip 内第 10 帧触发
  - 默认持续 8 帧
  - 只在内存输入中添加强光，不修改 Cityscapes 原始图像
- 对 `bochum_000000` 的 clip 80 运行 PIDNet + SEA-RAFT warp 可视化实验。

### Reason

- 用户希望验证官方 spring 训练参数加载后，在突然强光扰动下 SEA-RAFT warp 后历史 mask 的输出效果。
- 突发强光比整段持续强光更适合观察短时扰动对 PIDNet mask 和 optical-flow alignment 的影响。

### Result

- 输出目录：
  - `outputs/bochum_000000_clip080_springM_sudden_glare/`
- 输出视频：
  - `clip_080_031458_031487/temporal_panel_h264.mp4`
- 输出性能矩阵：
  - `clip_080_031458_031487/performance_matrix.csv`
- 视频格式：
  - H.264 / `avc1`
  - `yuv420p`
  - `2048x1024`
  - `17 FPS`
  - `30` 帧
- 强光设置：
  - `weather_mode_request = sudden_glare`
  - `weather_strength = 0.9`
  - `weather_trigger_frame = 10`
  - `weather_duration = 8`
- 总体指标：
  - `mean_temporal_iou_raw_vs_warped ≈ 0.966`
  - `mean_frame_difference_raw_vs_warped ≈ 0.0204`
  - `mean_frame_total_time_s ≈ 0.324`
  - `total_clip_compute_time_s ≈ 12.55 s`
- 分段指标：
  - 强光前 frame 1-9：Temporal IoU ≈ `0.976`，Frame Difference ≈ `0.0150`
  - 强光中 frame 10-17：Temporal IoU ≈ `0.967`，Frame Difference ≈ `0.0219`
  - 强光后 frame 18-29：Temporal IoU ≈ `0.958`，Frame Difference ≈ `0.0235`

### Known Issues

- 当前指标只比较当前 PIDNet mask 与 warp 后历史 mask，没有引入 ground truth，因此不能代表语义精度，只能反映时序对齐一致性。
- 强光扰动下 Temporal IoU 略有下降，说明光照突变会影响当前 mask 与历史 warp mask 的一致性；需要进一步测试 temporal fusion 是否能在不覆盖当前障碍物的前提下降低这种扰动。

## 2026-06-02

### Modified Module

- PIDNet
- SEA-RAFT
- Temporal Fusion
- Visualization
- Metrics

### Changes

- 将时序实验工作尺寸固定为 `540x960`：
  - 输入图像先缩放到 `960x540`
  - PIDNet 推理前临时 padding 到 `960x544`
  - PIDNet 输出 mask 裁回并保持为 `540x960`
  - SEA-RAFT flow、warp mask、fused mask 均在 `540x960` 尺寸下运行
- 新增 3 类特殊环境模拟：
  - `water_drop_slide`：模拟水珠从摄像头上方滑落至下方
  - `short_glare`：随机 1-5 帧突发强光
  - `snow_occlusion`：随机 1-3 帧大雪小范围遮挡摄像头
- 新增 `--suite-special-weather` 批处理入口：
  - 每个环境随机选择 5 个不重复 clip
  - 固定 seed，保证实验可复现
- 修改四宫格输出：
  - 左上：当前帧图像 + 当前 PIDNet mask
  - 左下：当前 PIDNet mask
  - 右上：fused mask
  - 右下：当前 mask 与 fused mask 的三色差异图
- 三色差异图规则：
  - 绿色：当前 mask 与 fused mask 交集
  - 蓝色：仅当前 mask 占据
  - 红色：仅 fused mask 占据
- temporal fusion 改为：
  - warp 上一帧 fused mask
  - 与当前 PIDNet mask 按 `alpha=0.7` 融合
  - 保留 `non_free_threshold=0.2` 的动态障碍物安全规则

### Reason

- 用户要求 PIDNet 输出 mask 压缩到 SEA-RAFT spring 权重对应的 `540x960` 格式。
- 需要在更具有扰动性的特殊环境下验证 PIDNet + SEA-RAFT + temporal fusion 的稳定性，而不是只在简单正常交通场景中观察。
- 新四宫格更直接展示当前单帧结果、融合结果以及二者差异区域。

### Result

- 输出目录：
  - `outputs/bochum_000000_special_env_suite_540x960_springM/`
- 使用 SEA-RAFT spring-M 权重：
  - `src/external/SEA-RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth`
  - `src/external/SEA-RAFT/config/eval/spring-M.json`
- 随机抽样结果：
  - `water_drop_slide`：clip `5, 20, 44, 66, 95`
  - `short_glare`：clip `4, 11, 34, 39, 46`
  - `snow_occlusion`：clip `36, 43, 58, 82, 85`
- 输出文件：
  - `15` 个 `temporal_panel_h264.mp4`
  - `15` 个 `performance_matrix.csv`
  - `summary.csv`
  - `environment_metrics_summary.csv`
- 视频格式：
  - H.264 / `avc1`
  - `yuv420p`
  - `1920x1080`
  - 每个宫格 `960x540`
  - `17 FPS`
  - 每个 clip `30` 帧
- 环境级均值：
  - `water_drop_slide`：
    - raw vs warped Temporal IoU ≈ `0.965`
    - fused vs warped Temporal IoU ≈ `0.975`
    - raw vs fused Frame Difference ≈ `0.00352`
  - `short_glare`：
    - raw vs warped Temporal IoU ≈ `0.952`
    - fused vs warped Temporal IoU ≈ `0.964`
    - raw vs fused Frame Difference ≈ `0.00429`
  - `snow_occlusion`：
    - raw vs warped Temporal IoU ≈ `0.896`
    - fused vs warped Temporal IoU ≈ `0.905`
    - raw vs fused Frame Difference ≈ `0.00768`
- suite 总均值：
  - raw vs warped Temporal IoU ≈ `0.938`
  - fused vs warped Temporal IoU ≈ `0.948`
  - raw vs fused Frame Difference ≈ `0.00516`
  - 平均单帧处理时间 ≈ `0.0836 s`
  - 平均每个 30 帧 clip 处理时间 ≈ `5.24 s`

### Known Issues

- `540x960` 下速度明显提升，但仍未达到实时工程要求；需要继续做 FP16、TensorRT/ONNX、异步流水线和模型频率调度。
- 当前特殊环境是算法模拟，不等价于真实传感器退化；后续若有真实雨雪/眩光视频，需要重复同样评估。
- 当前指标仍以时序一致性为主，没有 Cityscapes gtFine sequence 对应标签时，不能评价 free-space 语义精度。

## 2026-06-02

### Modified Module

- Visualization
- Synthetic Weather
- Metrics

### Changes

- 按用户反馈重新调整 3 类特殊环境：
  - `short_glare`：提高强光亮度，扩大椭圆光斑范围。
  - `snow_occlusion`：取消大范围整屏遮挡，改为 1-3 帧小块雪团遮挡，并允许连续帧中位置漂移。
  - `water_drop_slide`：同时生成 2-8 个水滴，降低透明度，提高折射和拖尾效果，并加快下落速度。
- 删除 `outputs/bochum_000000_special_env_suite_540x960_springM/` 下旧的 15 个 clip 文件夹。
- 使用同一路径重新生成 3 个环境 × 5 个 clip 的视频和指标。
- 重新生成：
  - `summary.csv`
  - `environment_metrics_summary.csv`

### Reason

- 上一版强光干扰不够明显。
- 上一版雪遮挡效果过大，接近整屏退化，不符合“小范围遮住摄像头”的实验设定。
- 上一版水滴过透明、数量过少、下落速度偏慢，不利于观察局部遮挡和光流对齐对 mask 稳定性的影响。

### Result

- 正式输出目录保持不变：
  - `outputs/bochum_000000_special_env_suite_540x960_springM/`
- 输出数量：
  - `15` 个 `temporal_panel_h264.mp4`
  - `15` 个 clip 级 `performance_matrix.csv`
  - `summary.csv`
  - `environment_metrics_summary.csv`
- 视频格式：
  - H.264 / `avc1`
  - `yuv420p`
  - `1920x1080`
  - 每个宫格 `960x540`
  - 每个 clip `30` 帧
- 更新后环境级均值：
  - `water_drop_slide`：
    - raw vs warped Temporal IoU ≈ `0.955`
    - fused vs warped Temporal IoU ≈ `0.967`
    - raw vs fused Frame Difference ≈ `0.00440`
  - `short_glare`：
    - raw vs warped Temporal IoU ≈ `0.945`
    - fused vs warped Temporal IoU ≈ `0.957`
    - raw vs fused Frame Difference ≈ `0.00476`
  - `snow_occlusion`：
    - raw vs warped Temporal IoU ≈ `0.891`
    - fused vs warped Temporal IoU ≈ `0.901`
    - raw vs fused Frame Difference ≈ `0.00753`
- suite 总均值：
  - raw vs warped Temporal IoU ≈ `0.930`
  - fused vs warped Temporal IoU ≈ `0.942`
  - raw vs fused Frame Difference ≈ `0.00556`

### Known Issues

- 水滴模拟增强后计算开销上升，`water_drop_slide` 平均每个 30 帧 clip 约 `7.22 s`。
- 当前 snow occlusion 是局部遮挡，但仍是合成退化；真实雪片贴附镜头时的形态、透明度和运动模式需要真实视频进一步验证。

## 2026-06-02

### Modified Module

- Temporal Fusion
- Warp
- Metrics
- Visualization Workflow

### Changes

- 新增 `src/temporal/mask_buffer.py`：
  - `MaskBuffer` 保存多帧历史 fused mask。
  - 每个新帧只运行一次 SEA-RAFT 得到 `Flow_{t-1→t}`。
  - 用同一个 flow 将 buffer 中所有历史 mask warp 到当前帧坐标系。
  - 历史顺序为 newest → oldest。
- 将脚本中的单帧 `previous_fused_mask` 替换为多帧 history buffer。
- 新增 CLI 参数：
  - `--history-size`，默认 `3`
  - `--history-decay`，默认 `0.6`
  - `--artifact-alpha`，默认 `0.35`
  - `--artifact-non-free-threshold`，默认 `0.05`
- 新增退化帧自适应融合：
  - 普通帧使用 `--alpha` 和 `--non-free-threshold`
  - 特殊环境触发帧使用 `--artifact-alpha` 和 `--artifact-non-free-threshold`
- 更新 `Run.sh`，加入多帧历史和退化帧融合参数。
- 更新指标字段：
  - `temporal_iou_raw_vs_history`
  - `frame_difference_raw_vs_history`
  - `temporal_iou_fused_vs_history`
  - `frame_difference_fused_vs_history`
  - `history_count`
  - `effective_alpha`
  - `effective_non_free_threshold`

### Reason

- 单帧历史只利用 `t-1` 的 mask，水滴、强光、雪遮挡这种短时扰动下，历史信息不足。
- 不希望改动 SEA-RAFT 主体结构，因此采用外层 mask buffer 的方式增强时序融合。
- 特殊环境下 PIDNet 当前帧容易出现局部 road 低置信度，固定 `alpha` 和固定安全阈值会削弱历史 mask 的恢复能力。

### Result

- 语法检查通过：
  - `python3 -m py_compile src/temporal/mask_buffer.py src/scripts/run_cityscapes_sequence_clip_panels.py`
- 单 clip 验证：
  - 输出目录：`outputs/multiframe_smoke_clip46/`
  - 环境：`water_drop_slide`
  - `history_size=3`
  - `history_count` 从 `0 → 1 → 2 → 3` 后保持为 `3`
  - 特殊环境帧使用 `effective_alpha=0.25`
  - 特殊环境帧使用 `effective_non_free_threshold=0.03`
- 验证指标：
  - `mean_temporal_iou_raw_vs_history ≈ 0.870`
  - `mean_temporal_iou_fused_vs_history ≈ 0.952`
  - `mean_frame_total_time_s ≈ 0.100`
  - `total_clip_compute_time_s ≈ 7.51 s`

### Known Issues

- 多帧 history buffer 增加了额外 `cv2.remap` 开销，但没有增加 SEA-RAFT 调用次数。
- 当前多帧历史使用指数衰减加权，没有显式检测真实障碍物与相机退化区域；后续可以加入 artifact mask 或置信度图做局部权重控制。

## 2026-06-03

### Modified Module

- Docker Environment
- External Model
- YOLOP

### Changes

- 将 YOLOP 官方仓库 clone 到 `src/external/YOLOP`。
- 记录 YOLOP 当前 commit：
  - `8d8f68df318c71f01d6f813c024df646c7d1978f`
- 在项目 `requirements.txt` 中补充 YOLOP demo 所需依赖：
  - `Cython`
  - `tensorboardX`
  - `seaborn`
  - `prefetch_generator`
  - `imageio`
  - `scikit-learn`
- 在 Docker 容器 `perceive` 内安装并验证 YOLOP 依赖。
- 对 `src/external/YOLOP/tools/demo.py` 做一处兼容补丁：
  - 官方参数 `--weights` 使用 `nargs='+'`，显式传入权重时会得到 list。
  - 本地补丁将 list 权重路径转换为第一个路径，保证 `torch.load` 可以正常读取。

### Reason

- 评估 YOLOP 是否可以在 RTX 4060 + 当前 Docker 环境中运行，为后续比较 PIDNet 单帧 free-space 输出与 YOLOP drivable area 输出提供实验基础。
- 保持 Docker-first 工作流，不依赖宿主机 Python 环境。
- 不改动 YOLOP 主体网络结构，只做 demo 参数兼容和依赖适配。

### Result

- Docker 容器环境验证：
  - Python `3.10.12`
  - PyTorch `2.2.0+cu121`
  - CUDA 可用
  - GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`
  - OpenCV `4.5.4`
  - NumPy `1.26.4`
- YOLOP 模型构建验证通过：
  - model：`MCnet`
- 官方 demo 单图推理验证通过：
  - 输入：`src/external/YOLOP/test.jpg`
  - 权重：`src/external/YOLOP/weights/End-to-end.pth`
  - 输出：`src/external/YOLOP/inference/output_smoke_explicit/test.jpg`
  - 单图 smoke test 观测耗时：
    - `inf ≈ 0.1294 s/frame`
    - `nms ≈ 0.2951 s/frame`

### Known Issues

- YOLOP 官方环境较旧，当前采用 PyTorch 2.2 + CUDA 12.1 兼容运行，没有降级到官方旧版本。
- 当前只完成 YOLOP 原仓库 demo 级别环境适配，尚未接入本项目的模块化 wrapper、mask 输出格式、时序融合和指标评估流程。
- YOLOP 同时输出目标检测、drivable area 和 lane line；若用于本项目，应只抽取 drivable area 分支作为单帧 free-space baseline，不引入规划或 BEV 任务。

## 2026-06-03

### Modified Module

- YOLOP
- Demo Validation
- Docker Environment

### Changes

- 在 Docker 容器 `perceive` 内运行 YOLOP 官方 drivable area demo。
- 使用 YOLOP 自带图片集进行多图 smoke test：
  - 输入：`src/external/YOLOP/inference/images`
  - 输出：`outputs/yolop_drivable_demo_images`
- 使用 YOLOP 自带视频进行视频链路 smoke test：
  - 输入：`src/external/YOLOP/inference/videos/1.mp4`
  - 输出：`outputs/yolop_drivable_demo_video/1.mp4`
  - 预览帧：`outputs/yolop_drivable_demo_video/preview.jpg`

### Reason

- 验证 YOLOP 在当前 Docker + RTX 4060 环境下是否可以正常完成可通行区域推理。
- 通过图片和视频两类输入检查依赖、CUDA、模型加载、OpenCV 视频读写、可视化输出是否适配。

### Result

- 环境验证：
  - PyTorch `2.2.0+cu121`
  - CUDA 可用
  - GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`
  - OpenCV `4.5.4`
  - NumPy `1.26.4`
  - YOLOP model：`MCnet`
- 图片集 demo：
  - 6 张图片全部推理成功
  - `inf ≈ 0.0305 s/frame`
  - `nms ≈ 0.0524 s/frame`
- 视频 demo：
  - 输入视频：`267` 帧，`30 FPS`，时长 `8.9 s`
  - 输出视频：`267` 帧，`30 FPS`，时长 `8.9 s`
  - 总处理耗时：`28.914 s`
  - 端到端吞吐约 `9.33 FPS`
  - 网络前向：`inf ≈ 0.0099 s/frame`
  - 检测后处理：`nms ≈ 0.0021 s/frame`
- 可视化检查通过：
  - 绿色 overlay 表示 drivable area。
  - 检测框和 lane 可视化正常输出。

### Known Issues

- 当前 demo 输出仍包含目标检测框和 lane line，可通行区域不是单独 mask 文件。
- YOLOP 官方 demo 的端到端速度包含视频解码、OpenCV 绘制和 mp4 写入，因此不能直接等同于纯模型推理速度。
- 若用于本项目对比，需要新增 wrapper，只提取 drivable area probability / binary mask，并对齐本项目 `float32 H×W [0,1]` 的 free-space mask 格式。

## 2026-06-03

### Modified Module

- YOLOP
- Visualization
- Video Export

### Changes

- 将 YOLOP demo 生成的 OpenCV `mp4v` 视频转码为 H.264：
  - 原视频：`outputs/yolop_drivable_demo_video/1.mp4`
  - 转码后：`outputs/yolop_drivable_demo_video/1_h264.mp4`
- 使用 `ffmpeg` 设置：
  - codec：`libx264`
  - pixel format：`yuv420p`
  - `-movflags +faststart`

### Reason

- OpenCV 默认写出的 `mpeg4/mp4v` 在 VSCode、浏览器或部分系统播放器中可能无法直接播放。
- H.264 `avc1` + `yuv420p` 的兼容性更好，适合作为项目输出视频格式。

### Result

- 新视频验证通过：
  - codec：`h264`
  - codec tag：`avc1`
  - 分辨率：`1280×720`
  - 帧数：`267`
  - FPS：`30`
  - 时长：`8.9 s`
- 已成功从新视频解码预览帧：
  - `outputs/yolop_drivable_demo_video/preview_h264.jpg`

### Known Issues

- YOLOP 官方 demo 内部仍使用 OpenCV `mp4v` 写视频；后续如果要长期使用，需要统一增加视频导出工具或在 wrapper 输出阶段统一转 H.264。

## 2026-06-03

### Modified Module

- YOLOP
- Video Demo
- Visualization

### Changes

- 使用项目根目录 `eg.mp4` 作为 YOLOP 输入进行视频推理。
- 输出目录：
  - `outputs/yolop_eg_demo`
- 生成两个视频版本：
  - OpenCV 原始输出：`outputs/yolop_eg_demo/eg.mp4`
  - H.264 兼容输出：`outputs/yolop_eg_demo/eg_h264.mp4`
- 生成预览帧：
  - `outputs/yolop_eg_demo/preview_h264.jpg`

### Reason

- 验证 YOLOP 对项目已有视频输入的适配性，而不仅是官方 demo 数据。
- 检查当前 Docker 环境下 YOLOP 的视频读写、GPU 推理和 drivable area 可视化输出是否稳定。
- 使用 H.264 转码保证输出视频可以在 VSCode、浏览器和系统播放器中直接播放。

### Result

- 输入视频信息：
  - 分辨率：`1280×720`
  - FPS：`30`
  - 帧数：`488`
  - 时长：`16.2667 s`
- YOLOP 推理结果：
  - 总耗时：`31.544 s`
  - 端到端吞吐约 `15.63 FPS`
  - 网络前向：`inf ≈ 0.0098 s/frame`
  - 检测后处理：`nms ≈ 0.0015 s/frame`
- H.264 输出验证：
  - codec：`h264`
  - codec tag：`avc1`
  - pixel format：`yuv420p`
  - 帧数：`488`
  - 时长：`16.2667 s`
- 可视化检查通过：
  - 绿色区域为 YOLOP drivable area 输出。
  - 逆光场景下仍能输出大面积可通行区域，但边界与局部遮挡区域存在明显粗糙和误检。

### Known Issues

- 当前仍是 YOLOP 官方 demo 可视化结果，包含检测框和 lane line，不是项目需要的单通道 free-space mask。
- `eg.mp4` 存在强光/逆光影响，YOLOP 的 drivable area 边界有不稳定区域，后续可用于和 PIDNet、时序融合方法做对比。

## 2026-06-03

### Modified Module

- YOLOP
- SEA-RAFT
- Temporal Fusion
- Documentation
- STGRU Reproduction Plan

### Changes

- 将项目主线从 `PIDNet single-frame free-space` 调整为 `YOLOP single-frame drivable/free-space`。
- 新增项目内 YOLOP wrapper：
  - `src/yolop/yolop_wrapper.py`
  - 只暴露 YOLOP drivable-area segmentation head。
  - 输出格式为 `float32 H×W [0,1]`。
  - 不输出检测框，不输出 lane line。
- 新增 YOLOP 单帧/视频推理脚本：
  - `src/scripts/run_yolop_inference.py`
  - 输出 `raw_masks/*.npy`、manifest、H.264 overlay video、H.264 mask video。
- 新增 YOLOP + SEA-RAFT 时序融合脚本：
  - `src/scripts/run_yolop_temporal_fusion.py`
  - 当前仍使用 alpha fusion 作为 baseline。
  - 后续 STGRU 复现将替换该 fusion 模块。
- 将 `src/scripts/run_online_temporal_smoke.py` 默认输入从 PIDNet 切换为 YOLOP。
- 更新项目文档：
  - `README.md`
  - `AGENTS.md`
  - `docs/online_inference_plan.md`
  - `docs/yolop_stgru_reproduction_plan.md`
- 更新 `Run.sh`，默认运行：
  - `YOLOP → SEA-RAFT → warp → temporal fusion`

### Reason

- YOLOP 同时具备 drivable-area segmentation 分支和实时性优势，更适合作为当前项目的单帧可通行区域 baseline。
- 本项目不需要 YOLOP 的目标检测框，因此必须通过项目 wrapper 屏蔽 detection visualization。
- 论文复现方向要求固定 single-frame segmentation、optical flow、warp 三个模块，然后把手工 alpha fusion 替换为可训练 STGRU / GRFP。

### Result

- YOLOP wrapper smoke test 通过：
  - 输入：`eg.mp4`
  - 帧数：`8`
  - 输出：`outputs/yolop_eg_project_wrapper_smoke`
  - H.264 overlay video：
    - codec：`h264`
    - codec tag：`avc1`
    - 帧数：`8`
- YOLOP + SEA-RAFT temporal smoke test 通过：
  - 输入：`eg.mp4`
  - 帧数：`3`
  - 输出：`outputs/yolop_eg_temporal_smoke`
  - temporal panel：
    - codec：`h264`
    - codec tag：`avc1`
    - 帧数：`3`
- 语法检查通过：
  - `src/yolop/yolop_wrapper.py`
  - `src/scripts/run_yolop_inference.py`
  - `src/scripts/run_yolop_temporal_fusion.py`
  - `src/scripts/run_online_temporal_smoke.py`
  - `src/temporal/fusion.py`

### Known Issues

- 当前时序融合仍是手工 alpha fusion，不是论文中的可训练 STGRU。
- YOLOP 官方训练脚本仍以 BDD100K 多任务数据为默认输入；如果只训练 drivable 分支，后续需要新增项目内 fine-tune 脚本或改造 dataset/loss。
- SEA-RAFT 当前保持 frozen inference；后续复现论文时建议先训练 STGRU，不建议一开始端到端训练 optical flow。

## 2026-06-03

### Modified Module

- YOLOP
- Visualization

### Changes

- 使用项目内 YOLOP wrapper 重新生成 `eg.mp4` 的 drivable-only demo。
- 输出不包含 YOLOP 检测框。
- 输出不包含 lane line。
- 仅保留：
  - 可通行区域 overlay 视频
  - 单通道可通行区域 mask 视频
  - `raw_masks/*.npy`

### Reason

- 官方 YOLOP demo 会同时绘制检测框和车道线，不符合当前项目“YOLOP 只负责单帧可通行区域识别”的边界。
- 后续进入 SEA-RAFT / STGRU 时，需要的是干净的 drivable/free-space mask，而不是多任务可视化结果。

### Result

- 输入：`eg.mp4`
- 输出目录：`outputs/yolop_eg_drivable_only_demo`
- 输出视频：
  - `outputs/yolop_eg_drivable_only_demo/videos/eg_overlay_h264.mp4`
  - `outputs/yolop_eg_drivable_only_demo/videos/eg_mask_h264.mp4`
- 视频验证：
  - codec：`h264`
  - codec tag：`avc1`
  - 分辨率：`1280×720`
  - FPS：`30`
  - 帧数：`488`
  - 时长：`16.2667 s`
- 预览帧检查通过：
  - 无检测框
  - 无车道线
  - 仅显示绿色可通行区域

### Known Issues

- 当前 overlay 使用概率 mask 直接显示，因此绿色透明度会随 YOLOP 置信度变化。
- 逆光区域仍会影响 YOLOP 单帧可通行区域边界，后续需要通过 STGRU 或单帧模型微调进一步验证改善空间。

## 2026-06-03

### Modified Module

- YOLOP
- SEA-RAFT
- Warp
- Temporal Fusion
- Visualization

### Changes

- 在官方 YOLOP demo 原视频上运行“不带 STGRU”的时序基线流程。
- 单帧可通行区域由 YOLOP 输出 binary drivable mask。
- 相邻帧光流由 SEA-RAFT 输出，用于将历史 mask warp 到当前帧。
- 使用 alpha 融合：
  - `alpha = 0.7`
  - `non_free_threshold = 0.2`
- 输出与 YOLOP-only demo 可对比的视频和 30FPS 帧目录。

### Reason

- 建立 YOLOP-only 与 YOLOP + SEA-RAFT temporal fusion 的直接对比基线。
- 当前阶段不引入 STGRU，先验证光流对齐和简单融合本身对连续帧可通行区域稳定性的影响。

### Result

- 输入视频：`src/external/YOLOP/inference/videos/1.mp4`
- 输出目录：`outputs/yolop_official_demo_searaft_fusion_no_stgru`
- 输出视频：
  - `videos/raw_overlay_h264.mp4`
  - `videos/fused_overlay_h264.mp4`
  - `videos/fused_mask_h264.mp4`
  - `videos/temporal_panel_h264.mp4`
- 输出帧目录：
  - `raw_overlay_frames_30fps`
  - `raw_mask_frames_30fps`
  - `fused_overlay_frames_30fps`
  - `fused_mask_frames_30fps`
- 输出数组：
  - `raw_masks`
  - `warped_masks`
  - `fused_masks`
- 视频验证：
  - codec：`h264`
  - codec tag：`avc1`
  - FPS：`30`
  - 帧数：`267`
  - 时长：`8.9 s`

### Known Issues

- 当前融合仍是固定权重 alpha，没有学习到遮挡、误检和真实障碍物变化之间的差异。
- SEA-RAFT 只提供运动对齐，不会主动修正 YOLOP 的语义错误。
- 第一帧没有历史帧，因此 fused mask 等于 raw mask。

## 2026-06-03

### Modified Module

- YOLOP
- SEA-RAFT
- Visualization
- Scripts

### Changes

- 为 `run_yolop_inference.py` 增加：
  - `--save-frames`
  - `--no-video`
- 为 `run_yolop_temporal_fusion.py` 增加：
  - `--no-video`
- 使用官方 YOLOP demo 原视频 `src/external/YOLOP/inference/videos/1.mp4` 输出两组 30FPS 图像帧：
  - YOLOP-only
  - YOLOP + SEA-RAFT temporal fusion，不带 STGRU
- 本次实验不导出 mp4 视频。

### Reason

- 便于逐帧观察 YOLOP 单帧结果和 SEA-RAFT 时序融合结果之间的差异。
- 避免视频编码影响画面对比，也方便后续按帧计算指标或重新合成指定格式的视频。

### Result

- YOLOP-only 输出目录：`outputs/yolop_official_demo_frames_only`
- YOLOP + SEA-RAFT 输出目录：`outputs/yolop_official_demo_searaft_frames_only`
- 两组均使用：
  - 输入：`1.mp4`
  - FPS：`30`
  - mask mode：`binary`
  - 帧数：`267`
- 已确认没有输出 `mp4/avi/mov` 视频文件。
- 图像帧数量检查：
  - `overlay_frames_30fps`：267
  - `mask_frames_30fps`：267
  - `raw_overlay_frames_30fps`：267
  - `raw_mask_frames_30fps`：267
  - `fused_overlay_frames_30fps`：267
  - `fused_mask_frames_30fps`：267

### Known Issues

- YOLOP + SEA-RAFT 当前仍是固定 alpha 融合，不具备 STGRU 的时序状态学习能力。
- SEA-RAFT 对 YOLOP mask 的帮助主要体现在历史 mask 对齐与短期平滑，不会凭空恢复 YOLOP 从未识别出的语义区域。

## 2026-06-03

### Modified Module

- YOLOP
- SEA-RAFT
- Temporal Fusion
- Visualization
- Scripts

### Changes

- 为 `run_yolop_inference.py` 增加目标分辨率参数：
  - `--target-width`
  - `--target-height`
- 为 `run_yolop_temporal_fusion.py` 增加与 PIDNet clip 实验一致的参数接口：
  - `--target-width`
  - `--target-height`
  - `--history-size`
  - `--history-decay`
- 在 YOLOP + SEA-RAFT temporal fusion 中加入多帧 `MaskBuffer`：
  - 每帧只调用一次 SEA-RAFT 估计 `Flow_{t-1→t}`
  - 将历史 fused mask buffer 全部 warp 到当前帧坐标系
  - 使用 `history_decay` 合成历史参考 mask
  - 再用 alpha fusion 与当前 YOLOP mask 融合
- 使用官方 YOLOP demo 原视频 `1.mp4` 重新输出一组与 PIDNet clip 实验参数对齐的图像帧结果。

### Reason

- 之前 YOLOP + SEA-RAFT 只使用上一帧历史 mask，和 PIDNet clip 实验的三帧 history buffer 不完全一致。
- 用户需要先不考虑 STGRU，仅比较 SEA-RAFT + 手工 temporal fusion 是否带来可观察变化。
- 使用 `540x960` 与 spring-M 权重可以对齐之前 PIDNet 特殊环境实验的主要设置。

### Result

- 输入视频：`src/external/YOLOP/inference/videos/1.mp4`
- YOLOP-only 输出目录：`outputs/yolop_official_demo_pidnet_params_yolop_only_frames`
- YOLOP + SEA-RAFT 输出目录：`outputs/yolop_official_demo_pidnet_params_searaft_frames`
- 参数：
  - `target_width = 960`
  - `target_height = 540`
  - `mask_mode = probability`
  - `alpha = 0.7`
  - `non_free_threshold = 0.2`
  - `history_size = 3`
  - `history_decay = 0.6`
  - SEA-RAFT config：`config/eval/spring-M.json`
  - SEA-RAFT checkpoint：`models/Tartan-C-T-TSKH-spring540x960-M.pth`
- 输出均为 30FPS 图像帧，不导出视频。
- 帧数检查：
  - YOLOP-only overlay：267
  - YOLOP-only mask：267
  - YOLOP + SEA-RAFT raw overlay：267
  - YOLOP + SEA-RAFT raw mask：267
  - YOLOP + SEA-RAFT fused overlay：267
  - YOLOP + SEA-RAFT fused mask：267
  - YOLOP + SEA-RAFT raw/warped/fused `.npy`：各 267
- raw mask 与 fused mask 数值差异：
  - 平均 `mean_abs_diff`：约 `0.000527`
  - 最大帧 `mean_abs_diff`：约 `0.002043`
  - 平均 `diff > 0.01` 像素比例：约 `1.70%`

### Known Issues

- 当前输入视频是正常连续交通场景，YOLOP 单帧输出已经较稳定，因此 SEA-RAFT 融合后的肉眼差异很小。
- probability overlay 会让低置信度区域呈现轻微绿色雾感；这不是语义类别污染，而是概率 mask 可视化导致的透明叠加效果。
- 若需要更清晰的视觉差异，应额外输出 binary 可视化或差异热力图。

## 2026-06-03

### Modified Module

- YOLOP
- SEA-RAFT
- Visualization
- Scripts

### Changes

- 明确 `fuse` 在当前项目中表示 temporal fusion，即将当前帧单帧 free-space mask 与 SEA-RAFT 对齐后的历史 mask 融合为 `fused_mask`。
- 为 YOLOP-only 与 YOLOP + SEA-RAFT 脚本增加显示层参数：
  - `--vis-threshold`
  - `--vis-binary`
- 重新生成上一组 `1.mp4` 的图像帧输出：
  - 融合仍使用 probability mask
  - `.npy` 数组仍保存连续概率值
  - PNG 可视化使用 `vis_threshold = 0.5` 与 binary 渲染

### Reason

- probability mask 直接 overlay 时，背景低置信度区域也会被绿色通道叠加，造成“绿色雾感”。
- 该问题属于可视化表达问题，不应修改模型输出或 temporal fusion 的概率计算。
- 将计算 mask 与显示 mask 解耦后，可以同时保留概率融合能力和干净的人工观察画面。

### Result

- 已覆盖输出：
  - `outputs/yolop_official_demo_pidnet_params_yolop_only_frames`
  - `outputs/yolop_official_demo_pidnet_params_searaft_frames`
- 两组输出仍为 30FPS 图像帧，不导出视频。
- 帧数检查：
  - YOLOP-only overlay：267
  - YOLOP-only mask：267
  - YOLOP + SEA-RAFT raw overlay：267
  - YOLOP + SEA-RAFT raw mask：267
  - YOLOP + SEA-RAFT fused overlay：267
  - YOLOP + SEA-RAFT fused mask：267
- 抽帧检查显示背景绿色雾感已去除。

### Known Issues

- 当前二值可视化会弱化 probability mask 内部的置信度变化；如果需要分析置信度，应查看 `.npy` 数组或另行输出 heatmap。

## 2026-06-05

### Modified Module

- Project Structure
- YOLOP
- SEA-RAFT
- STGRU
- Utils
- Docker / File Permission

### Changes

- 删除根目录异常文件：
  - `=0.12`
  - `=0.29`
  - `=1.0`
  - `=1.2`
  - `=2.31`
  - `=2.6`
- 这些文件是 shell 将未正确引用的 pip 版本约束误解析为重定向后生成的临时文件，不属于项目代码。
- 将项目自有代码按功能重新整理：
  - `src/YOLOP/`
  - `src/SEA_RAFT/`
  - `src/STGRU/`
  - `src/utils/`
- 将共享代码迁移到 `src/utils/`：
  - `src/utils/temporal/`
  - `src/utils/visualization/`
  - `src/utils/datasets/`
  - `src/utils/scripts/`
  - `src/utils/legacy/`
- 将早期 ROS2 / PIDNet / RAFT 占位包移动到：
  - `src/utils/legacy/ros2_packages/`
- 将生成结果目录从 `outputs/` 统一为 `output/`。
- 将数据目录从 `datasets/` 统一为 `data/`。
- 将根目录示例视频移动到 `data/demo/eg.mp4`。
- 将权重集中到：
  - `weights/YOLOP/`
  - `weights/SEA-RAFT/`
  - `weights/STGRU/`
- 保留外部仓库内权重路径的软链接，避免旧官方命令直接失效。
- 更新：
  - `README.md`
  - `Run.sh`
  - `.gitignore`
  - `AGENTS.md`
  - `docs/online_inference_plan.md`

### Reason

- 当前项目已经从临时实验状态进入可维护工程状态，需要清晰区分：
  - 项目自有代码
  - 第三方外部仓库
  - 模型权重
  - 数据集
  - 实验输出
- 后续需要 Git 维护，因此大权重、数据集和输出文件必须从默认提交内容中隔离。
- Docker 容器默认 root 写入会导致宿主机删除/修改困难，因此统一修正 `data/`、`output/`、`weights/` 的文件归属。

### Result

- 新主入口：
  - YOLOP-only：`src/YOLOP/run_yolop_inference.py`
  - YOLOP + SEA-RAFT：`src/SEA_RAFT/run_yolop_temporal_fusion.py`
- 当前 `src/` 第一层主结构：
  - `src/YOLOP/`
  - `src/SEA_RAFT/`
  - `src/STGRU/`
  - `src/utils/`
  - `src/external/`
- 当前主输出目录：`output/`
- 当前数据目录：`data/`
- 当前权重目录：`weights/`
- 验证：
  - `python3 -m py_compile` 通过
  - Docker 容器内 `run_yolop_inference.py --help` 通过
  - Docker 容器内 `run_yolop_temporal_fusion.py --help` 通过
  - YOLOP 和 SEA-RAFT 权重集中路径检查通过

### Known Issues

- 根目录存在一个空的 `.git/` 目录，但当前不是有效 Git 仓库，`git status` 会失败；正式维护前需要重新 `git init` 或修复 `.git/`。
- `src/external/` 下仍保留第三方仓库本体，默认通过 `.gitignore` 排除；如需版本化第三方依赖，建议后续改为 git submodule。
- `src/SEA_RAFT/` 使用下划线是因为 Python 包名不能包含 `-`，文档和命令仍称模型为 SEA-RAFT。

## 2026-06-05

### Modified Module

- Project Structure
- YOLOP
- SEA-RAFT
- Utils / Legacy

### Changes

- 移除全局 `src/external/` 目录。
- 将第三方官方仓库移动到各自功能模块内部：
  - `src/YOLOP/external/YOLOP/`
  - `src/SEA_RAFT/external/SEA-RAFT/`
- 将旧 PIDNet 官方仓库移动到 legacy：
  - `src/utils/legacy/external/PIDNet/`
- 更新所有主入口默认路径：
  - YOLOP repo
  - SEA-RAFT repo
  - SEA-RAFT config
  - legacy PIDNet repo
- 更新：
  - `.gitignore`
  - `README.md`
  - `src/README.md`
  - `AGENTS.md`
  - `Run.sh`

### Reason

- 用户要求 `YOLOP`、`SEA-RAFT`、`STGRU` 三个功能文件夹各自维护自己的外部项目和核心代码。
- 全局 `src/external/` 会让工程结构变得松散，不利于按功能模块维护。

### Result

- 当前 `src/` 顶层结构：
  - `src/YOLOP/`
  - `src/SEA_RAFT/`
  - `src/STGRU/`
  - `src/utils/`
- 已确认：
  - 全局 `src/external/` 不存在
  - Docker 内 YOLOP 入口 `--help` 通过
  - Docker 内 SEA-RAFT 入口 `--help` 通过
  - 新 YOLOP 官方 demo 视频路径存在
  - 新 SEA-RAFT config 路径存在
  - 权重集中目录软链接正常

### Known Issues

- 第三方官方仓库仍默认被 `.gitignore` 忽略；若后续希望 Git 追踪外部仓库版本，应改为 submodule 或在文档中固定 clone commit。

## 2026-06-05

### Modified Module

- STGRU
- YOLOP
- SEA-RAFT
- Temporal Fusion
- Pipeline

### Changes

- 新增 `src/STGRU/stgru_module.py`，完成 STGRU 的基础结构设计：
  - `STGRUCell` 使用当前 YOLOP mask、warp 后历史 mask、图像光度误差计算 reset gate / update gate。
  - `STGRUFusionModule` 提供 numpy 输入输出接口，方便和 YOLOP、SEA-RAFT 在同一进程中通过变量通信。
- 新增统一大融合入口：
  - `src/utils/pipeline/freespace_temporal_pipeline.py`
  - `src/utils/scripts/run_freespace_pipeline.py`
- 将流程改为同一 Python 进程内实例化三个模块：
  - YOLOP 输出当前帧 free-space mask。
  - SEA-RAFT 输出相邻帧 optical flow。
  - Temporal / STGRU 根据 flow warp 历史 mask 并融合。
- 新增 `src/utils/temporal/warp_image.py`，用于将上一帧 RGB 图像按 flow 对齐到当前帧，给 STGRU 计算光度误差。
- 修正 SEA-RAFT wrapper 和项目级 `utils` 包的同名导入冲突，避免修改 SEA-RAFT 官方源码。
- 更新 `Run.sh`，默认调用统一 pipeline，当前默认仍使用 `alpha` 融合；`stgru` 模式可通过参数开启。

### Reason

- 用户要求先完成 STGRU 结构设计，再进行 YOLOP、SEA-RAFT、STGRU 的大融合。
- 当前阶段重点是建立清晰的模块边界和内存通信方式，而不是继续通过文件中转 mask / flow。
- STGRU 目前没有训练权重，因此需要先完成可训练、可替换 checkpoint 的结构，再进入训练和实验对齐阶段。

### Result

- 已完成统一流程：
  - `RGB frame -> YOLOP mask -> SEA-RAFT flow -> history warp -> alpha/STGRU fusion -> visualization frames / arrays`
- Docker 内完成验证：
  - `run_freespace_pipeline.py --help` 通过。
  - `fusion-mode alpha` 使用 `eg.mp4` 处理 2 帧 smoke test 通过。
  - `fusion-mode stgru` 使用未训练 STGRU 处理 2 帧 smoke test 通过。
- 当前 smoke test 输出：
  - `output/pipeline_alpha_smoke/`
  - `output/pipeline_stgru_smoke/`

### Known Issues

- STGRU 目前只是结构实现，未训练 checkpoint 下的输出只能用于链路验证，不能用于论文结论或性能比较。
- SEA-RAFT 在过小分辨率下会因为特征金字塔下采样失败，建议当前统一使用 `960x540` 或更高分辨率进行测试。
- 下一步需要准备连续帧训练数据和 free-space 标签，训练 STGRU 的 gate，使其学习何时信任当前 YOLOP，何时信任 warp 后历史结果。

## 2026-06-05

### Modified Module

- STGRU
- Training
- Documentation

### Changes

- 新增 `src/STGRU/train_stgru.py`，作为 STGRU 训练入口。
- 训练输入默认从工作目录 `data/` 读取：
  - Cityscapes bootstrap：`data/cityscapes/`
  - 预计算真实样本：`data/stgru_samples/*.csv`
- 训练权重默认输出到：
  - `weights/STGRU/stgru_best.pth`
  - `weights/STGRU/stgru_latest.pth`
- 支持两种训练模式：
  - 使用 `leftImg8bit_sequence + gtFine` 生成结构预训练/调试样本。
  - 使用预计算 YOLOP mask、SEA-RAFT warped mask、target mask 的 CSV 进行正式训练。
- 更新 `src/STGRU/README.md`，补充训练数据目录、命令和 CSV 字段说明。

### Reason

- STGRU 已经完成结构设计，但需要训练入口才能从未训练 gate 进入可验证模型。
- 正式实验需要让 STGRU 学习真实 YOLOP 单帧误差和 SEA-RAFT warp 历史结果之间的取舍，而不是只依赖手工 alpha。

### Result

- Docker 容器内验证：
  - `python3 src/STGRU/train_stgru.py --help` 通过。
  - 使用 `/tmp` 构造的 4 个小型预计算样本完成 1 epoch smoke test。
  - smoke test 成功输出 best/latest checkpoint 和 `training_log.csv`。

### Known Issues

- Cityscapes bootstrap 模式只有第 19 帧 gtFine 真值，脚本会合成 YOLOP-like 当前 mask 和历史 mask，适合结构预训练或调试，不适合作为最终论文性能结论。
- 正式训练建议先用当前 YOLOP + SEA-RAFT pipeline 生成真实 `current_mask / warped_mask / target_mask` CSV，再训练 STGRU。

## 2026-06-07

### Modified Module

- STGRU
- Training Workflow

### Changes

- 新增 `Run_STGRU.sh`，用于只运行 STGRU 训练，不触发 YOLOP、SEA-RAFT 或完整融合流程。
- `Run_STGRU.sh smoke` 默认使用少量样本、低分辨率进行训练链路检查。
- `Run_STGRU.sh train` 默认使用 `960x540` 和完整 Cityscapes bootstrap 数据进行 STGRU 训练。
- 当前训练输入固定为：
  - `/workspace/data/cityscapes`
- 当前训练输出固定为：
  - `/workspace/weights/STGRU_smoke`
  - `/workspace/weights/STGRU`

### Reason

- 当前阶段先聚焦 STGRU 训练，避免 YOLOP / SEA-RAFT 推理耗时干扰训练脚本和数据路径调试。
- 先通过 smoke test 验证数据结构、CUDA、loss 和 checkpoint 写入，再进入完整训练。

### Result

- 已检查当前数据：
  - `gtFine/train` 有 2975 个 `labelIds`
  - `gtFine/val` 有 500 个 `labelIds`
  - `leftImg8bit_sequence/train` 当前有 2880 张连续帧
  - `leftImg8bit_sequence/val` 当前没有连续帧
- `Run_STGRU.sh` shell 语法检查通过。

### Known Issues

- 当前没有运行中的 `perceive` Docker 容器，因此未直接启动训练。
- 当前 sequence 只有 train/bochum 子集，val 侧会回退到普通 `leftImg8bit`，适合先训练结构，不适合作为最终严格时序验证。

## 2026-06-07

### Modified Module

- STGRU
- Training

### Changes

- 修复 `train_stgru.py` 在 `--amp` 模式下的 loss 计算问题。
- 原因是 PyTorch 不允许在 autocast 区域内对概率值直接调用 `binary_cross_entropy`。
- 当前处理方式：
  - STGRU forward 仍可使用 AMP。
  - BCE / Dice loss 统一切回 FP32 计算。

### Reason

- 保持 4060 等显卡上训练时的 AMP 加速和显存节省。
- 避免将 STGRU 输出结构改成 logits，减少对现有推理接口的影响。

### Result

- 已运行 `./Run_STGRU.sh smoke`。
- smoke test 成功完成：
  - train: 64 samples, 1 epoch
  - val: 32 samples
  - 输出 `weights/STGRU_smoke/stgru_best.pth`
  - 输出 `weights/STGRU_smoke/stgru_latest.pth`
  - 输出 `weights/STGRU_smoke/training_log.csv`

### Known Issues

- smoke 只验证训练链路，不代表模型效果。
- 当前 bootstrap 数据仍是合成 YOLOP-like mask，正式训练应使用真实 YOLOP + SEA-RAFT 预计算样本。

## 2026-06-07

### Modified Module

- STGRU
- Cityscapes Preprocessing
- Training
- Evaluation
- Visualization

### Changes

- 新增 `src/STGRU/prepare_cityscapes_binary.py`：
  - 将 Cityscapes 多类别 `labelIds` 转换为二值 free-space mask。
  - 当前默认 `labelId=7` 作为 road / free-space。
- 新增 `src/STGRU/precompute_stgru_samples.py`：
  - 批量生成 STGRU 训练需要的真实输入。
  - 输入包括 `current_mask`、`warped_mask`、`target_mask`、`photometric_error`、`flow`、`warped_image`。
  - `current_mask` 和 `previous_mask` 来自 YOLOP。
  - `flow` 来自 SEA-RAFT。
  - `warped_mask` 由 previous YOLOP mask 经 SEA-RAFT flow warp 得到。
- 更新 `src/STGRU/train_stgru.py`：
  - 支持 `--test-sample-list`。
  - 训练完成后自动输出 val/test 的 IoU、Precision、Recall、F1、Accuracy。
- 新增 `src/STGRU/run_stgru_video.py`：
  - 使用训练后的 STGRU checkpoint 对视频输出 fused mask 和 overlay 视频。
- 更新 `Run_STGRU.sh`：
  - 新增 `precompute-smoke`
  - 新增 `precompute`
  - 新增 `train-precomputed-smoke`
  - 新增 `train-precomputed`
  - 新增 `video`

### Reason

- 贴近论文主线：固定 YOLOP 和 SEA-RAFT，预计算单帧语义结果和光流对齐结果，只训练 STGRU 时序融合模块。
- Cityscapes 原始标签是多类别语义标签，STGRU 当前任务是二值 free-space，因此需要在训练前显式转换类别定义。

### Result

- 已生成 Cityscapes 二值标签：
  - `data/cityscapes_binary/`
  - 共 5000 张 binary mask。
- 已完成真实预计算 smoke：
  - `data/stgru_samples_smoke/train.csv`：2 samples
  - `data/stgru_samples_smoke/val.csv`：1 sample
  - `data/stgru_samples_smoke/test.csv`：1 sample
- 已完成预计算样本训练 smoke：
  - `weights/STGRU_precompute_smoke/stgru_best.pth`
  - `weights/STGRU_precompute_smoke/stgru_latest.pth`
  - `weights/STGRU_precompute_smoke/training_log.csv`
  - `weights/STGRU_precompute_smoke/evaluation_summary.csv`
- smoke 指标：
  - val IoU: 0.8749
  - val F1: 0.9333
  - test IoU: 0.0
  - test Accuracy: 0.6262
- 已导出 STGRU 视频 smoke：
  - `output/stgru_video_smoke/stgru_mask.mp4`
  - `output/stgru_video_smoke/stgru_overlay.mp4`
  - `output/stgru_video_smoke/stgru_mask_h264.mp4`
  - `output/stgru_video_smoke/stgru_overlay_h264.mp4`

### Known Issues

- 当前没有发现 `data/demo/1.mp4`，视频导出脚本回退使用了 `data/demo/eg.mp4`。
- 当前 train 真实时序样本来自已有的 `leftImg8bit_sequence/train/bochum` 子集。
- 当前 val/test 没有 sequence，预计算脚本回退使用同帧图像，适合链路验证，不适合最终时序性能结论。
- test smoke 只有 1 个样本，且当前指标不代表模型真实性能。

## 2026-06-07

### Modified Module

- STGRU
- Visualization

### Changes

- 使用 YOLOP 官方仓库自带原始视频运行 STGRU demo：
  - `src/YOLOP/external/YOLOP/inference/videos/1.mp4`
- 输出 STGRU fused mask 视频和 overlay 视频。
- 额外转码 H.264 / yuv420p 版本，避免播放器兼容问题。

### Reason

- 用户要求在 YOLOP demo 的 `1.mp4` 上输出结果。
- 之前查找视频时排除了 `external` 目录，导致误用 `data/demo/eg.mp4` 作为回退输入。

### Result

- 输出目录：
  - `output/yolop_1mp4_stgru_demo/`
- 输出文件：
  - `stgru_mask.mp4`
  - `stgru_overlay.mp4`
  - `stgru_mask_h264.mp4`
  - `stgru_overlay_h264.mp4`

### Known Issues

- 当前使用的是 `weights/STGRU_precompute_smoke/stgru_best.pth`，只是链路验证权重，不代表最终 STGRU 效果。

## 2026-06-07

### Modified Module

- STGRU
- Temporal Fusion
- Visualization

### Changes

- 修复 STGRU fused mask 大面积铺满画面的错误。
- `STGRUCell` 不再对已经是概率的融合结果再次 `softmax`。
- `STGRUFusionModule` 新增 no-hallucination 约束：
  - fused free-space 不能超过当前 YOLOP mask 和 warped history mask 提供的支持上界。
  - 当前帧高置信 non-free 区域仍保持当前 YOLOP 结果。
  - `support_margin` 只加在当前帧支持上，不重复加到 recurrent history 上，避免背景概率逐帧漂移到 0.5 以上。
- 重新使用 YOLOP 官方 `1.mp4` 输出修正版 STGRU demo。

### Reason

- 原错误不是 SEA-RAFT 凭空生成 mask，而是 STGRU 融合层允许未被当前帧或历史帧支持的区域被输出为 free-space。
- 旧实现把概率再次作为 logits 做 `softmax`，会把低置信背景区域推向 0.5。
- 错误 fused mask 被写入 temporal buffer 后，下一帧又作为 warped history 输入，导致错误被递归传播。

### Result

- 修复前前几帧统计：
  - frame 2 fused `>=0.5` 像素比例约 0.9793。
  - frame 3 fused `>=0.5` 像素比例约 0.9923。
- 修复后 30 帧统计：
  - frame 0 fused `>=0.5` 比例约 0.1467。
  - frame 10 fused `>=0.5` 比例约 0.1490。
  - frame 29 fused `>=0.5` 比例约 0.1700。
- 修正版视频输出：
  - `output/yolop_1mp4_stgru_demo_fixed/stgru_mask_h264.mp4`
  - `output/yolop_1mp4_stgru_demo_fixed/stgru_overlay_h264.mp4`
- 修正版视频抽样 mask 面积比例：
  - frame 0: 0.1463
  - frame 50: 0.1717
  - frame 100: 0.1926
  - frame 150: 0.2217
  - frame 266: 0.2025

### Known Issues

- 当前仍使用 smoke 训练权重，视频只能证明“不会再凭空铺满全图”，不能证明 STGRU 已经达到最终效果。
- 后续正式训练需要用完整 sequence train/val/test 预计算样本重新训练 STGRU。

## 2026-06-12

### Modified Module

- BDD100K Downloader
- STGRU Dataset
- STGRU Precompute

### Changes

- 更新 `download_bdd100k_video_scenes.py`，支持 BDD100K STGRU 训练数据准备：
  - 默认下载/整理 100 个场景。
  - 支持 `--selection-mode stratified`，优先根据 BDD image labels 中的 `weather / scene / timeofday` 分层随机选择，尽可能覆盖不同环境。
  - 支持 `--random-seed` 控制随机复现。
  - 支持从 drivable map 标签中筛选有可通行区域监督的 scene。
  - 支持 `--prepare-stgru`，下载后对每个场景抽取 9-12 秒共 90 帧。
  - 以 10 秒标注帧为中心，截取左右各 10 帧，形成 21 帧 STGRU sequence。
  - 将 BDD drivable map 转成二值 `target_mask.npy`。
  - 默认划分 `80 train / 10 val / 10 test`，数量可通过参数调整。
- 新增 `src/STGRU/precompute_bdd100k_stgru_samples.py`：
  - 读取 BDD STGRU scene manifest。
  - 对中心帧和前一帧运行 YOLOP。
  - 使用 SEA-RAFT 估计光流并 warp previous mask。
  - 生成 `train_stgru.py` 可直接读取的 `train.csv / val.csv / test.csv`。
- 新增 `Run_BDD100K_STGRU.sh`：
  - `download`
  - `precompute`
  - `train`
  - `all`
  - 支持通过环境变量调整下载数量、划分数量、seed、分辨率、batch size、epoch。
- 恢复 `.gitignore` 对 `weights/` 的忽略，并忽略 `*.before_*` 备份文件。

### Reason

- 云平台训练需要能够直接从 BDD100K 下载视频和 drivable 标签，并自动整理成 STGRU 训练格式。
- BDD100K 的 drivable 标签对应视频第 10 秒标注帧，因此需要围绕 9-12 秒抽帧，并以 10 秒帧作为监督中心。
- STGRU 训练需要时序上下文帧，但监督标签只在中心帧，因此数据准备需要明确保存 sequence 和中心帧 target。

### Result

- 本地完成语法验证：
  - `download_bdd100k_video_scenes.py --help` 通过。
  - `src/STGRU/precompute_bdd100k_stgru_samples.py --help` 通过。
  - `Run_BDD100K_STGRU.sh` shell 语法检查通过。
- 使用 `/tmp` 构造 fake BDD 场景完成抽帧 smoke：
  - 9-12 秒抽出 90 帧。
  - 10 秒中心帧左右各 10 帧得到 21 帧 sequence。
  - 成功生成 train scene manifest 和 binary target mask。

### Known Issues

- BDD100K 官方下载入口需要合法登录 Cookie 或显式提供直链，脚本不会绕过访问控制。
- 分层覆盖依赖 `bdd100k_labels_images_*.json` 中的 attributes；如果只提供 video zip 而不提供 image label/drivable map 包，脚本会退化为随机视频选择。
- 当前 BDD 预计算脚本每个 scene 生成一个中心帧监督样本；如需利用 21 帧内多个传播步训练，需要后续扩展 recurrent unroll 训练。

## 2026-06-10

### Modified Module

- Dataset
- Scripts

### Changes

- 新增 `download_bdd100k_video_scenes.py`。
- 支持从 BDD100K 下载页抓取 video / label 链接，包括 `window.open(...)` 按钮形式的下载地址。
- 已将 BDD100K `Videos` 按钮对应地址纳入默认候选：
  - `http://128.32.162.150/bdd100k/bdd100k_videos.zip`
- 支持通过 manifest 或直链手工提供下载地址。
- 支持 browser cookie / Netscape cookie 文件，用于已登录 BDD100K 用户门户后的合法下载。
- 将下载到的视频按单个 scene 拆成独立文件夹：
  - `scenes/<scene_id>/video/`
  - `scenes/<scene_id>/labels/`
  - `scenes/<scene_id>/meta.json`
- 增加 `--num-scenes`、`--max-total-size`、`--split`、`--scene-name`、`--scene-list` 等命令行参数。
- 对网络下载量和最终落盘目录同时做大小限制，默认不超过 `1G`，超过后立即停止。
- 对远程 HTTP zip 支持 Range 分块读取：
  - 先读取 Zip64 central directory。
  - 只下载目标 scene 对应的视频成员和 label 成员。
  - 不下载完整 `bdd100k_videos.zip`。
- 默认只下载 `bdd100k_labels.zip` 中与目标 scene 匹配的 JSON label；其他 label 包可通过 `--label-url` 或 `--label-pattern` 显式指定。

### Reason

- 后续 YOLOP / SEA-RAFT / Temporal Fusion 实验需要可控规模的 BDD100K 视频子集。
- 单次下载完整 BDD100K video 数据过大，不适合当前项目的增量调试。
- 按 scene 组织视频和标签，便于逐场景运行 temporal consistency 评估和可视化导出。

### Result

- 已用本地 mock BDD100K video zip 和 label zip 进行 smoke test。
- 测试结果显示脚本可以拆出独立 scene 文件夹，并将匹配的 JSON label 写入对应目录。
- 已对官方 `bdd100k_videos.zip` 执行远程 zip 目录解析测试：
  - 完整 video zip 大约 `1.79 TiB`。
  - 服务器支持 `Accept-Ranges: bytes`。
  - 读取 central directory 约 `12.33 MiB`，可列出 `100000` 个视频成员。
- 已对 1 个 train scene 执行真实 Range 下载测试：
  - 成功下载 `0000f77c-6257be58.mov`。
  - 成功从 `bdd100k_labels.zip` 下载同名 JSON label。
  - 输出结构符合单 scene 文件夹组织。

### Known Issues

- 若服务器关闭 HTTP Range，脚本会回退到整包下载逻辑；此时完整 video archive 会超过 `1G` 并按限制停止。
- 当前 label 默认使用 `bdd100k_labels.zip`。如果实验需要 drivable area label，还需要显式传入 `bdd100k_drivable_maps.zip` 或相关 2021 drivable label 包，并确认其文件命名能与 video scene id 对齐。

## 2026-06-11

### Modified Module

- Dataset
- Scripts

### Changes

- 明确 BDD100K `Videos` 下载内容是 `.mov` 视频 clip，不是已切好的图片序列。
- 修正 `download_bdd100k_video_scenes.py` 的 label 匹配逻辑：
  - 支持 `*_drivable_color.png`
  - 支持 `*_drivable_id.png`
  - 支持 `*_train_color.png` / `*_train_id.png`
- 默认 label 下载范围从仅 `bdd100k_labels.zip` 扩展为：
  - `bdd100k_labels.zip`
  - `bdd100k_drivable_maps.zip`
- 新增视频后处理参数：
  - `--clip-start`
  - `--clip-end`
  - `--clip-duration`
  - `--extract-frames`
  - `--frame-fps`
  - `--discard-full-video`
- 使用 `ffmpeg` 将下载后的视频截取为指定时间片段，并可导出图片帧。

### Reason

- 当前项目的 YOLOP / SEA-RAFT / STGRU 流程更适合使用连续图片帧或短视频片段调试。
- 用户需要从约 `10s` 附近截取 `7s-13s` 的局部片段，避免每次处理完整 40s 视频。
- 基础 JSON label 不包含像素级 free-space / drivable mask，需要额外下载 drivable map 文件。

### Result

- 已用本地已有 BDD100K `.mov` 验证 `7s-13s` 截取：
  - 输出 clip 时长约 `6.006s`
  - `--frame-fps 2` 输出 `12` 张图片帧
- `python3 -m py_compile download_bdd100k_video_scenes.py` 通过。

### Known Issues

- 远程 zip 内的视频不能可靠地只下载 `7s-13s` 字节范围；脚本仍需先下载完整单 scene `.mov`，再本地截取片段。
- `bdd100k_drivable_maps.zip` 是按 image/frame id 提供的 drivable mask；并非所有 video scene 都一定有对应的分割 mask。

## 2026-06-12

### Modified Module

- Dataset
- STGRU
- Scripts

### Changes

- 修改 `download_bdd100k_video_scenes.py`：
  - 新增 `--local-drivable-root`，默认读取 `data/bdd100k_drivable_maps`。
  - 新增 `--bdd-drivable-values`，默认将 `*_drivable_id.png` 中的 `1,2` 视为 free-space。
  - 在 `--prepare-stgru` 阶段优先从本地 BDD100K drivable maps 中按 scene id 匹配监督标签。
  - 本地标签索引会根据 `--split` 过滤 `train/val/test`，避免 train 视频和 val 标签错配。
  - 输出 STGRU scene manifest 时新增 `source_label` 字段，便于追踪监督标签来源。
- 修改 `Run_BDD100K_STGRU.sh`：
  - 增加 `BDD_DRIVABLE_ROOT` 和 `BDD_DRIVABLE_VALUES` 环境变量。
  - 批处理下载/整理阶段自动传入本地 drivable maps 路径。

### Reason

- 本地已经存在 BDD100K drivable maps 精细标注帧和像素级标签，训练 STGRU 时应直接使用这些标签作为监督信号。
- 避免训练准备阶段依赖远程 label 包下载，也避免只下载视频后找不到对应 drivable mask。

### Result

- 已验证本地 `data/bdd100k_drivable_maps` 可建立标签索引：
  - `train`: `70000` 个 scene label
  - `val`: `10000` 个 scene label
- 已验证 `*_drivable_id.png` 可转换为 `float32` 的二值 `target_mask.npy`：
  - shape: `720 x 1280`
  - value range: `0/1`
- `python3 -m py_compile download_bdd100k_video_scenes.py` 通过。
- `bash -n Run_BDD100K_STGRU.sh` 通过。

### Known Issues

- 当前 BDD100K STGRU 数据准备仍以第 10 秒精细标注帧作为单帧监督中心，只生成中心帧附近 `+-10` 帧上下文。
- 若后续要训练完整多步 recurrent STGRU，需要把当前 manifest 扩展为多时间步 supervision 或 sequence-level unroll。

## 2026-06-12

### Modified Module

- Dataset
- Scripts

### Changes

- 修复 `download_bdd100k_video_scenes.py` 在远程 BDD label zip 目录解析超时时直接退出的问题。
- 新增 `--skip-remote-label-selection` 参数：
  - 启用后不再尝试读取远程 label zip 的 image attributes。
  - 直接使用本地 `data/bdd100k_drivable_maps` 随机选择有监督标签的 scene。
- 修改 `Run_BDD100K_STGRU.sh`：
  - 新增 `SKIP_REMOTE_LABEL_SELECTION` 环境变量。
  - 默认值为 `1`，适配云平台网络不稳定或远程 label zip 响应慢的情况。

### Reason

- 云平台运行时读取远程 BDD label zip central directory 可能出现 `TimeoutError`。
- 当前训练目标只需要保证下载的视频 scene 有对应 drivable mask 监督；本地 drivable maps 已经足够完成这个匹配。

### Result

- 远程 label zip 解析失败时，脚本会 warning 并回退到本地 drivable maps 随机选择。
- 批处理入口默认跳过远程 label 解析，减少下载前置阶段的不稳定性。
- `python3 -m py_compile download_bdd100k_video_scenes.py` 通过。
- `bash -n Run_BDD100K_STGRU.sh` 通过。

### Known Issues

- 跳过远程 image attributes 后，场景选择不再按 weather / scene / timeofday 分层，只能保证从有 drivable mask 的 scene 中随机采样。
- 如果后续必须覆盖天气和时间属性，需要额外提供本地 `bdd100k_labels_images_*.json`，再从本地 JSON 做分层选择。

## 2026-06-12

### Modified Module

- SEA-RAFT
- STGRU
- Dataset

### Changes

- 修复 BDD100K STGRU 预计算阶段的 `ModuleNotFoundError: No module named 'utils.temporal'`。
- 原因是 SEA-RAFT 加载时会临时导入第三方库自己的顶层 `utils`，污染 `sys.modules` 后导致项目内 `src/utils/temporal` 无法被正确解析。
- 修改 `src/SEA_RAFT/sea_raft_wrapper.py`：
  - 在加载 SEA-RAFT 内部 `utils.utils` 后，若原本没有 `utils` 模块，则清理临时缓存。
  - 若原本已有项目 `utils`，则恢复原模块。
- 修改 `src/STGRU/precompute_bdd100k_stgru_samples.py`：
  - 在脚本初始化时提前导入项目内 `warp_mask_with_flow` 和 `warp_image_with_flow`。

### Reason

- BDD100K 数据准备已经成功生成 STGRU scene manifest，但预计算 YOLOP + SEA-RAFT 输入时被第三方库命名冲突中断。
- 该修复保证项目自有模块和外部模型仓库可以在同一进程内稳定协作。

### Result

- 已在 Docker 容器 `perceive` 中继续运行，不重新下载数据。
- `./Run_BDD100K_STGRU.sh precompute` 成功：
  - train: `3` samples
  - val: `1` sample
  - test: `1` sample
- `EPOCHS=1 BATCH_SIZE=2 ./Run_BDD100K_STGRU.sh train` 成功：
  - 输出 `weights/STGRU_BDD100K/stgru_best.pth`
  - 输出 `weights/STGRU_BDD100K/stgru_latest.pth`
  - 输出 `training_log.csv`
  - 输出 `evaluation_summary.csv`
- smoke 指标：
  - val IoU: `0.7707`
  - test IoU: `0.5547`

### Known Issues

- 当前 smoke 只使用 `5` 个 BDD100K scene，指标只用于验证流程可运行，不能代表模型真实性能。
- 正式训练仍需要扩大到 `100` 个或更多 scene，并观察 val/test 是否稳定。

## 2026-06-12

### Modified Module

- Repository
- Docker
- Documentation

### Changes

- 整理 GitHub 提交前的仓库结构：
  - 从 Git 索引中移除已跟踪的大权重文件和生成结果。
  - 保留 `weights/YOLOP/.gitkeep`、`weights/SEA-RAFT/.gitkeep`、`weights/STGRU/.gitkeep`，维持权重目录结构。
  - 更新 `.gitignore`，忽略第三方库内部权重、外部 demo 输出、Codex 本地目录、数据集、输出和训练权重。
  - 更新 `.dockerignore`，避免 Docker build context 包含 `data/`、`output/`、`weights/` 等大目录。
- 新增 `docs/remote_training_ssh.md`：
  - 记录 GitHub push、SSH 登录服务器、Docker 构建、权重/BDD 标签同步、smoke 训练和正式 100 scene 训练流程。
- 更新 `README.md`：
  - 增加 BDD100K STGRU 训练入口和必要资产说明。

### Reason

- 准备将项目推送到 GitHub，并在远程 GPU 服务器上通过 SSH 进行最终部署和训练。
- 避免将本地数据集、输出视频、训练中间结果和大权重文件提交到 GitHub。

### Result

- 当前 Git 跟踪文件中最大文件约为 YOLOP demo 视频 `11.8 MB`，已移除 `95 MB` 级 YOLOP 权重和 `78 MB` 级 SEA-RAFT 权重的 Git 跟踪。
- Docker build context 将不再包含本地 BDD100K 数据、实验输出和权重目录。

### Known Issues

- 早期提交历史中可能仍包含曾经提交过的大权重文件；当前整理保证后续提交不再继续跟踪这些文件。
- 如果 GitHub push 因历史大文件失败，需要使用 `git filter-repo` 或 BFG 清理历史。

## 2026-06-12

### Modified Module

- Repository
- Documentation
- Scripts

### Changes

- 根据项目部署需求，恢复将 YOLOP 和 SEA-RAFT 必要 `.pth` 权重纳入 Git：
  - `weights/YOLOP/End-to-end.pth`
  - `weights/SEA-RAFT/Tartan-C-T-TSKH-spring540x960-M.pth`
- `.gitignore` 保留对 STGRU smoke 输出、训练结果、ONNX 导出和数据集的忽略，但放行上述两个必要权重。
- `Run_BDD100K_STGRU.sh` 新增可配置变量：
  - `YOLOP_CHECKPOINT`
  - `SEA_RAFT_CONFIG`
  - `SEA_RAFT_CHECKPOINT`
  - `SEA_RAFT_URL`
- 更新远程部署文档：
  - 说明 YOLOP / SEA-RAFT 必要权重随 Git 下发。
  - 保留 BDD100K drivable maps 通过 `rsync` 或云端下载准备。
  - 增加切换本地 SEA-RAFT `spring-M` checkpoint 的命令。

### Reason

- 远程服务器部署时，STGRU precompute 必须可直接访问 YOLOP 权重。
- 用户希望将 SEA-RAFT 权重也随仓库维护，减少远程服务器对 Hugging Face 网络访问的依赖。

### Result

- 当前将必要 `.pth` 权重恢复到 Git 暂存区。
- 不提交 `weights/STGRU_BDD100K/`、`weights/STGRU_smoke/`、`weights/STGRU_precompute_smoke/` 等训练输出。

### Known Issues

- 当前环境没有安装 `git lfs`，权重将作为普通 Git 文件提交。
- YOLOP 权重约 `92 MB`，SEA-RAFT 权重约 `76 MB`，低于 GitHub 单文件 `100 MB` 限制，但 push 时可能出现 large file warning。

## 2026-06-12

### Modified Module

- Environment
- Documentation

### Changes

- 新增 `Setup_Cloud_Env.sh`，用于云平台已经提供容器、但不支持 Docker-in-Docker 的情况。
- 脚本默认适配云平台已有环境：
  - Python `3.12`
  - PyTorch `2.3`
  - CUDA `12.1`
- 新增 `INSTALL_TORCH` 控制：
  - `auto`: 默认模式，已有可用 CUDA PyTorch 时不重装。
  - `0`: 强制跳过 PyTorch 安装。
  - `1`: 强制安装指定版本 PyTorch。
- 默认 PyTorch 备用安装版本调整为：
  - `torch==2.3.0`
  - `torchvision==0.18.0`
  - `torchaudio==2.3.0`
  - CUDA wheel index: `cu121`
- 更新 `docs/remote_training_ssh.md`：
  - 增加 Docker 与非 Docker 两套云端部署路径。
  - 说明 Python 3.12 / PyTorch 2.3 / CUDA 12.1 环境下推荐 `INSTALL_TORCH=0`。

### Reason

- 云平台本身已经运行在容器内，无法再次 build Docker。
- Python 3.12 不适合强行安装旧版 PyTorch 2.2，因此需要复用平台提供的 PyTorch 2.3 环境。

### Result

- `bash -n Setup_Cloud_Env.sh` 通过。
- 云平台部署流程调整为：
  - `git clone`
  - `INSTALL_TORCH=0 ./Setup_Cloud_Env.sh`
  - 准备 `data/bdd100k_drivable_maps`
  - 运行 `Run_BDD100K_STGRU.sh`

### Known Issues

- 如果云平台禁止 `apt-get`，需要使用 `INSTALL_APT=0`，并确保平台镜像已预装 `ffmpeg`、OpenCV 运行库和基础编译工具。

## 2026-06-12

### Modified Module

- Environment
- Documentation

### Changes

- 将云端非 Docker 环境脚本改回项目原始环境要求：
  - Python `3.10`
  - PyTorch `2.2.0`
  - torchvision `0.17.0`
  - torchaudio `2.2.0`
  - CUDA wheel index `cu121`
- `Setup_Cloud_Env.sh` 新增 Python 版本检查：
  - 默认要求 `python3` 为 `3.10`。
  - 如果云平台当前是 Python `3.12`，脚本会直接退出并提示切换云平台镜像/模板。
  - 可通过 `PYTHON_BIN=python3.10` 指定 Python 3.10 可执行文件。
- `INSTALL_TORCH` 默认改回 `1`，即按项目要求安装指定 PyTorch 版本。
- 更新 `docs/remote_training_ssh.md`：
  - 明确云平台应选择 Python 3.10 模板，而不是让项目适配 Python 3.12 / PyTorch 2.3。

### Reason

- 项目环境已经在 Dockerfile 中固定为 PyTorch 2.2 系列，并围绕 YOLOP、SEA-RAFT、STGRU 的兼容性进行调试。
- 为减少远程训练中的不可控变量，云平台应适配项目标准环境，而不是反向修改项目依赖。

### Result

- 非 Docker 云平台部署路径与 Dockerfile 的核心 Python / PyTorch 版本重新对齐。

### Known Issues

- 若云平台只提供 Python 3.12 容器且无法切换镜像，则不能直接使用当前项目标准环境；需要更换云平台模板或申请 Python 3.10 镜像。

## 2026-06-12

### Modified Module

- Environment
- Documentation

### Changes

- 在 `Setup_Cloud_Env.sh` 中新增受限平台兼容开关：
  - `ALLOW_PYTHON_MISMATCH=1`
- 默认仍要求项目标准环境：
  - Python `3.10`
  - PyTorch `2.2.0`
- 当云平台只能提供 `Python 3.12 + PyTorch 2.3` 时，可显式运行：
  - `ALLOW_PYTHON_MISMATCH=1 INSTALL_TORCH=0 ./Setup_Cloud_Env.sh`
- 更新 `docs/remote_training_ssh.md`，加入平台受限时的 smoke 验证流程。

### Reason

- 当前云平台只有 Python `3.12` 和 PyTorch `2.3`，无法完全按项目标准环境配置。
- 为避免彻底阻塞远程训练，提供一个显式的兼容模式，但不把它标记为标准复现实验环境。

### Result

- 标准路径仍保持 Python `3.10` + PyTorch `2.2.0`。
- 平台受限路径可以先安装项目其余依赖并执行小规模 smoke，验证 YOLOP、SEA-RAFT、STGRU 链路是否能在 PyTorch `2.3` 下工作。

### Known Issues

- 兼容模式未经完整正式训练验证；如果 smoke 或正式训练出现版本相关问题，需要切换云平台模板到 Python `3.10`。

## 2026-06-12

### Modified Module

- Environment
- Docker
- STGRU

### Changes

- 新增本地云平台模拟镜像：
  - `docker/Dockerfile.cloud-py312-torch23`
  - Python `3.12`
  - PyTorch `2.3.0+cu121`
  - torchvision `0.18.0+cu121`
  - torchaudio `2.3.0+cu121`
- 将模拟镜像中的 `opencv-python-headless` 固定为 `4.9.0.80`，避免新版 OpenCV 强制拉起 NumPy `2.x`。
- 在本地 Docker 中验证云平台受限环境：
  - `ALLOW_PYTHON_MISMATCH=1 INSTALL_TORCH=0 INSTALL_APT=0 ./Setup_Cloud_Env.sh`
  - YOLOP wrapper 导入
  - SEA-RAFT wrapper 导入
  - STGRU wrapper 导入
  - STGRU CUDA 前向融合

### Reason

- 云平台当前只提供 Python `3.12` 和 PyTorch `2.3`，与项目标准环境 Python `3.10` + PyTorch `2.2` 不一致。
- 需要在本地复刻该环境，提前暴露版本兼容问题，降低远程训练部署风险。

### Result

- 本地镜像 `perceive-cloud-py312-torch23:latest` 构建并验证通过。
- 验证环境：
  - Python `3.12.13`
  - PyTorch `2.3.0+cu121`
  - CUDA `12.1`
  - OpenCV `4.9.0`
  - NumPy `1.26.4`
  - GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- STGRU 融合输出验证通过，输出 shape 为 `64x96` 的 `float32` mask。

### Known Issues

- 该镜像是云平台受限环境的 smoke 验证，不是项目标准复现实验环境。
- 尚未在 Python `3.12` + PyTorch `2.3` 下完成完整 BDD100K 预计算和正式 STGRU 训练。
- 如果远程正式训练出现版本相关问题，优先切换回项目标准环境 Python `3.10` + PyTorch `2.2`。

## 2026-06-12

### Modified Module

- STGRU
- Cityscapes Dataset
- Training Script

### Changes

- 新增 `Run_Cityscapes_STGRU.sh`。
- 脚本串联 Cityscapes 关键帧监督训练流程：
  - 将 `gtFine` 的 `labelIds` 转换为 road/free-space 二值 mask。
  - 使用 `leftImg8bit_sequence` 中关键帧及其历史帧构建连续帧输入。
  - 预计算 YOLOP 当前帧 mask、历史帧 mask、SEA-RAFT flow、warped mask、photometric error。
  - 调用 `train_stgru.py` 使用预计算 CSV 训练 STGRU，并输出验证/测试指标。
- 默认训练分辨率为 `960x540`，对齐 SEA-RAFT spring 540x960 权重使用习惯。

### Reason

- STGRU 训练需要同时具备连续帧输入和关键帧监督标签。
- Cityscapes 的 `leftImg8bit_sequence` 提供连续帧，`gtFine` 提供关键帧精细标注，二者可以组成 STGRU 的监督训练样本。

### Result

- `bash -n Run_Cityscapes_STGRU.sh` 通过。
- `precompute_stgru_samples.py`、`train_stgru.py`、`prepare_cityscapes_binary.py` 编译检查通过。

### Known Issues

- 如果某个 split 缺少 `leftImg8bit_sequence`，且 `REQUIRE_SEQUENCE=1`，该 split 会跳过缺失样本。
- Cityscapes 的关键帧标注稀疏，训练监督只发生在有 gtFine 的关键帧位置，不等价于完整视频逐帧监督。

## 2026-06-12

### Modified Module

- BDD100K Dataset
- STGRU Data Preparation

### Changes

- 新增 `src/utils/datasets/download_bdd100k_keyframes.py`。
  - 从 BDD100K `images_100k` 压缩包中按 scene id 下载关键帧 image。
  - 支持远程 zip Range 下载，也支持本地 `bdd100k_images_100k.zip`。
  - 支持从 `scene-list`、BDD STGRU CSV、本地 drivable maps 中收集 scene id。
- 新增 `src/utils/datasets/match_bdd100k_keyframes.py`。
  - 将视频切出的 10s 附近帧与 BDD100K image keyframe 进行批量匹配。
  - 使用灰度差异、HSV 直方图相关性、ORB 特征匹配的加权分数选择最佳帧。
  - 将最佳匹配帧作为关键帧，并复制其前 5 帧作为 STGRU 输入片段。

### Reason

- BDD100K drivable-area 标签对应的是 image keyframe，而视频切帧可能存在时间戳/编码偏移。
- 需要通过图像匹配找到视频帧中最接近官方关键帧的那一帧，避免监督标签和视频帧错位。

### Result

- 两个脚本均通过 Python 编译检查。
- 两个脚本的 CLI help 输出正常。

### Known Issues

- 图像匹配是工程近似方法，若视频画面压缩严重或 keyframe 与视频并非同源，匹配分数可能下降。
- 低分匹配需要人工抽查，可通过 `--min-score` 设置阈值过滤。

## 2026-06-12

### Modified Module

- Environment
- Cloud Setup

### Changes

- 在 `requirements.txt` 中加入 `opencv-python-headless==4.9.0.80`。
- 在 `Setup_Cloud_Env.sh` 中加入 `cv2` 导入兜底检测：
  - 如果 `import cv2` 失败，自动安装 `opencv-python-headless==4.9.0.80`。
  - 同时保持 `numpy>=1.23,<2.0`，避免 OpenCV 新版拉起 NumPy `2.x`。
- 环境检查输出中增加 NumPy 版本。

### Reason

- 云平台容器中可能禁用 `apt-get` 或使用 `INSTALL_APT=0`，导致没有系统版 `python3-opencv`。
- STGRU 数据准备、视频切帧、mask 处理都依赖 OpenCV，缺失 `cv2` 会直接阻塞环境验证和后续训练。

### Result

- `bash -n Setup_Cloud_Env.sh` 通过。
- 后续云端重新运行环境脚本时会自动补齐 `cv2`。

### Known Issues

- 如果云平台禁止联网安装 pip 包，需要手动上传对应 wheel 或更换预装 OpenCV 的镜像。

## 2026-06-12

### Modified Module

- SEA-RAFT
- BDD100K STGRU
- Cityscapes STGRU
- Documentation

### Changes

- 将 `Run_BDD100K_STGRU.sh` 默认 SEA-RAFT 配置从 Hugging Face `spring-S` 改为本地 `spring-M` checkpoint：
  - config：`src/SEA_RAFT/external/SEA-RAFT/config/eval/spring-M.json`
  - checkpoint：`weights/SEA-RAFT/Tartan-C-T-TSKH-spring540x960-M.pth`
  - URL：默认空
- 将 `Run_Cityscapes_STGRU.sh`、`precompute_bdd100k_stgru_samples.py`、`precompute_stgru_samples.py` 同步改为默认本地 checkpoint。
- 在 `sea_raft_wrapper.py` 中为 Hugging Face 加载失败增加更明确的错误提示。
- 更新远程训练文档和 README，说明默认离线使用本地 SEA-RAFT 权重。

### Reason

- 云平台 smoke 阶段通过 Hugging Face 在线加载 SEA-RAFT 时出现 `RuntimeError: Cannot send a request, as the client has been closed`。
- 该问题属于 Hugging Face 下载链路/网络/版本兼容问题，不应阻塞本地已有 checkpoint 的 smoke 验证。

### Result

- BDD100K 和 Cityscapes STGRU 预计算默认不再访问 Hugging Face。
- `bash -n` 和相关 Python 编译检查通过。

### Known Issues

- 如果手动设置 `SEA_RAFT_CHECKPOINT=` 且指定 `SEA_RAFT_URL`，仍会依赖 Hugging Face 网络环境。
