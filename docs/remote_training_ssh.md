# SSH 远程服务器部署与 BDD100K STGRU 训练流程

本文档用于把当前项目推送到 GitHub 后，在远程 GPU 服务器上通过 SSH 完成部署、下载 BDD100K video scenes、预计算 YOLOP + SEA-RAFT 输入，并训练 STGRU。

## 1. 本地提交与推送

在本地项目根目录执行：

```bash
git status --short
git add .gitignore .dockerignore README.md docs/remote_training_ssh.md docs/dev_log.md \
  Run_BDD100K_STGRU.sh download_bdd100k_video_scenes.py \
  src/SEA_RAFT/sea_raft_wrapper.py \
  src/STGRU/precompute_bdd100k_stgru_samples.py \
  weights/YOLOP/End-to-end.pth \
  weights/SEA-RAFT/Tartan-C-T-TSKH-spring540x960-M.pth \
  weights/YOLOP/.gitkeep weights/SEA-RAFT/.gitkeep weights/STGRU/.gitkeep
git commit -m "prepare remote bdd100k stgru training"
git push origin master
```

本项目按当前工程要求将 YOLOP 和 SEA-RAFT 的必要 `.pth` 权重提交到 Git。不要提交：

```text
data/
output/
weights/STGRU_BDD100K/
weights/STGRU_smoke/
weights/STGRU_precompute_smoke/
src/YOLOP/external/YOLOP/weights/
src/SEA_RAFT/external/SEA-RAFT/models/
```

## 2. 远程服务器基础检查

SSH 登录服务器：

```bash
ssh <user>@<server_ip>
```

检查 GPU 和 Docker：

```bash
nvidia-smi
docker --version
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
```

如果最后一条失败，先安装 NVIDIA Container Toolkit，并重启 Docker。

## 3. 远程 clone 项目

推荐使用 SSH URL：

```bash
mkdir -p ~/workspace
cd ~/workspace
git clone git@github.com:Violet-T/Drivable-area-Perceive.git perceive
cd perceive
```

如果服务器没有配置 GitHub SSH key，可以临时用 HTTPS：

```bash
git clone https://github.com/Violet-T/Drivable-area-Perceive.git perceive
```

## 4. 构建 Docker 镜像

```bash
docker build -t freespace_temporal:latest .
```

进入容器：

```bash
docker run --gpus all -it --rm \
  --name perceive \
  -v "$PWD":/workspace \
  -w /workspace \
  freespace_temporal:latest
```

容器内检查：

```bash
nvidia-smi
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 5. 准备权重与标签数据

YOLOP 和 SEA-RAFT 的必要权重随 Git 仓库下发，默认路径：

```text
weights/YOLOP/End-to-end.pth
weights/SEA-RAFT/Tartan-C-T-TSKH-spring540x960-M.pth
```

BDD100K drivable maps 建议放在：

```text
data/bdd100k_drivable_maps/
├── labels/train/*_drivable_id.png
├── labels/val/*_drivable_id.png
├── color_labels/train/*_drivable_color.png
└── color_labels/val/*_drivable_color.png
```

可以从本地同步到服务器：

```bash
rsync -avP data/bdd100k_drivable_maps <user>@<server_ip>:~/workspace/perceive/data/
```

SEA-RAFT 默认仍使用 Hugging Face 的 `spring-S` 模型，便于保持当前 smoke 结果一致：

```text
MemorySlices/Tartan-C-T-TSKH-spring540x960-S
```

如遇到 Hugging Face 限速，可在容器内设置：

```bash
export HF_TOKEN=<your_huggingface_token>
```

如果要强制使用 Git 中的本地 SEA-RAFT `spring-M` 权重，运行时改为：

```bash
SEA_RAFT_CONFIG=src/SEA_RAFT/external/SEA-RAFT/config/eval/spring-M.json \
SEA_RAFT_CHECKPOINT=weights/SEA-RAFT/Tartan-C-T-TSKH-spring540x960-M.pth \
SEA_RAFT_URL= \
./Run_BDD100K_STGRU.sh precompute
```

## 6. 小规模 smoke 训练

先不要直接跑 100 个 scene。容器内执行：

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

成功后应生成：

```text
data/bdd100k_video_scenes/
data/bdd100k_stgru/
data/stgru_samples_bdd100k/
weights/STGRU_BDD100K/stgru_best.pth
weights/STGRU_BDD100K/stgru_latest.pth
weights/STGRU_BDD100K/training_log.csv
weights/STGRU_BDD100K/evaluation_summary.csv
```

如果下载已经完成但训练中断，可以分阶段继续：

```bash
./Run_BDD100K_STGRU.sh precompute
EPOCHS=1 BATCH_SIZE=2 ./Run_BDD100K_STGRU.sh train
```

## 7. 正式 100 scene 训练

smoke 通过后再执行：

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

默认输入尺寸：

```text
IMAGE_WIDTH=960
IMAGE_HEIGHT=540
```

该尺寸与当前 SEA-RAFT `spring-S` 配置一致：

```text
image_size: [540, 960]
```

## 8. 常见问题

### 远程 label zip 超时

默认已经跳过远程 label 分层解析：

```bash
SKIP_REMOTE_LABEL_SELECTION=1
```

脚本会使用本地 `data/bdd100k_drivable_maps` 随机选择有监督标签的 scene。

### 想只使用 direct drivable

默认 `1,2` 都作为 free-space：

```bash
BDD_DRIVABLE_VALUES=1,2
```

只使用 direct drivable：

```bash
BDD_DRIVABLE_VALUES=1 ./Run_BDD100K_STGRU.sh all
```

### 权限变成 root

如果容器输出文件在宿主机上变成 root 所有：

```bash
docker exec perceive chown -R $(id -u):$(id -g) /workspace/data /workspace/output /workspace/weights
```

或者容器退出后在服务器上执行：

```bash
sudo chown -R "$USER:$USER" data output weights
```
