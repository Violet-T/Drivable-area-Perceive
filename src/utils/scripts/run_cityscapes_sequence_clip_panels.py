"""按 Cityscapes sequence 的 30 帧连续片段导出 temporal panel 视频。"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "utils" / "legacy" / "ros2_packages" / "pidnet"))

from pidnet.pidnet_wrapper import PIDNetFreeSpaceWrapper, read_image_bgr  # noqa: E402
from SEA_RAFT.sea_raft_wrapper import SEARAFTWrapper  # noqa: E402
from utils.temporal.fusion import fuse_masks  # noqa: E402
from utils.temporal.mask_buffer import MaskBuffer, combine_history_masks  # noqa: E402
from utils.visualization.overlay import mask_to_bgr, overlay_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PIDNet + SEA-RAFT temporal fusion per Cityscapes 30-frame clip"
    )
    parser.add_argument("--clips-csv", default="/workspace/data/cityscapes/bochum_000000_clips.csv")
    parser.add_argument("--output-dir", default="/workspace/output/bochum_000000_clip_panels")
    parser.add_argument("--clip-start", type=int, default=0)
    parser.add_argument("--clip-end", type=int, default=None)
    parser.add_argument("--max-clips", type=int, default=0, help="0 means all selected clips.")
    parser.add_argument("--fps", type=int, default=17)
    parser.add_argument("--panel-scale", type=float, default=1.0)
    parser.add_argument("--target-height", type=int, default=540)
    parser.add_argument("--target-width", type=int, default=960)
    parser.add_argument("--clip-ids", default="", help="Comma-separated clip ids. Overrides --clip-start/--clip-end.")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--non-free-threshold", type=float, default=0.2)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--history-decay", type=float, default=0.6)
    parser.add_argument(
        "--artifact-alpha",
        type=float,
        default=0.35,
        help="特殊环境触发帧使用的 current mask 权重；设为负数可禁用。",
    )
    parser.add_argument(
        "--artifact-non-free-threshold",
        type=float,
        default=0.05,
        help="特殊环境触发帧使用的 non-free threshold，降低后历史 road 更容易参与恢复。",
    )
    parser.add_argument(
        "--weather-mode",
        default="none",
        choices=[
            "none",
            "glare",
            "fog",
            "rain",
            "snow",
            "mixed",
            "sudden_glare",
            "water_drop_slide",
            "short_glare",
            "snow_occlusion",
        ],
        help="仅在内存中模拟天气/强光干扰，不改写原始 Cityscapes 图像。",
    )
    parser.add_argument("--weather-strength", type=float, default=0.55)
    parser.add_argument("--weather-trigger-frame", type=int, default=10)
    parser.add_argument("--weather-duration", type=int, default=8)
    parser.add_argument("--weather-seed", type=int, default=20260602)
    parser.add_argument("--suite-special-weather", action="store_true")
    parser.add_argument("--suite-scenes-per-env", type=int, default=5)
    parser.add_argument("--suite-seed", type=int, default=20260602)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--pidnet-repo", default="/workspace/src/utils/legacy/external/PIDNet")
    parser.add_argument(
        "--pidnet-checkpoint",
        default="/workspace/src/utils/legacy/external/PIDNet/pretrained_models/cityscapes/PIDNet_S_Cityscapes_test.pt",
    )
    parser.add_argument("--pidnet-arch", default="pidnet-s", choices=["pidnet-s", "pidnet-m", "pidnet-l"])
    parser.add_argument("--free-space-classes", default="road")
    parser.add_argument(
        "--mask-mode",
        default="probability",
        choices=["binary", "probability"],
        help="probability 更适合 temporal fusion；binary 用于观察硬分类输出。",
    )
    parser.add_argument("--sea-raft-repo", default="/workspace/src/SEA_RAFT/external/SEA-RAFT")
    parser.add_argument(
        "--sea-raft-config",
        default="/workspace/src/SEA_RAFT/external/SEA-RAFT/config/eval/spring-S.json",
    )
    parser.add_argument("--sea-raft-checkpoint", default="")
    parser.add_argument("--sea-raft-url", default="MemorySlices/Tartan-C-T-TSKH-spring540x960-S")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_clips(path: Path, clip_start: int, clip_end: int | None, max_clips: int) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open()))
    selected: list[dict[str, str]] = []
    for row in rows:
        clip_id = int(row["clip_id"])
        if clip_id < clip_start:
            continue
        if clip_end is not None and clip_id > clip_end:
            continue
        selected.append(row)
        if max_clips and len(selected) >= max_clips:
            break
    if not selected:
        raise ValueError("No clips selected.")
    return selected


def read_clips_by_ids(path: Path, clip_ids: list[int]) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open()))
    row_by_id = {int(row["clip_id"]): row for row in rows}
    missing = [clip_id for clip_id in clip_ids if clip_id not in row_by_id]
    if missing:
        raise ValueError(f"Unknown clip ids: {missing}")
    return [row_by_id[clip_id] for clip_id in clip_ids]


def parse_clip_ids(raw_clip_ids: str) -> list[int]:
    if not raw_clip_ids.strip():
        return []
    return [int(item.strip()) for item in raw_clip_ids.split(",") if item.strip()]


def frame_paths_for_clip(clip: dict[str, str]) -> list[Path]:
    start_path = Path(clip["start_path"])
    start_frame = int(clip["start_frame"])
    end_frame = int(clip["end_frame"])
    name = start_path.name
    suffix = "_leftImg8bit.png"
    prefix = name.split(f"_{start_frame:06d}{suffix}")[0]
    directory = start_path.parent
    frames = [directory / f"{prefix}_{frame:06d}{suffix}" for frame in range(start_frame, end_frame + 1)]
    missing = [path for path in frames if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frame in clip {clip['clip_id']}: {missing[0]}")
    return frames


def open_writer(path: Path, fps: int, frame_shape: tuple[int, int, int]) -> cv2.VideoWriter:
    height, width = frame_shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def resize_panel(panel: np.ndarray, panel_scale: float) -> np.ndarray:
    if panel_scale <= 0:
        raise ValueError("panel_scale must be positive")
    if abs(panel_scale - 1.0) < 1e-6:
        return panel
    height, width = panel.shape[:2]
    new_size = (max(1, int(round(width * panel_scale))), max(1, int(round(height * panel_scale))))
    return cv2.resize(panel, new_size, interpolation=cv2.INTER_AREA)


def add_glare(image_bgr: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    result = image_bgr.astype(np.float32)
    center_x = int(width * rng.uniform(0.18, 0.82))
    center_y = int(height * rng.uniform(-0.08, 0.30))
    radius_x = int(width * rng.uniform(0.28, 0.48))
    radius_y = int(height * rng.uniform(0.20, 0.38))
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    dist2 = ((grid_x - center_x) / max(1, radius_x)) ** 2 + ((grid_y - center_y) / max(1, radius_y)) ** 2
    glare = np.exp(-dist2 * 1.35).astype(np.float32)
    veil = np.clip(glare * (0.62 + 0.33 * strength), 0.0, 0.95)
    white = np.full_like(result, 255.0)
    result = result * (1.0 - veil[..., None]) + white * veil[..., None]
    result += glare[..., None] * (255.0 * strength * 0.72)
    return np.clip(result, 0, 255).astype(np.uint8)


def add_fog(image_bgr: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    base = image_bgr.astype(np.float32)
    vertical = np.linspace(0.35, 1.0, height, dtype=np.float32)[:, None]
    noise = rng.normal(0.0, 0.035, (height, width)).astype(np.float32)
    fog_alpha = np.clip((vertical + noise) * strength, 0.0, 0.75)
    fog_color = np.array([225, 225, 225], dtype=np.float32)
    result = base * (1.0 - fog_alpha[..., None]) + fog_color * fog_alpha[..., None]
    return np.clip(result, 0, 255).astype(np.uint8)


def add_rain(image_bgr: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    result = (image_bgr.astype(np.float32) * (1.0 - 0.18 * strength)).astype(np.uint8)
    height, width = result.shape[:2]
    layer = np.zeros_like(result, dtype=np.uint8)
    drops = int(width * height * 0.00008 * (0.4 + strength))
    length = int(18 + 28 * strength)
    for _ in range(drops):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        cv2.line(layer, (x, y), (min(width - 1, x + length // 3), min(height - 1, y + length)), (210, 210, 210), 1)
    layer = cv2.GaussianBlur(layer, (3, 3), 0)
    return cv2.addWeighted(result, 1.0, layer, min(0.85, 0.65 + 0.25 * strength), 0)


def add_snow(image_bgr: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    result = image_bgr.astype(np.float32)
    height, width = image_bgr.shape[:2]
    flakes = rng.random((height, width))
    threshold = 1.0 - 0.008 * (0.5 + strength)
    snow = (flakes > threshold).astype(np.float32)
    snow = cv2.GaussianBlur(snow, (5, 5), 0)
    result = result * (1.0 - 0.10 * strength) + snow[..., None] * 255.0 * (0.8 + 0.5 * strength)
    return np.clip(result, 0, 255).astype(np.uint8)


def add_water_drop_slide(
    image_bgr: np.ndarray,
    strength: float,
    clip_id: int,
    frame_index: int,
    total_frames: int,
    seed: int,
) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    rng = np.random.default_rng(seed + clip_id * 1_003)
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = grid_x.copy()
    map_y = grid_y.copy()
    alpha_total = np.zeros((height, width), dtype=np.float32)
    trail_total = np.zeros((height, width), dtype=np.float32)
    drop_count = int(rng.integers(2, 9))
    for drop_idx in range(drop_count):
        local_rng = np.random.default_rng(seed + clip_id * 12_983 + drop_idx * 977)
        center_x = int(width * local_rng.uniform(0.12, 0.88))
        radius_x = int(width * local_rng.uniform(0.025, 0.055))
        radius_y = int(height * local_rng.uniform(0.050, 0.095))
        start_offset = local_rng.uniform(-0.35, 0.20)
        speed = local_rng.uniform(1.35, 2.10)
        progress = np.clip(frame_index / max(1, total_frames - 1) * speed + start_offset, -0.35, 1.35)
        center_y = int(-radius_y + progress * (height + 2 * radius_y))
        norm = ((grid_x - center_x) / max(1, radius_x)) ** 2 + ((grid_y - center_y) / max(1, radius_y)) ** 2
        alpha = np.clip(1.0 - norm, 0.0, 1.0) ** 0.32
        alpha *= 0.84 + 0.16 * strength
        alpha_total = np.maximum(alpha_total, alpha)
        map_x += (grid_x - center_x) * alpha * 0.080 * strength
        map_y += (grid_y - center_y) * alpha * 0.125 * strength
        trail = np.zeros((height, width), dtype=np.float32)
        if -radius_y <= center_y <= height + radius_y:
            trail_start = max(0, center_y - int(radius_y * 2.8))
            trail_end = min(height - 1, center_y)
            cv2.line(trail, (center_x, trail_start), (center_x, trail_end), 1.0, max(2, radius_x // 5))
            trail = cv2.GaussianBlur(trail, (0, 0), sigmaX=max(2, radius_x / 3), sigmaY=max(4, radius_y / 2))
            trail_total = np.maximum(trail_total, trail)
    distorted = cv2.remap(
        image_bgr,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    blur = cv2.GaussianBlur(distorted, (0, 0), sigmaX=2.4)
    drop_color = np.full_like(image_bgr, 218)
    mixed = cv2.addWeighted(blur, 0.76, drop_color, 0.24, 0)
    opacity = np.clip(alpha_total * 0.92, 0.0, 0.96)
    result = image_bgr.astype(np.float32) * (1.0 - opacity[..., None]) + mixed.astype(np.float32) * opacity[..., None]
    result = result * (1.0 - trail_total[..., None] * 0.42 * strength) + 232.0 * trail_total[..., None] * 0.42 * strength
    return np.clip(result, 0, 255).astype(np.uint8)


def add_snow_occlusion(
    image_bgr: np.ndarray,
    strength: float,
    clip_id: int,
    frame_index: int,
    trigger_frame: int,
    duration: int,
    seed: int,
) -> tuple[np.ndarray, bool]:
    rng = np.random.default_rng(seed + clip_id * 7_919)
    actual_duration = int(rng.integers(1, max(2, min(3, duration) + 1)))
    if frame_index < trigger_frame or frame_index >= trigger_frame + actual_duration:
        return image_bgr, False
    height, width = image_bgr.shape[:2]
    local_index = frame_index - trigger_frame
    start_x = width * rng.uniform(0.15, 0.78)
    start_y = height * rng.uniform(0.06, 0.45)
    drift_x = width * rng.uniform(-0.10, 0.12)
    drift_y = height * rng.uniform(0.02, 0.14)
    center_x = int(start_x + drift_x * local_index)
    center_y = int(start_y + drift_y * local_index)
    radius_x = int(width * rng.uniform(0.030, 0.070))
    radius_y = int(height * rng.uniform(0.035, 0.085))
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    dist2 = ((grid_x - center_x) / max(1, radius_x)) ** 2 + ((grid_y - center_y) / max(1, radius_y)) ** 2
    alpha = np.exp(-dist2 * 1.8).astype(np.float32)
    speckles = rng.random((height, width)).astype(np.float32)
    speckles = cv2.GaussianBlur((speckles > 0.985).astype(np.float32), (3, 3), 0)
    alpha = np.clip(alpha * (0.72 + 0.25 * strength) + speckles * alpha * 0.28, 0.0, 0.88)
    snow_color = np.array([245, 245, 245], dtype=np.float32)
    blur = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=max(2.0, radius_x / 8.0)).astype(np.float32)
    result = blur * (1.0 - alpha[..., None]) + snow_color * alpha[..., None]
    return np.clip(result, 0, 255).astype(np.uint8), True


def short_event_window(
    clip_id: int,
    total_frames: int,
    seed: int,
    min_duration: int,
    max_duration: int,
) -> tuple[int, int]:
    rng = np.random.default_rng(seed + clip_id * 15_485)
    duration = int(rng.integers(min_duration, max_duration + 1))
    max_start = max(0, total_frames - duration - 1)
    trigger = int(rng.integers(1, max_start + 1)) if max_start > 1 else 0
    return trigger, duration


def apply_weather(
    image_bgr: np.ndarray,
    mode: str,
    strength: float,
    clip_id: int,
    frame_index: int,
    trigger_frame: int,
    duration: int,
    total_frames: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    if mode == "none":
        return image_bgr, "none"
    rng = np.random.default_rng(seed + clip_id * 100_003 + frame_index * 97)
    strength = float(np.clip(strength, 0.0, 1.0))
    active_mode = mode
    if mode == "mixed":
        active_mode = ["glare", "fog", "rain", "snow"][(clip_id + frame_index // 8) % 4]
    if mode == "water_drop_slide":
        return add_water_drop_slide(image_bgr, strength, clip_id, frame_index, total_frames, seed), active_mode
    if mode == "sudden_glare":
        # 突发强光只在指定时间窗口内作用，用于观察短时扰动下 warp mask 的对齐效果。
        if frame_index < trigger_frame or frame_index >= trigger_frame + max(1, duration):
            return image_bgr, "none"
        active_mode = "glare"
    if mode == "short_glare":
        trigger_frame, duration = short_event_window(clip_id, total_frames, seed, 1, 5)
        if frame_index < trigger_frame or frame_index >= trigger_frame + duration:
            return image_bgr, "none"
        active_mode = "glare"
    if mode == "snow_occlusion":
        trigger_frame, duration = short_event_window(clip_id, total_frames, seed, 1, 3)
        occluded, active = add_snow_occlusion(
            image_bgr, strength, clip_id, frame_index, trigger_frame, duration, seed
        )
        return occluded, "snow_occlusion" if active else "none"
    if active_mode == "glare":
        return add_glare(image_bgr, strength, rng), active_mode
    if active_mode == "fog":
        return add_fog(image_bgr, strength, rng), active_mode
    if active_mode == "rain":
        return add_rain(image_bgr, strength, rng), active_mode
    if active_mode == "snow":
        return add_snow(image_bgr, strength, rng), active_mode
    raise ValueError(f"Unsupported weather mode: {mode}")


def resize_to_target(image_bgr: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    if target_width <= 0 or target_height <= 0:
        return image_bgr
    if image_bgr.shape[1] == target_width and image_bgr.shape[0] == target_height:
        return image_bgr
    return cv2.resize(image_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)


def resize_mask_to_target(mask: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    if mask.shape == (target_height, target_width):
        return mask.astype(np.float32)
    resized = cv2.resize(mask.astype(np.float32), (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def infer_pidnet_mask_at_target(pidnet: PIDNetFreeSpaceWrapper, image_bgr: np.ndarray) -> np.ndarray:
    # PIDNet 多尺度分支需要更规整的空间尺寸；先补齐到 8 的倍数，再裁回目标 HxW。
    height, width = image_bgr.shape[:2]
    padded_height = int(np.ceil(height / 8.0) * 8)
    padded_width = int(np.ceil(width / 8.0) * 8)
    pad_bottom = padded_height - height
    pad_right = padded_width - width
    if pad_bottom or pad_right:
        padded = cv2.copyMakeBorder(
            image_bgr,
            0,
            pad_bottom,
            0,
            pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )
    else:
        padded = image_bgr
    mask = pidnet.infer_free_space(padded)
    mask = mask[:height, :width]
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def comparison_mask_bgr(current_mask: np.ndarray, fused_mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    current = current_mask >= threshold
    fused = fused_mask >= threshold
    comparison = np.zeros((*current.shape, 3), dtype=np.uint8)
    comparison[np.logical_and(current, fused)] = (0, 220, 0)      # 交集：绿色
    comparison[np.logical_and(current, ~fused)] = (255, 80, 0)   # 仅当前帧：蓝色
    comparison[np.logical_and(~current, fused)] = (0, 0, 255)    # 仅融合结果：红色
    return comparison


def make_fusion_panel(image_bgr: np.ndarray, raw_mask: np.ndarray, fused_mask: np.ndarray) -> np.ndarray:
    # 布局：左上 当前图像+当前mask；左下 当前mask；右上 fused mask；右下 当前mask与fused mask差异。
    current_overlay = overlay_mask(image_bgr, raw_mask)
    current_mask_bgr = mask_to_bgr(raw_mask)
    fused_mask_bgr = mask_to_bgr(fused_mask)
    comparison_bgr = comparison_mask_bgr(raw_mask, fused_mask)
    top = cv2.hconcat([current_overlay, fused_mask_bgr])
    bottom = cv2.hconcat([current_mask_bgr, comparison_bgr])
    return cv2.vconcat([top, bottom])


def binary_iou(mask_a: np.ndarray, mask_b: np.ndarray, threshold: float = 0.5) -> float:
    a = mask_a >= threshold
    b = mask_b >= threshold
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def select_suite_clips(
    clips_csv: Path,
    scenes_per_env: int,
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    rows = list(csv.DictReader(clips_csv.open()))
    if scenes_per_env <= 0:
        raise ValueError("scenes_per_env must be positive")
    environments = ["water_drop_slide", "short_glare", "snow_occlusion"]
    required = scenes_per_env * len(environments)
    if len(rows) < required:
        raise ValueError(f"Need at least {required} clips for non-overlapping suite selection.")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(rows), size=required, replace=False)
    selected: dict[str, list[dict[str, str]]] = {}
    cursor = 0
    for env in environments:
        env_indices = sorted(indices[cursor : cursor + scenes_per_env], key=lambda idx: int(rows[idx]["clip_id"]))
        selected[env] = [rows[int(idx)] for idx in env_indices]
        cursor += scenes_per_env
    return selected


def encode_h264(input_path: Path, output_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def run_clip(
    clip: dict[str, str],
    output_root: Path,
    fps: int,
    panel_scale: float,
    target_width: int,
    target_height: int,
    pidnet: PIDNetFreeSpaceWrapper,
    sea_raft: SEARAFTWrapper,
    overwrite: bool,
    weather_mode: str,
    weather_strength: float,
    weather_trigger_frame: int,
    weather_duration: int,
    weather_seed: int,
    alpha: float,
    non_free_threshold: float,
    history_size: int,
    history_decay: float,
    artifact_alpha: float,
    artifact_non_free_threshold: float,
) -> Path:
    clip_id = int(clip["clip_id"])
    clip_dir = output_root / f"clip_{clip_id:03d}_{int(clip['start_frame']):06d}_{int(clip['end_frame']):06d}"
    video_path = clip_dir / "temporal_panel_h264.mp4"
    metrics_path = clip_dir / "performance_matrix.csv"
    if video_path.exists() and metrics_path.exists() and not overwrite:
        return video_path
    clip_dir.mkdir(parents=True, exist_ok=True)

    temp_video_path = clip_dir / "_temporal_panel_mp4v.mp4"
    frames = frame_paths_for_clip(clip)

    writer = None
    previous_bgr: np.ndarray | None = None
    mask_buffer = MaskBuffer(max_history=history_size)
    metrics_rows: list[list[object]] = []

    clip_start_time = time.perf_counter()
    for frame_index, frame_path in enumerate(tqdm(frames, desc=f"clip {clip_id:03d}", leave=False)):
        original_bgr = read_image_bgr(frame_path)
        resized_bgr = resize_to_target(original_bgr, target_width=target_width, target_height=target_height)
        image_bgr, active_weather = apply_weather(
            resized_bgr,
            mode=weather_mode,
            strength=weather_strength,
            clip_id=clip_id,
            frame_index=frame_index,
            trigger_frame=weather_trigger_frame,
            duration=weather_duration,
            total_frames=len(frames),
            seed=weather_seed,
        )
        # raw_mask/fused_mask/warped_mask 均为 float32 HxW, value range [0, 1]。
        t0 = time.perf_counter()
        raw_mask = infer_pidnet_mask_at_target(pidnet, image_bgr)
        raw_mask = resize_mask_to_target(raw_mask, target_width=target_width, target_height=target_height)
        pidnet_time = time.perf_counter() - t0

        flow_time = 0.0
        warp_time = 0.0
        fusion_time = 0.0
        if previous_bgr is None or len(mask_buffer) == 0:
            history_mask = raw_mask.copy()
            fused_mask = raw_mask.copy()
            effective_alpha = alpha
            effective_non_free_threshold = non_free_threshold
        else:
            t1 = time.perf_counter()
            flow = sea_raft.infer_flow(previous_bgr, image_bgr)
            flow_time = time.perf_counter() - t1
            t2 = time.perf_counter()
            mask_buffer.warp_all(flow)
            history_mask = combine_history_masks(mask_buffer.masks, decay=history_decay)
            if history_mask is None:
                history_mask = raw_mask.copy()
            warp_time = time.perf_counter() - t2
            t_fusion = time.perf_counter()
            effective_alpha = alpha
            effective_non_free_threshold = non_free_threshold
            if active_weather != "none":
                if artifact_alpha >= 0.0:
                    effective_alpha = artifact_alpha
                effective_non_free_threshold = artifact_non_free_threshold
            fused_mask = fuse_masks(
                raw_mask,
                history_mask,
                alpha=effective_alpha,
                non_free_threshold=effective_non_free_threshold,
            )
            fusion_time = time.perf_counter() - t_fusion

        t3 = time.perf_counter()
        panel = make_fusion_panel(image_bgr, raw_mask, fused_mask)
        panel = resize_panel(panel, panel_scale)
        if writer is None:
            writer = open_writer(temp_video_path, fps, panel.shape)
        writer.write(panel)
        panel_time = time.perf_counter() - t3

        metrics_rows.append(
            [
                frame_index,
                frame_path.name,
                active_weather,
                pidnet_time,
                flow_time,
                warp_time,
                fusion_time,
                panel_time,
                pidnet_time + flow_time + warp_time + fusion_time + panel_time,
                binary_iou(raw_mask, history_mask) if frame_index > 0 else "",
                float(np.mean(np.abs(raw_mask - history_mask))) if frame_index > 0 else "",
                binary_iou(fused_mask, history_mask) if frame_index > 0 else "",
                float(np.mean(np.abs(fused_mask - history_mask))) if frame_index > 0 else "",
                binary_iou(raw_mask, fused_mask),
                float(np.mean(np.abs(raw_mask - fused_mask))),
                float(raw_mask.mean()),
                float(history_mask.mean()),
                float(fused_mask.mean()),
                len(mask_buffer),
                effective_alpha,
                effective_non_free_threshold,
            ]
        )
        previous_bgr = image_bgr
        mask_buffer.append(fused_mask)

    if writer is not None:
        writer.release()
    encode_start = time.perf_counter()
    encode_h264(temp_video_path, video_path)
    encode_time = time.perf_counter() - encode_start
    temp_video_path.unlink(missing_ok=True)

    with metrics_path.open("w", newline="") as fh:
        writer_csv = csv.writer(fh)
        writer_csv.writerow(
            [
                "frame_index",
                "frame_name",
                "weather_mode",
                "pidnet_time_s",
                "sea_raft_time_s",
                "warp_time_s",
                "fusion_time_s",
                "panel_write_time_s",
                "frame_total_time_s",
                "temporal_iou_raw_vs_history",
                "frame_difference_raw_vs_history",
                "temporal_iou_fused_vs_history",
                "frame_difference_fused_vs_history",
                "iou_raw_vs_fused",
                "frame_difference_raw_vs_fused",
                "raw_mask_mean",
                "history_mask_mean",
                "fused_mask_mean",
                "history_count",
                "effective_alpha",
                "effective_non_free_threshold",
            ]
        )
        writer_csv.writerows(metrics_rows)
        frame_totals = np.array([row[8] for row in metrics_rows], dtype=np.float32)
        raw_warp_ious = [row[9] for row in metrics_rows if row[9] != ""]
        raw_warp_diffs = [row[10] for row in metrics_rows if row[10] != ""]
        fused_warp_ious = [row[11] for row in metrics_rows if row[11] != ""]
        fused_warp_diffs = [row[12] for row in metrics_rows if row[12] != ""]
        raw_fused_ious = [row[13] for row in metrics_rows if row[13] != ""]
        raw_fused_diffs = [row[14] for row in metrics_rows if row[14] != ""]
        writer_csv.writerow([])
        writer_csv.writerow(["summary_key", "summary_value"])
        writer_csv.writerow(["clip_id", clip_id])
        writer_csv.writerow(["num_frames", len(metrics_rows)])
        writer_csv.writerow(["target_height", target_height])
        writer_csv.writerow(["target_width", target_width])
        writer_csv.writerow(["weather_mode_request", weather_mode])
        writer_csv.writerow(["weather_strength", weather_strength])
        writer_csv.writerow(["weather_trigger_frame", weather_trigger_frame])
        writer_csv.writerow(["weather_duration", weather_duration])
        writer_csv.writerow(["weather_seed", weather_seed])
        writer_csv.writerow(["alpha", alpha])
        writer_csv.writerow(["non_free_threshold", non_free_threshold])
        writer_csv.writerow(["history_size", history_size])
        writer_csv.writerow(["history_decay", history_decay])
        writer_csv.writerow(["artifact_alpha", artifact_alpha])
        writer_csv.writerow(["artifact_non_free_threshold", artifact_non_free_threshold])
        writer_csv.writerow(["mean_frame_total_time_s", float(frame_totals.mean())])
        writer_csv.writerow(["total_clip_compute_time_s", float(time.perf_counter() - clip_start_time)])
        writer_csv.writerow(["h264_encode_time_s", float(encode_time)])
        writer_csv.writerow(
            ["mean_temporal_iou_raw_vs_history", float(np.mean(raw_warp_ious)) if raw_warp_ious else ""]
        )
        writer_csv.writerow(
            ["mean_frame_difference_raw_vs_history", float(np.mean(raw_warp_diffs)) if raw_warp_diffs else ""]
        )
        writer_csv.writerow(
            ["mean_temporal_iou_fused_vs_history", float(np.mean(fused_warp_ious)) if fused_warp_ious else ""]
        )
        writer_csv.writerow(
            ["mean_frame_difference_fused_vs_history", float(np.mean(fused_warp_diffs)) if fused_warp_diffs else ""]
        )
        writer_csv.writerow(["mean_iou_raw_vs_fused", float(np.mean(raw_fused_ious)) if raw_fused_ious else ""])
        writer_csv.writerow(
            ["mean_frame_difference_raw_vs_fused", float(np.mean(raw_fused_diffs)) if raw_fused_diffs else ""]
        )
    return video_path


def main() -> None:
    args = parse_args()
    clips_csv = Path(args.clips_csv)
    if not clips_csv.exists():
        raise FileNotFoundError(clips_csv)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"output_dir: {output_root}")

    pidnet = PIDNetFreeSpaceWrapper(
        pidnet_repo=args.pidnet_repo,
        checkpoint_path=args.pidnet_checkpoint,
        arch=args.pidnet_arch,
        free_space_classes=args.free_space_classes,
        mask_mode=args.mask_mode,
        device=args.device,
    )
    sea_raft = SEARAFTWrapper(
        sea_raft_repo=args.sea_raft_repo,
        config_path=args.sea_raft_config,
        checkpoint_path=args.sea_raft_checkpoint or None,
        model_url=args.sea_raft_url if not args.sea_raft_checkpoint else None,
        device=args.device,
    )

    summary_path = output_root / "summary.csv"
    if args.suite_special_weather:
        suite = select_suite_clips(clips_csv, args.suite_scenes_per_env, args.suite_seed)
        print(
            "suite_selection: "
            + "; ".join(f"{env}={[int(clip['clip_id']) for clip in clips]}" for env, clips in suite.items())
        )
        with summary_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["environment", "clip_id", "start_frame", "end_frame", "num_frames", "video_path"])
            for environment, clips in suite.items():
                env_root = output_root / environment
                for clip in tqdm(clips, desc=environment):
                    video_path = run_clip(
                        clip=clip,
                        output_root=env_root,
                        fps=args.fps,
                        panel_scale=args.panel_scale,
                        target_width=args.target_width,
                        target_height=args.target_height,
                        pidnet=pidnet,
                        sea_raft=sea_raft,
                        overwrite=args.overwrite,
                        weather_mode=environment,
                        weather_strength=args.weather_strength,
                        weather_trigger_frame=args.weather_trigger_frame,
                        weather_duration=args.weather_duration,
                        weather_seed=args.weather_seed,
                        alpha=args.alpha,
                        non_free_threshold=args.non_free_threshold,
                        history_size=args.history_size,
                        history_decay=args.history_decay,
                        artifact_alpha=args.artifact_alpha,
                        artifact_non_free_threshold=args.artifact_non_free_threshold,
                    )
                    writer.writerow(
                        [
                            environment,
                            clip["clip_id"],
                            clip["start_frame"],
                            clip["end_frame"],
                            clip["num_frames"],
                            video_path,
                        ]
                    )
                    fh.flush()
        print(f"summary: {summary_path}")
        return

    clip_ids = parse_clip_ids(args.clip_ids)
    if clip_ids:
        clips = read_clips_by_ids(clips_csv, clip_ids)
    else:
        clips = read_clips(clips_csv, args.clip_start, args.clip_end, args.max_clips)
    print(f"selected_clips: {len(clips)}")
    with summary_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["environment", "clip_id", "start_frame", "end_frame", "num_frames", "video_path"])
        for clip in tqdm(clips, desc="clips"):
            video_path = run_clip(
                clip=clip,
                output_root=output_root,
                fps=args.fps,
                panel_scale=args.panel_scale,
                target_width=args.target_width,
                target_height=args.target_height,
                pidnet=pidnet,
                sea_raft=sea_raft,
                overwrite=args.overwrite,
                weather_mode=args.weather_mode,
                weather_strength=args.weather_strength,
                weather_trigger_frame=args.weather_trigger_frame,
                weather_duration=args.weather_duration,
                weather_seed=args.weather_seed,
                alpha=args.alpha,
                non_free_threshold=args.non_free_threshold,
                history_size=args.history_size,
                history_decay=args.history_decay,
                artifact_alpha=args.artifact_alpha,
                artifact_non_free_threshold=args.artifact_non_free_threshold,
            )
            writer.writerow(
                [
                    args.weather_mode,
                    clip["clip_id"],
                    clip["start_frame"],
                    clip["end_frame"],
                    clip["num_frames"],
                    video_path,
                ]
            )
            fh.flush()
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
