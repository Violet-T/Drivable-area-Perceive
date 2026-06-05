"""Run YOLOP + SEA-RAFT temporal free-space fusion on a video."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from SEA_RAFT.sea_raft_wrapper import SEARAFTWrapper  # noqa: E402
from utils.temporal.fusion import fuse_masks  # noqa: E402
from utils.temporal.mask_buffer import MaskBuffer, combine_history_masks  # noqa: E402
from utils.temporal.warp import warp_mask_with_flow  # noqa: E402
from utils.visualization.overlay import make_temporal_panel, mask_to_bgr, overlay_mask  # noqa: E402
from YOLOP.yolop_wrapper import YOLOPFreeSpaceWrapper  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOP + SEA-RAFT temporal free-space fusion")
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output-dir", default="/workspace/output/yolop_temporal_fusion")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames.")
    parser.add_argument("--fps", type=float, default=0.0, help="0 uses input video FPS.")
    parser.add_argument("--target-width", type=int, default=0, help="Resize frames before inference; 0 keeps input width.")
    parser.add_argument("--target-height", type=int, default=0, help="Resize frames before inference; 0 keeps input height.")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--non-free-threshold", type=float, default=0.2)
    parser.add_argument("--history-size", type=int, default=1)
    parser.add_argument("--history-decay", type=float, default=0.6)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--yolop-repo", default="/workspace/src/YOLOP/external/YOLOP")
    parser.add_argument("--yolop-checkpoint", default="/workspace/weights/YOLOP/End-to-end.pth")
    parser.add_argument("--yolop-img-size", type=int, default=640)
    parser.add_argument("--mask-mode", default="probability", choices=["probability", "binary"])
    parser.add_argument("--sea-raft-repo", default="/workspace/src/SEA_RAFT/external/SEA-RAFT")
    parser.add_argument("--sea-raft-config", default="/workspace/src/SEA_RAFT/external/SEA-RAFT/config/eval/spring-S.json")
    parser.add_argument("--sea-raft-checkpoint", default="")
    parser.add_argument("--sea-raft-url", default="MemorySlices/Tartan-C-T-TSKH-spring540x960-S")
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--save-frames", action="store_true", help="Save 30FPS raw/fused overlay and mask PNG frames.")
    parser.add_argument("--no-video", action="store_true", help="Do not export MP4 videos.")
    parser.add_argument(
        "--vis-threshold",
        type=float,
        default=0.0,
        help="Suppress visualization mask values below this threshold; does not change fusion arrays.",
    )
    parser.add_argument(
        "--vis-binary",
        action="store_true",
        help="Render visualization masks as binary after vis-threshold; does not change fusion arrays.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def open_writer(path: Path, fps: float, frame_shape: tuple[int, int, int]) -> cv2.VideoWriter:
    height, width = frame_shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def convert_to_h264(src: Path, dst: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def resize_frame_if_needed(image_bgr: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Return BGR frame resized to target HxW when both target dimensions are set."""
    if target_width <= 0 and target_height <= 0:
        return image_bgr
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target-width and target-height must be set together.")
    if image_bgr.shape[1] == target_width and image_bgr.shape[0] == target_height:
        return image_bgr
    return cv2.resize(image_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)


def mask_for_visualization(mask: np.ndarray, threshold: float, binary: bool) -> np.ndarray:
    """Prepare a display-only HxW mask while keeping model/fusion outputs unchanged."""
    visual = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if threshold > 0.0:
        visual = np.where(visual >= threshold, visual, 0.0).astype(np.float32)
    if binary:
        visual = (visual >= max(threshold, 0.5)).astype(np.float32)
    return visual


def main() -> None:
    args = parse_args()
    if args.history_size < 1:
        raise ValueError("history-size must be >= 1")
    input_video = Path(args.input_video)
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    if args.max_frames:
        total_frames = min(total_frames, args.max_frames) if total_frames else args.max_frames

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite to replace outputs.")
    video_dir = output_dir / "videos"
    raw_dir = output_dir / "raw_masks"
    warped_dir = output_dir / "warped_masks"
    fused_dir = output_dir / "fused_masks"
    frame_dirs = {
        "raw_overlay": output_dir / "raw_overlay_frames_30fps",
        "raw_mask": output_dir / "raw_mask_frames_30fps",
        "fused_overlay": output_dir / "fused_overlay_frames_30fps",
        "fused_mask": output_dir / "fused_mask_frames_30fps",
    }
    directories = [raw_dir, warped_dir, fused_dir]
    if args.save_frames:
        directories.extend(frame_dirs.values())
    if not args.no_video:
        directories.append(video_dir)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    yolop = YOLOPFreeSpaceWrapper(
        yolop_repo=args.yolop_repo,
        checkpoint_path=args.yolop_checkpoint,
        img_size=args.yolop_img_size,
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

    temp_raw_video = video_dir / "raw_overlay_mp4v.mp4"
    temp_fused_video = video_dir / "fused_overlay_mp4v.mp4"
    temp_fused_mask_video = video_dir / "fused_mask_mp4v.mp4"
    temp_panel_video = video_dir / "temporal_panel_mp4v.mp4"
    raw_video = video_dir / "raw_overlay_h264.mp4"
    fused_video = video_dir / "fused_overlay_h264.mp4"
    fused_mask_video = video_dir / "fused_mask_h264.mp4"
    panel_video = video_dir / "temporal_panel_h264.mp4"
    raw_writer = fused_writer = fused_mask_writer = panel_writer = None
    manifest_path = output_dir / "temporal_manifest.csv"

    previous_bgr: np.ndarray | None = None
    mask_buffer = MaskBuffer(max_history=args.history_size)
    frame_id = 0
    with manifest_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame_id", "raw_mask_path", "warped_mask_path", "fused_mask_path"])
        progress = tqdm(total=total_frames if total_frames else None, desc="YOLOP temporal fusion")
        while True:
            ok, image_bgr = cap.read()
            if not ok:
                break
            if args.max_frames and frame_id >= args.max_frames:
                break
            image_bgr = resize_frame_if_needed(image_bgr, args.target_width, args.target_height)

            raw_mask = yolop.infer_drivable_mask(image_bgr)
            if previous_bgr is None or len(mask_buffer) == 0:
                warped_mask = np.zeros_like(raw_mask, dtype=np.float32)
                fused_mask = raw_mask
            else:
                flow = sea_raft.infer_flow(previous_bgr, image_bgr)
                mask_buffer.warp_all(flow)
                warped_mask = combine_history_masks(mask_buffer.masks, decay=args.history_decay)
                if warped_mask is None:
                    warped_mask = np.zeros_like(raw_mask, dtype=np.float32)
                fused_mask = fuse_masks(
                    raw_mask,
                    warped_mask,
                    alpha=args.alpha,
                    non_free_threshold=args.non_free_threshold,
                )

            raw_path = raw_dir / f"{frame_id:06d}.npy"
            warped_path = warped_dir / f"{frame_id:06d}.npy"
            fused_path = fused_dir / f"{frame_id:06d}.npy"
            if args.save_arrays:
                np.save(raw_path, raw_mask.astype(np.float32))
                np.save(warped_path, warped_mask.astype(np.float32))
                np.save(fused_path, fused_mask.astype(np.float32))

            raw_visual = mask_for_visualization(raw_mask, args.vis_threshold, args.vis_binary)
            warped_visual = mask_for_visualization(warped_mask, args.vis_threshold, args.vis_binary)
            fused_visual = mask_for_visualization(fused_mask, args.vis_threshold, args.vis_binary)
            raw_overlay = overlay_mask(image_bgr, raw_visual)
            raw_mask_bgr = mask_to_bgr(raw_visual)
            fused_overlay = overlay_mask(image_bgr, fused_visual)
            fused_mask_bgr = mask_to_bgr(fused_visual)
            if not args.no_video:
                panel = make_temporal_panel(image_bgr, raw_visual, warped_visual, fused_visual)
            if not args.no_video and raw_writer is None:
                raw_writer = open_writer(temp_raw_video, fps, raw_overlay.shape)
                fused_writer = open_writer(temp_fused_video, fps, fused_overlay.shape)
                fused_mask_writer = open_writer(temp_fused_mask_video, fps, fused_mask_bgr.shape)
                panel_writer = open_writer(temp_panel_video, fps, panel.shape)
            if not args.no_video:
                raw_writer.write(raw_overlay)
                fused_writer.write(fused_overlay)
                fused_mask_writer.write(fused_mask_bgr)
                panel_writer.write(panel)

            if args.save_frames:
                cv2.imwrite(str(frame_dirs["raw_overlay"] / f"{frame_id:06d}.png"), raw_overlay)
                cv2.imwrite(str(frame_dirs["raw_mask"] / f"{frame_id:06d}.png"), raw_mask_bgr)
                cv2.imwrite(str(frame_dirs["fused_overlay"] / f"{frame_id:06d}.png"), fused_overlay)
                cv2.imwrite(str(frame_dirs["fused_mask"] / f"{frame_id:06d}.png"), fused_mask_bgr)

            writer.writerow(
                [
                    frame_id,
                    raw_path if args.save_arrays else "",
                    warped_path if args.save_arrays else "",
                    fused_path if args.save_arrays else "",
                ]
            )
            previous_bgr = image_bgr
            mask_buffer.append(fused_mask)
            frame_id += 1
            progress.update(1)
        progress.close()
    cap.release()

    for writer_obj in [raw_writer, fused_writer, fused_mask_writer, panel_writer]:
        if writer_obj is not None:
            writer_obj.release()

    if not args.no_video:
        convert_to_h264(temp_raw_video, raw_video)
        convert_to_h264(temp_fused_video, fused_video)
        convert_to_h264(temp_fused_mask_video, fused_mask_video)
        convert_to_h264(temp_panel_video, panel_video)
    print(f"output_dir: {output_dir}")
    print(f"manifest: {manifest_path}")
    print(f"alpha: {args.alpha}")
    print(f"non_free_threshold: {args.non_free_threshold}")
    print(f"history_size: {args.history_size}")
    print(f"history_decay: {args.history_decay}")
    if args.save_frames:
        print(f"raw_overlay_frames: {frame_dirs['raw_overlay']}")
        print(f"raw_mask_frames: {frame_dirs['raw_mask']}")
        print(f"fused_overlay_frames: {frame_dirs['fused_overlay']}")
        print(f"fused_mask_frames: {frame_dirs['fused_mask']}")
    if not args.no_video:
        print(f"raw_overlay_video: {raw_video}")
        print(f"fused_overlay_video: {fused_video}")
        print(f"fused_mask_video: {fused_mask_video}")
        print(f"temporal_panel_video: {panel_video}")


if __name__ == "__main__":
    main()
