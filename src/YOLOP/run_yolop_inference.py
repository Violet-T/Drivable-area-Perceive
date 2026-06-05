"""Run YOLOP single-frame drivable-area inference."""

from __future__ import annotations

import argparse
import csv
import glob
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.visualization.overlay import mask_to_bgr, overlay_mask  # noqa: E402
from YOLOP.yolop_wrapper import YOLOPFreeSpaceWrapper, read_image_bgr  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOP drivable-area/free-space inference")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-video", default="", help="Input video path.")
    source.add_argument("--input-glob", default="", help="Sorted image sequence glob.")
    parser.add_argument("--output-dir", default="/workspace/output/yolop_inference")
    parser.add_argument("--yolop-repo", default="/workspace/src/YOLOP/external/YOLOP")
    parser.add_argument("--checkpoint", default="/workspace/weights/YOLOP/End-to-end.pth")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--mask-mode", default="probability", choices=["probability", "binary"])
    parser.add_argument("--fps", type=float, default=0.0, help="0 uses input video FPS or 10 for image glob.")
    parser.add_argument("--target-width", type=int, default=0, help="Resize frames before inference; 0 keeps input width.")
    parser.add_argument("--target-height", type=int, default=0, help="Resize frames before inference; 0 keeps input height.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames.")
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument(
        "--vis-threshold",
        type=float,
        default=0.0,
        help="Suppress visualization mask values below this threshold; does not change saved raw masks.",
    )
    parser.add_argument(
        "--vis-binary",
        action="store_true",
        help="Render visualization masks as binary after vis-threshold; does not change saved raw masks.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--save-frames", action="store_true", help="Save overlay and mask PNG frames.")
    parser.add_argument("--no-video", action="store_true", help="Do not export MP4 videos.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def open_writer(path: Path, fps: float, frame_shape: tuple[int, int, int]) -> cv2.VideoWriter:
    height, width = frame_shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def convert_to_h264(src: Path, dst: Path) -> None:
    cmd = [
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
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def iter_video(video_path: Path, max_frames: int) -> tuple[list[tuple[int, np.ndarray]], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[tuple[int, np.ndarray]] = []
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append((frame_id, frame))
        frame_id += 1
        if max_frames and frame_id >= max_frames:
            break
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from {video_path}")
    return frames, fps


def iter_images(input_glob: str, max_frames: int) -> tuple[list[tuple[int, np.ndarray, str]], float]:
    paths = sorted(glob.glob(input_glob))
    if max_frames:
        paths = paths[:max_frames]
    if not paths:
        raise ValueError(f"No images matched: {input_glob}")
    frames = [(idx, read_image_bgr(path), path) for idx, path in enumerate(paths)]
    return frames, 10.0


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
    """Prepare a display-only HxW mask while keeping model outputs unchanged."""
    visual = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if threshold > 0.0:
        visual = np.where(visual >= threshold, visual, 0.0).astype(np.float32)
    if binary:
        visual = (visual >= max(threshold, 0.5)).astype(np.float32)
    return visual


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite to replace outputs.")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_masks"
    preview_dir = output_dir / "previews"
    frame_dirs = {
        "overlay": output_dir / "overlay_frames_30fps",
        "mask": output_dir / "mask_frames_30fps",
    }
    video_dir = output_dir / "videos"
    directories = [raw_dir]
    if args.save_preview:
        directories.append(preview_dir)
    if args.save_frames:
        directories.extend(frame_dirs.values())
    if not args.no_video:
        directories.append(video_dir)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    yolop = YOLOPFreeSpaceWrapper(
        yolop_repo=args.yolop_repo,
        checkpoint_path=args.checkpoint,
        img_size=args.img_size,
        mask_mode=args.mask_mode,
        device=args.device,
    )

    if args.input_video:
        video_path = Path(args.input_video)
        video_frames, source_fps = iter_video(video_path, args.max_frames)
        frames = [(frame_id, frame, str(video_path)) for frame_id, frame in video_frames]
        output_fps = args.fps or source_fps
        source_name = video_path.stem
    else:
        frames, source_fps = iter_images(args.input_glob, args.max_frames)
        output_fps = args.fps or source_fps
        source_name = "image_sequence"

    manifest_path = output_dir / "manifest.csv"
    temp_overlay_video = video_dir / f"{source_name}_overlay_mp4v.mp4"
    temp_mask_video = video_dir / f"{source_name}_mask_mp4v.mp4"
    overlay_video = video_dir / f"{source_name}_overlay_h264.mp4"
    mask_video = video_dir / f"{source_name}_mask_h264.mp4"
    overlay_writer = mask_writer = None

    with manifest_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame_id", "rgb_path", "raw_mask_path", "mask_frame_path", "overlay_frame_path"])
        for frame_id, image_bgr, rgb_path in tqdm(frames, desc="YOLOP masks"):
            image_bgr = resize_frame_if_needed(image_bgr, args.target_width, args.target_height)
            mask = yolop.infer_drivable_mask(image_bgr)
            raw_path = raw_dir / f"{frame_id:06d}.npy"
            preview_path = preview_dir / f"{frame_id:06d}_mask.png"
            overlay_path = preview_dir / f"{frame_id:06d}_overlay.png"
            np.save(raw_path, mask.astype(np.float32))

            visual_mask = mask_for_visualization(mask, args.vis_threshold, args.vis_binary)
            mask_bgr = mask_to_bgr(visual_mask)
            overlay_bgr = overlay_mask(image_bgr, visual_mask, alpha=args.overlay_alpha)
            if args.save_preview:
                cv2.imwrite(str(preview_path), mask_bgr)
                cv2.imwrite(str(overlay_path), overlay_bgr)
            if args.save_frames:
                cv2.imwrite(str(frame_dirs["mask"] / f"{frame_id:06d}.png"), mask_bgr)
                cv2.imwrite(str(frame_dirs["overlay"] / f"{frame_id:06d}.png"), overlay_bgr)

            if not args.no_video and overlay_writer is None:
                overlay_writer = open_writer(temp_overlay_video, output_fps, overlay_bgr.shape)
                mask_writer = open_writer(temp_mask_video, output_fps, mask_bgr.shape)
            if not args.no_video:
                overlay_writer.write(overlay_bgr)
                mask_writer.write(mask_bgr)
            writer.writerow(
                [
                    frame_id,
                    rgb_path,
                    raw_path,
                    frame_dirs["mask"] / f"{frame_id:06d}.png" if args.save_frames else "",
                    frame_dirs["overlay"] / f"{frame_id:06d}.png" if args.save_frames else "",
                ]
            )

    for writer_obj in [overlay_writer, mask_writer]:
        if writer_obj is not None:
            writer_obj.release()

    if not args.no_video:
        convert_to_h264(temp_overlay_video, overlay_video)
        convert_to_h264(temp_mask_video, mask_video)
    print(f"output_dir: {output_dir}")
    print(f"manifest: {manifest_path}")
    print(f"raw_masks: {raw_dir}")
    if args.save_frames:
        print(f"overlay_frames: {frame_dirs['overlay']}")
        print(f"mask_frames: {frame_dirs['mask']}")
    if not args.no_video:
        print(f"overlay_video: {overlay_video}")
        print(f"mask_video: {mask_video}")


if __name__ == "__main__":
    main()
