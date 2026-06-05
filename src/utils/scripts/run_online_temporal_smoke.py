"""Run a short in-memory YOLOP + SEA-RAFT temporal fusion smoke test."""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from SEA_RAFT.sea_raft_wrapper import SEARAFTWrapper  # noqa: E402
from utils.temporal.fusion import fuse_masks  # noqa: E402
from utils.temporal.warp import warp_mask_with_flow  # noqa: E402
from utils.visualization.overlay import make_temporal_panel, mask_to_bgr  # noqa: E402
from YOLOP.yolop_wrapper import YOLOPFreeSpaceWrapper, read_image_bgr  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online YOLOP temporal free-space smoke test")
    parser.add_argument(
        "--input-glob",
        default="/workspace/src/YOLOP/external/YOLOP/inference/images/*.jpg",
        help="Sorted image sequence glob.",
    )
    parser.add_argument("--output-dir", default="/workspace/output/temporal_smoke")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--resize-width", type=int, default=512, help="0 keeps original size.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--non-free-threshold", type=float, default=0.2)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--yolop-repo", default="/workspace/src/YOLOP/external/YOLOP")
    parser.add_argument(
        "--yolop-checkpoint",
        default="/workspace/weights/YOLOP/End-to-end.pth",
    )
    parser.add_argument("--yolop-img-size", type=int, default=640)
    parser.add_argument("--mask-mode", default="probability", choices=["binary", "probability"])
    parser.add_argument("--sea-raft-repo", default="/workspace/src/SEA_RAFT/external/SEA-RAFT")
    parser.add_argument(
        "--sea-raft-config",
        default="/workspace/src/SEA_RAFT/external/SEA-RAFT/config/eval/spring-S.json",
    )
    parser.add_argument("--sea-raft-checkpoint", default="")
    parser.add_argument("--sea-raft-url", default="MemorySlices/Tartan-C-T-TSKH-spring540x960-S")
    return parser.parse_args()


def resize_if_needed(image_bgr: np.ndarray, resize_width: int) -> np.ndarray:
    if resize_width <= 0 or image_bgr.shape[1] <= resize_width:
        return image_bgr
    scale = resize_width / image_bgr.shape[1]
    height = int(round(image_bgr.shape[0] * scale))
    return cv2.resize(image_bgr, (resize_width, height), interpolation=cv2.INTER_AREA)


def open_writer(path: Path, fps: int, frame_shape: tuple[int, int, int]) -> cv2.VideoWriter:
    height, width = frame_shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def main() -> None:
    args = parse_args()
    image_paths = sorted(glob.glob(args.input_glob))
    if args.max_frames:
        image_paths = image_paths[: args.max_frames]
    if len(image_paths) < 2:
        raise ValueError("At least two images are required for SEA-RAFT temporal fusion.")

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw_masks"
    warped_dir = output_dir / "warped_masks"
    fused_dir = output_dir / "fused_masks"
    flow_dir = output_dir / "flows"
    preview_dir = output_dir / "previews"
    video_dir = output_dir / "videos"
    for directory in [raw_dir, warped_dir, fused_dir, flow_dir, preview_dir, video_dir]:
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

    manifest_path = output_dir / "temporal_manifest.csv"
    raw_writer = fused_writer = panel_writer = None
    previous_bgr: np.ndarray | None = None
    previous_fused: np.ndarray | None = None

    with manifest_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "frame_id",
                "rgb_path",
                "raw_mask_path",
                "flow_path",
                "warped_mask_path",
                "fused_mask_path",
            ]
        )

        for frame_id, image_path in enumerate(tqdm(image_paths, desc="temporal smoke")):
            image_bgr = resize_if_needed(read_image_bgr(image_path), args.resize_width)
            raw_mask = yolop.infer_drivable_mask(image_bgr)

            if previous_bgr is None or previous_fused is None:
                flow = np.zeros((*raw_mask.shape, 2), dtype=np.float32)
                warped_mask = np.zeros_like(raw_mask, dtype=np.float32)
                fused_mask = raw_mask
            else:
                flow = sea_raft.infer_flow(previous_bgr, image_bgr)
                warped_mask = warp_mask_with_flow(previous_fused, flow)
                fused_mask = fuse_masks(
                    raw_mask,
                    warped_mask,
                    alpha=args.alpha,
                    non_free_threshold=args.non_free_threshold,
                )

            raw_path = raw_dir / f"{frame_id:06d}.npy"
            flow_path = flow_dir / f"{frame_id:06d}.npy"
            warped_path = warped_dir / f"{frame_id:06d}.npy"
            fused_path = fused_dir / f"{frame_id:06d}.npy"
            np.save(raw_path, raw_mask.astype(np.float32))
            np.save(flow_path, flow.astype(np.float32))
            np.save(warped_path, warped_mask.astype(np.float32))
            np.save(fused_path, fused_mask.astype(np.float32))

            raw_bgr = mask_to_bgr(raw_mask)
            fused_bgr = mask_to_bgr(fused_mask)
            panel = make_temporal_panel(image_bgr, raw_mask, warped_mask, fused_mask)
            cv2.imwrite(str(preview_dir / f"{frame_id:06d}_raw.png"), raw_bgr)
            cv2.imwrite(str(preview_dir / f"{frame_id:06d}_fused.png"), fused_bgr)
            cv2.imwrite(str(preview_dir / f"{frame_id:06d}_panel.png"), panel)

            if raw_writer is None:
                raw_writer = open_writer(video_dir / "raw_mask.mp4", args.fps, raw_bgr.shape)
                fused_writer = open_writer(video_dir / "fused_mask.mp4", args.fps, fused_bgr.shape)
                panel_writer = open_writer(video_dir / "temporal_panel.mp4", args.fps, panel.shape)
            raw_writer.write(raw_bgr)
            fused_writer.write(fused_bgr)
            panel_writer.write(panel)

            writer.writerow([frame_id, image_path, raw_path, flow_path, warped_path, fused_path])
            previous_bgr = image_bgr
            previous_fused = fused_mask

    for writer_obj in [raw_writer, fused_writer, panel_writer]:
        if writer_obj is not None:
            writer_obj.release()

    print(f"output_dir: {output_dir}")
    print(f"manifest: {manifest_path}")
    print(f"raw_mask_video: {video_dir / 'raw_mask.mp4'}")
    print(f"fused_mask_video: {video_dir / 'fused_mask.mp4'}")
    print(f"temporal_panel_video: {video_dir / 'temporal_panel.mp4'}")


if __name__ == "__main__":
    main()
