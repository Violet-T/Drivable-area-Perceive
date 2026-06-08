"""Run YOLOP + SEA-RAFT + trained STGRU on a video and export mask videos."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.pipeline.freespace_temporal_pipeline import FreeSpaceTemporalPipeline, resize_frame_if_needed  # noqa: E402
from utils.visualization.overlay import mask_to_bgr, overlay_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export STGRU free-space mask video")
    parser.add_argument("--input-video", default=str(PROJECT_ROOT / "data" / "demo" / "1.mp4"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "output" / "stgru_video"))
    parser.add_argument("--stgru-checkpoint", default=str(PROJECT_ROOT / "weights" / "STGRU" / "stgru_best.pth"))
    parser.add_argument("--target-width", type=int, default=960)
    parser.add_argument("--target-height", type=int, default=540)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--yolop-repo", default=str(PROJECT_ROOT / "src" / "YOLOP" / "external" / "YOLOP"))
    parser.add_argument("--yolop-checkpoint", default=str(PROJECT_ROOT / "weights" / "YOLOP" / "End-to-end.pth"))
    parser.add_argument("--yolop-img-size", type=int, default=640)
    parser.add_argument("--sea-raft-repo", default=str(PROJECT_ROOT / "src" / "SEA_RAFT" / "external" / "SEA-RAFT"))
    parser.add_argument("--sea-raft-config", default=str(PROJECT_ROOT / "src" / "SEA_RAFT" / "external" / "SEA-RAFT" / "config" / "eval" / "spring-S.json"))
    parser.add_argument("--sea-raft-checkpoint", default="")
    parser.add_argument("--sea-raft-url", default="MemorySlices/Tartan-C-T-TSKH-spring540x960-S")
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--history-decay", type=float, default=0.6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_input_video(path: Path) -> Path:
    if path.exists():
        return path
    fallback = PROJECT_ROOT / "data" / "demo" / "eg.mp4"
    if fallback.exists():
        print(f"WARNING: {path} not found, using fallback {fallback}")
        return fallback
    raise FileNotFoundError(path)


def make_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {path}")
    return writer


def main() -> None:
    args = parse_args()
    input_video = ensure_input_video(Path(args.input_video))
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = FreeSpaceTemporalPipeline(
        yolop_repo=args.yolop_repo,
        yolop_checkpoint=args.yolop_checkpoint,
        sea_raft_repo=args.sea_raft_repo,
        sea_raft_config=args.sea_raft_config,
        sea_raft_checkpoint=args.sea_raft_checkpoint or None,
        sea_raft_url=args.sea_raft_url if not args.sea_raft_checkpoint else None,
        stgru_checkpoint=args.stgru_checkpoint,
        fusion_mode="stgru",
        yolop_img_size=args.yolop_img_size,
        mask_mode="probability",
        history_size=args.history_size,
        history_decay=args.history_decay,
        device=args.device,
    )

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(input_video)
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = args.fps if args.fps > 0 else source_fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    if args.max_frames and total_frames:
        total_frames = min(total_frames, args.max_frames)

    mask_writer = make_writer(output_dir / "stgru_mask.mp4", fps, args.target_width, args.target_height)
    overlay_writer = make_writer(output_dir / "stgru_overlay.mp4", fps, args.target_width, args.target_height)
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_id", "input_video", "mask_video", "overlay_video"])
        frame_count = 0
        progress = tqdm(total=total_frames, desc="stgru-video")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and frame_count >= args.max_frames:
                break
            frame = resize_frame_if_needed(frame, args.target_width, args.target_height)
            result = pipeline.step(frame)
            binary_mask = (np.clip(result.fused_mask, 0.0, 1.0) >= args.threshold).astype(np.float32)
            mask_writer.write(mask_to_bgr(binary_mask))
            overlay_writer.write(overlay_mask(frame, binary_mask))
            writer.writerow([result.frame_id, input_video, output_dir / "stgru_mask.mp4", output_dir / "stgru_overlay.mp4"])
            frame_count += 1
            progress.update(1)
        progress.close()

    cap.release()
    mask_writer.release()
    overlay_writer.release()
    print(f"output_dir: {output_dir}")
    print(f"mask_video: {output_dir / 'stgru_mask.mp4'}")
    print(f"overlay_video: {output_dir / 'stgru_overlay.mp4'}")


if __name__ == "__main__":
    main()
