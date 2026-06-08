"""Run the in-memory YOLOP + SEA-RAFT + alpha/STGRU pipeline on a video."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.pipeline.freespace_temporal_pipeline import (  # noqa: E402
    FreeSpaceTemporalPipeline,
    PipelineResult,
    resize_frame_if_needed,
)
from utils.visualization.overlay import mask_to_bgr, overlay_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified free-space temporal pipeline")
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output-dir", default="/workspace/output/freespace_pipeline")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--target-width", type=int, default=0)
    parser.add_argument("--target-height", type=int, default=0)
    parser.add_argument("--fusion-mode", default="alpha", choices=["alpha", "stgru"])
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--non-free-threshold", type=float, default=0.2)
    parser.add_argument("--history-size", type=int, default=3)
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
    parser.add_argument("--stgru-checkpoint", default="")
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--vis-threshold", type=float, default=0.5)
    parser.add_argument("--vis-binary", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def visual_mask(mask: np.ndarray, threshold: float, binary: bool) -> np.ndarray:
    display = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if threshold > 0.0:
        display = np.where(display >= threshold, display, 0.0).astype(np.float32)
    if binary:
        display = (display >= max(threshold, 0.5)).astype(np.float32)
    return display


def save_result(
    result: PipelineResult,
    output_dir: Path,
    save_arrays: bool,
    save_frames: bool,
    threshold: float,
    binary: bool,
) -> tuple[str, str, str]:
    raw_path = warped_path = fused_path = ""
    if save_arrays:
        raw_dir = output_dir / "raw_masks"
        warped_dir = output_dir / "warped_masks"
        fused_dir = output_dir / "fused_masks"
        gate_dir = output_dir / "gates"
        for directory in [raw_dir, warped_dir, fused_dir, gate_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        raw_path = str(raw_dir / f"{result.frame_id:06d}.npy")
        warped_path = str(warped_dir / f"{result.frame_id:06d}.npy")
        fused_path = str(fused_dir / f"{result.frame_id:06d}.npy")
        np.save(raw_path, result.raw_mask)
        np.save(warped_path, result.warped_mask)
        np.save(fused_path, result.fused_mask)
        if result.update_gate is not None:
            np.save(gate_dir / f"{result.frame_id:06d}_update.npy", result.update_gate)
        if result.reset_gate is not None:
            np.save(gate_dir / f"{result.frame_id:06d}_reset.npy", result.reset_gate)

    if save_frames:
        raw_overlay_dir = output_dir / "raw_overlay_frames"
        fused_overlay_dir = output_dir / "fused_overlay_frames"
        fused_mask_dir = output_dir / "fused_mask_frames"
        for directory in [raw_overlay_dir, fused_overlay_dir, fused_mask_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        raw_vis = visual_mask(result.raw_mask, threshold, binary)
        fused_vis = visual_mask(result.fused_mask, threshold, binary)
        cv2.imwrite(str(raw_overlay_dir / f"{result.frame_id:06d}.png"), overlay_mask(result.image_bgr, raw_vis))
        cv2.imwrite(str(fused_overlay_dir / f"{result.frame_id:06d}.png"), overlay_mask(result.image_bgr, fused_vis))
        cv2.imwrite(str(fused_mask_dir / f"{result.frame_id:06d}.png"), mask_to_bgr(fused_vis))

    return raw_path, warped_path, fused_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite to replace outputs.")
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = FreeSpaceTemporalPipeline(
        yolop_repo=args.yolop_repo,
        yolop_checkpoint=args.yolop_checkpoint,
        sea_raft_repo=args.sea_raft_repo,
        sea_raft_config=args.sea_raft_config,
        sea_raft_checkpoint=args.sea_raft_checkpoint or None,
        sea_raft_url=args.sea_raft_url,
        stgru_checkpoint=args.stgru_checkpoint or None,
        fusion_mode=args.fusion_mode,
        yolop_img_size=args.yolop_img_size,
        mask_mode=args.mask_mode,
        alpha=args.alpha,
        non_free_threshold=args.non_free_threshold,
        history_size=args.history_size,
        history_decay=args.history_decay,
        device=args.device,
    )
    if args.fusion_mode == "stgru" and pipeline.stgru is not None and not pipeline.stgru.is_trained:
        print("WARNING: running untrained STGRU. Use this only for structure smoke tests.")

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise FileNotFoundError(args.input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    if args.max_frames and total_frames:
        total_frames = min(total_frames, args.max_frames)

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame_id", "fusion_mode", "raw_mask_path", "warped_mask_path", "fused_mask_path"])
        progress = tqdm(total=total_frames, desc=f"pipeline:{args.fusion_mode}")
        frame_count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and frame_count >= args.max_frames:
                break
            frame = resize_frame_if_needed(frame, args.target_width, args.target_height)
            result = pipeline.step(frame)
            raw_path, warped_path, fused_path = save_result(
                result,
                output_dir=output_dir,
                save_arrays=args.save_arrays,
                save_frames=args.save_frames,
                threshold=args.vis_threshold,
                binary=args.vis_binary,
            )
            writer.writerow([result.frame_id, result.fusion_mode, raw_path, warped_path, fused_path])
            frame_count += 1
            progress.update(1)
        progress.close()
    cap.release()
    print(f"output_dir: {output_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
