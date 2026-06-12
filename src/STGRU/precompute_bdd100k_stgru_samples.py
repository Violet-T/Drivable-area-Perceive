"""Precompute YOLOP + SEA-RAFT STGRU samples from prepared BDD100K scenes."""

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

from utils.temporal.warp import warp_mask_with_flow  # noqa: E402
from utils.temporal.warp_image import warp_image_with_flow  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute BDD100K STGRU samples")
    parser.add_argument("--bdd-stgru-root", default=str(PROJECT_ROOT / "data" / "bdd100k_stgru"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "data" / "stgru_samples_bdd100k"))
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--image-height", type=int, default=540)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--yolop-repo", default=str(PROJECT_ROOT / "src" / "YOLOP" / "external" / "YOLOP"))
    parser.add_argument("--yolop-checkpoint", default=str(PROJECT_ROOT / "weights" / "YOLOP" / "End-to-end.pth"))
    parser.add_argument("--yolop-img-size", type=int, default=640)
    parser.add_argument("--sea-raft-repo", default=str(PROJECT_ROOT / "src" / "SEA_RAFT" / "external" / "SEA-RAFT"))
    parser.add_argument("--sea-raft-config", default=str(PROJECT_ROOT / "src" / "SEA_RAFT" / "external" / "SEA-RAFT" / "config" / "eval" / "spring-S.json"))
    parser.add_argument("--sea-raft-checkpoint", default="")
    parser.add_argument("--sea-raft-url", default="MemorySlices/Tartan-C-T-TSKH-spring540x960-S")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resize_image(image_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    if image_bgr.shape[1] == width and image_bgr.shape[0] == height:
        return image_bgr
    return cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape[:2] == (height, width):
        return mask
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def photometric_error(current_bgr: np.ndarray, warped_previous_bgr: np.ndarray) -> np.ndarray:
    current_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    previous_gray = cv2.cvtColor(warped_previous_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return np.abs(current_gray - previous_gray).astype(np.float32)


def rel(path: Path, data_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(data_root.resolve()))
    except ValueError:
        return str(path.resolve())


def read_scene_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def precompute_row(
    row: dict[str, str],
    split: str,
    output_root: Path,
    data_root: Path,
    width: int,
    height: int,
    yolop,
    sea_raft,
    overwrite: bool,
) -> dict[str, str]:
    sample_id = row["sample_id"]
    split_root = output_root / split
    current_mask_path = split_root / "current_mask" / f"{sample_id}.npy"
    previous_mask_path = split_root / "previous_mask" / f"{sample_id}.npy"
    warped_mask_path = split_root / "warped_mask" / f"{sample_id}.npy"
    target_mask_path = split_root / "target_mask" / f"{sample_id}.npy"
    flow_path = split_root / "flow" / f"{sample_id}.npy"
    warped_image_path = split_root / "warped_image" / f"{sample_id}.png"
    photo_error_path = split_root / "photometric_error" / f"{sample_id}.npy"
    outputs = [
        current_mask_path,
        previous_mask_path,
        warped_mask_path,
        target_mask_path,
        flow_path,
        warped_image_path,
        photo_error_path,
    ]
    if all(path.exists() for path in outputs) and not overwrite:
        return build_manifest_row(row, data_root, current_mask_path, warped_mask_path, target_mask_path, warped_image_path, photo_error_path, previous_mask_path, flow_path)

    previous_bgr = cv2.imread(row["previous_image"], cv2.IMREAD_COLOR)
    current_bgr = cv2.imread(row["current_image"], cv2.IMREAD_COLOR)
    if previous_bgr is None or current_bgr is None:
        raise FileNotFoundError(f"Missing BDD scene frames for {sample_id}")
    previous_bgr = resize_image(previous_bgr, width, height)
    current_bgr = resize_image(current_bgr, width, height)
    target = np.load(row["target_mask"]).astype(np.float32)
    target = resize_mask(target, width, height)

    previous_mask = yolop.infer_drivable_mask(previous_bgr)
    current_mask = yolop.infer_drivable_mask(current_bgr)

    flow = sea_raft.infer_flow(previous_bgr, current_bgr)
    warped_mask = warp_mask_with_flow(previous_mask, flow)
    warped_previous_bgr = warp_image_with_flow(previous_bgr, flow)
    photo_error = photometric_error(current_bgr, warped_previous_bgr)

    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    np.save(current_mask_path, current_mask.astype(np.float32))
    np.save(previous_mask_path, previous_mask.astype(np.float32))
    np.save(warped_mask_path, warped_mask.astype(np.float32))
    np.save(target_mask_path, target.astype(np.float32))
    np.save(flow_path, flow.astype(np.float32))
    np.save(photo_error_path, photo_error.astype(np.float32))
    cv2.imwrite(str(warped_image_path), warped_previous_bgr)
    return build_manifest_row(row, data_root, current_mask_path, warped_mask_path, target_mask_path, warped_image_path, photo_error_path, previous_mask_path, flow_path)


def build_manifest_row(
    row: dict[str, str],
    data_root: Path,
    current_mask_path: Path,
    warped_mask_path: Path,
    target_mask_path: Path,
    warped_image_path: Path,
    photo_error_path: Path,
    previous_mask_path: Path,
    flow_path: Path,
) -> dict[str, str]:
    return {
        "sample_id": row["sample_id"],
        "scene_id": row.get("scene_id", row["sample_id"]),
        "current_mask": rel(current_mask_path, data_root),
        "warped_mask": rel(warped_mask_path, data_root),
        "target_mask": rel(target_mask_path, data_root),
        "current_image": row["current_image"],
        "warped_previous_image": rel(warped_image_path, data_root),
        "photometric_error": rel(photo_error_path, data_root),
        "previous_mask": rel(previous_mask_path, data_root),
        "flow": rel(flow_path, data_root),
        "sequence_dir": row.get("sequence_dir", ""),
    }


def main() -> None:
    args = parse_args()
    from SEA_RAFT.sea_raft_wrapper import SEARAFTWrapper
    from YOLOP.yolop_wrapper import YOLOPFreeSpaceWrapper

    bdd_stgru_root = Path(args.bdd_stgru_root)
    output_root = Path(args.output_root)
    data_root = Path(args.data_root)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]

    yolop = YOLOPFreeSpaceWrapper(
        yolop_repo=args.yolop_repo,
        checkpoint_path=args.yolop_checkpoint,
        img_size=args.yolop_img_size,
        mask_mode="probability",
        device=args.device,
    )
    sea_raft = SEARAFTWrapper(
        sea_raft_repo=args.sea_raft_repo,
        config_path=args.sea_raft_config,
        checkpoint_path=args.sea_raft_checkpoint or None,
        model_url=args.sea_raft_url if not args.sea_raft_checkpoint else None,
        device=args.device,
    )

    fieldnames = [
        "sample_id",
        "scene_id",
        "current_mask",
        "warped_mask",
        "target_mask",
        "current_image",
        "warped_previous_image",
        "photometric_error",
        "previous_mask",
        "flow",
        "sequence_dir",
    ]
    for split in splits:
        scene_rows = read_scene_rows(bdd_stgru_root / f"{split}_scenes.csv")
        manifest_rows = [
            precompute_row(
                row=row,
                split=split,
                output_root=output_root,
                data_root=data_root,
                width=args.image_width,
                height=args.image_height,
                yolop=yolop,
                sea_raft=sea_raft,
                overwrite=args.overwrite,
            )
            for row in tqdm(scene_rows, desc=f"bdd-precompute:{split}")
        ]
        csv_path = output_root / f"{split}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"{split}: {len(manifest_rows)} samples -> {csv_path}")


if __name__ == "__main__":
    main()
