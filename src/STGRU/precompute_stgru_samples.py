"""Precompute YOLOP + SEA-RAFT inputs for STGRU training."""

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

from SEA_RAFT.sea_raft_wrapper import SEARAFTWrapper  # noqa: E402
from YOLOP.yolop_wrapper import YOLOPFreeSpaceWrapper  # noqa: E402
from utils.temporal.warp import warp_mask_with_flow  # noqa: E402
from utils.temporal.warp_image import warp_image_with_flow  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute STGRU training samples")
    parser.add_argument("--cityscapes-root", default=str(PROJECT_ROOT / "data" / "cityscapes"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "data" / "stgru_samples"))
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--require-sequence", action="store_true")
    parser.add_argument("--previous-frame-offset", type=int, default=1)
    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--image-height", type=int, default=540)
    parser.add_argument("--free-label-ids", default="7")
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


def parse_label_ids(value: str) -> list[int]:
    ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not ids:
        raise ValueError("--free-label-ids must contain at least one id")
    return ids


def resize_image(image_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    if image_bgr.shape[1] == width and image_bgr.shape[0] == height:
        return image_bgr
    return cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)


def label_to_binary(label_path: Path, free_label_ids: list[int], width: int, height: int) -> np.ndarray:
    label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(label_path)
    if label.shape[:2] != (height, width):
        label = cv2.resize(label, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.isin(label, np.asarray(free_label_ids, dtype=label.dtype)).astype(np.float32)


def photometric_error(current_bgr: np.ndarray, warped_previous_bgr: np.ndarray) -> np.ndarray:
    current_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    previous_gray = cv2.cvtColor(warped_previous_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return np.abs(current_gray - previous_gray).astype(np.float32)


def cityscapes_bases(label_path: Path) -> tuple[str, str, int]:
    base = label_path.name.replace("_gtFine_labelIds.png", "")
    parts = base.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected Cityscapes label name: {label_path.name}")
    return base, "_".join(parts[:-1]), int(parts[-1])


def resolve_image_pair(
    cityscapes_root: Path,
    split: str,
    city: str,
    label_path: Path,
    previous_frame_offset: int,
    require_sequence: bool,
) -> tuple[Path, Path] | None:
    base, prefix, frame_id = cityscapes_bases(label_path)
    previous_base = f"{prefix}_{max(0, frame_id - previous_frame_offset):06d}"
    sequence_dir = cityscapes_root / "leftImg8bit_sequence" / split / city
    current = sequence_dir / f"{base}_leftImg8bit.png"
    previous = sequence_dir / f"{previous_base}_leftImg8bit.png"
    if current.exists() and previous.exists():
        return current, previous
    if require_sequence:
        return None
    current = cityscapes_root / "leftImg8bit" / split / city / f"{base}_leftImg8bit.png"
    if current.exists():
        return current, current
    return None


def rel(path: Path, data_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(data_root.resolve()))
    except ValueError:
        return str(path.resolve())


def save_sample(
    split: str,
    sample_id: str,
    label_path: Path,
    current_path: Path,
    previous_path: Path,
    output_root: Path,
    data_root: Path,
    width: int,
    height: int,
    free_label_ids: list[int],
    yolop: YOLOPFreeSpaceWrapper,
    sea_raft: SEARAFTWrapper,
    overwrite: bool,
) -> dict[str, str]:
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
        return {
            "sample_id": sample_id,
            "current_mask": rel(current_mask_path, data_root),
            "warped_mask": rel(warped_mask_path, data_root),
            "target_mask": rel(target_mask_path, data_root),
            "current_image": rel(current_path, data_root),
            "warped_previous_image": rel(warped_image_path, data_root),
            "photometric_error": rel(photo_error_path, data_root),
            "previous_mask": rel(previous_mask_path, data_root),
            "flow": rel(flow_path, data_root),
            "label_path": rel(label_path, data_root),
        }

    current_bgr = cv2.imread(str(current_path), cv2.IMREAD_COLOR)
    previous_bgr = cv2.imread(str(previous_path), cv2.IMREAD_COLOR)
    if current_bgr is None or previous_bgr is None:
        raise FileNotFoundError(f"Missing image pair: {previous_path}, {current_path}")
    current_bgr = resize_image(current_bgr, width, height)
    previous_bgr = resize_image(previous_bgr, width, height)

    current_mask = yolop.infer_drivable_mask(current_bgr)
    previous_mask = yolop.infer_drivable_mask(previous_bgr)
    if current_path == previous_path:
        flow = np.zeros((height, width, 2), dtype=np.float32)
    else:
        flow = sea_raft.infer_flow(previous_bgr, current_bgr)
    warped_mask = warp_mask_with_flow(previous_mask, flow)
    warped_previous_bgr = warp_image_with_flow(previous_bgr, flow)
    photo_error = photometric_error(current_bgr, warped_previous_bgr)
    target_mask = label_to_binary(label_path, free_label_ids, width, height)

    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    np.save(current_mask_path, current_mask.astype(np.float32))
    np.save(previous_mask_path, previous_mask.astype(np.float32))
    np.save(warped_mask_path, warped_mask.astype(np.float32))
    np.save(target_mask_path, target_mask.astype(np.float32))
    np.save(flow_path, flow.astype(np.float32))
    cv2.imwrite(str(warped_image_path), warped_previous_bgr)
    np.save(photo_error_path, photo_error.astype(np.float32))

    return {
        "sample_id": sample_id,
        "current_mask": rel(current_mask_path, data_root),
        "warped_mask": rel(warped_mask_path, data_root),
        "target_mask": rel(target_mask_path, data_root),
        "current_image": rel(current_path, data_root),
        "warped_previous_image": rel(warped_image_path, data_root),
        "photometric_error": rel(photo_error_path, data_root),
        "previous_mask": rel(previous_mask_path, data_root),
        "flow": rel(flow_path, data_root),
        "label_path": rel(label_path, data_root),
    }


def main() -> None:
    args = parse_args()
    cityscapes_root = Path(args.cityscapes_root)
    output_root = Path(args.output_root)
    data_root = Path(args.data_root)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    free_label_ids = parse_label_ids(args.free_label_ids)

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
        "current_mask",
        "warped_mask",
        "target_mask",
        "current_image",
        "warped_previous_image",
        "photometric_error",
        "previous_mask",
        "flow",
        "label_path",
    ]
    for split in splits:
        rows: list[dict[str, str]] = []
        label_paths = sorted((cityscapes_root / "gtFine" / split).glob("*/*_gtFine_labelIds.png"))
        for label_path in tqdm(label_paths, desc=f"precompute:{split}"):
            if args.max_samples_per_split > 0 and len(rows) >= args.max_samples_per_split:
                break
            city = label_path.parent.name
            image_pair = resolve_image_pair(
                cityscapes_root=cityscapes_root,
                split=split,
                city=city,
                label_path=label_path,
                previous_frame_offset=args.previous_frame_offset,
                require_sequence=args.require_sequence,
            )
            if image_pair is None:
                continue
            current_path, previous_path = image_pair
            sample_id = label_path.name.replace("_gtFine_labelIds.png", "")
            row = save_sample(
                split=split,
                sample_id=sample_id,
                label_path=label_path,
                current_path=current_path,
                previous_path=previous_path,
                output_root=output_root,
                data_root=data_root,
                width=args.image_width,
                height=args.image_height,
                free_label_ids=free_label_ids,
                yolop=yolop,
                sea_raft=sea_raft,
                overwrite=args.overwrite,
            )
            rows.append(row)

        csv_path = output_root / f"{split}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{split}: {len(rows)} samples -> {csv_path}")


if __name__ == "__main__":
    main()
