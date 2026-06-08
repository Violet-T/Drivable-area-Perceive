"""Convert Cityscapes labelIds into binary free-space masks."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare binary Cityscapes road/free-space masks")
    parser.add_argument("--cityscapes-root", default=str(PROJECT_ROOT / "data" / "cityscapes"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "data" / "cityscapes_binary"))
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--free-label-ids", default="7", help="Comma separated Cityscapes labelIds treated as free-space")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_label_ids(value: str) -> list[int]:
    ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not ids:
        raise ValueError("--free-label-ids must contain at least one id")
    return ids


def convert_label(label_path: Path, output_path: Path, free_label_ids: list[int], overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        return
    label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(label_path)
    mask = np.isin(label, np.asarray(free_label_ids, dtype=label.dtype)).astype(np.uint8) * 255
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask)


def main() -> None:
    args = parse_args()
    cityscapes_root = Path(args.cityscapes_root)
    output_root = Path(args.output_root)
    free_label_ids = parse_label_ids(args.free_label_ids)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]

    manifest_path = output_root / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "label_path", "binary_mask_path", "free_label_ids"])
        total = 0
        for split in splits:
            label_paths = sorted((cityscapes_root / "gtFine" / split).glob("*/*_gtFine_labelIds.png"))
            for label_path in tqdm(label_paths, desc=f"binary:{split}"):
                relative = label_path.relative_to(cityscapes_root / "gtFine" / split)
                output_path = output_root / split / relative.with_name(
                    relative.name.replace("_gtFine_labelIds.png", "_binary_free_space.png")
                )
                convert_label(label_path, output_path, free_label_ids, overwrite=args.overwrite)
                writer.writerow([split, label_path, output_path, ",".join(map(str, free_label_ids))])
                total += 1
    print(f"binary_masks: {output_root}")
    print(f"manifest: {manifest_path}")
    print(f"count: {total}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
