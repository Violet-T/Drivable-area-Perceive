#!/usr/bin/env python3
"""Download selected BDD100K image keyframes from the official images zip.

BDD100K drivable-area labels are keyed to the 10s image keyframe. This script
downloads only the matching image files for selected scene ids instead of
downloading the whole image package.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from download_bdd100k_video_scenes import (  # noqa: E402
    Budget,
    DownloadContext,
    build_headers,
    build_local_drivable_label_index,
    human_bytes,
    label_scene_stem,
    list_remote_zip_members,
    parse_size,
    read_cookie_file,
    read_remote_zip_member_bytes,
    remote_zip_is_usable,
    scene_stem,
    should_use_split,
)


DEFAULT_BDD_IMAGE_URL = "http://128.32.162.150/bdd100k/bdd100k_images_100k.zip"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download selected BDD100K keyframe images")
    parser.add_argument("--image-url", default=DEFAULT_BDD_IMAGE_URL, help="BDD100K images_100k zip URL")
    parser.add_argument("--local-archive", type=Path, default=None, help="Local bdd100k_images_100k.zip path")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "bdd100k_keyframes")
    parser.add_argument("--split", default="train", help="BDD split: train/val/test")
    parser.add_argument("--scene-name", action="append", default=[], help="Scene id to download; can repeat")
    parser.add_argument("--scene-list", type=Path, default=None, help="Text file with one scene id per line")
    parser.add_argument("--stgru-scenes-csv", action="append", default=[], help="CSV produced by BDD STGRU prepare step")
    parser.add_argument("--drivable-root", type=Path, default=None, help="Infer scene ids from local drivable maps")
    parser.add_argument("--max-scenes", type=int, default=0, help="Limit selected scenes after discovery; 0 means no limit")
    parser.add_argument("--cookie", default=None, help="Cookie header content")
    parser.add_argument("--cookie-file", type=Path, default=None, help="Netscape/browser cookie file")
    parser.add_argument("--max-total-size", type=parse_size, default=parse_size("20G"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def read_scene_list(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def read_scene_ids_from_csv(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    result: list[str] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scene_id = row.get("scene_id") or row.get("sample_id")
            if scene_id:
                result.append(scene_id)
    return result


def unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def collect_target_scene_ids(args: argparse.Namespace) -> list[str]:
    scene_ids = list(args.scene_name)
    scene_ids.extend(read_scene_list(args.scene_list))
    for csv_path in args.stgru_scenes_csv:
        scene_ids.extend(read_scene_ids_from_csv(Path(csv_path)))
    if args.drivable_root is not None:
        index = build_local_drivable_label_index(args.drivable_root, args.split or None)
        scene_ids.extend(sorted(index))
    scene_ids = unique_keep_order([item.strip() for item in scene_ids if item.strip()])
    if args.max_scenes > 0:
        scene_ids = scene_ids[: args.max_scenes]
    return scene_ids


def destination_for(output_root: Path, split: str, member_name: str, stem: str) -> Path:
    suffix = Path(member_name).suffix.lower()
    return output_root / "images" / split / f"{stem}{suffix}"


def write_manifest(output_root: Path, rows: list[dict[str, str]]) -> None:
    manifest_path = output_root / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scene_id", "split", "image_path", "source_member"]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logging.info("manifest: %s", manifest_path)


def extract_from_local_archive(args: argparse.Namespace, scene_ids: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    assert args.local_archive is not None
    with zipfile.ZipFile(args.local_archive) as archive:
        for info in archive.infolist():
            if info.is_dir() or Path(info.filename).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if args.split and not should_use_split(info.filename, args.split):
                continue
            stem = scene_stem(info.filename)
            if stem not in scene_ids:
                continue
            destination = destination_for(args.output_root, args.split, info.filename, stem)
            if destination.exists() and not args.overwrite:
                logging.info("reuse: %s", destination)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, destination.open("wb") as target:
                    target.write(source.read())
            rows.append(
                {
                    "scene_id": stem,
                    "split": args.split,
                    "image_path": str(destination),
                    "source_member": info.filename,
                }
            )
    return rows


def extract_from_remote_archive(args: argparse.Namespace, scene_ids: set[str]) -> list[dict[str, str]]:
    output_root = args.output_root.resolve()
    cache_dir = output_root / "_cache"
    manifest_dir = output_root / "manifests"
    for path in (cache_dir, manifest_dir):
        path.mkdir(parents=True, exist_ok=True)
    headers = build_headers(args.cookie, args.cookie_file)
    context = DownloadContext(
        output_dir=output_root,
        cache_dir=cache_dir,
        scenes_dir=output_root / "_unused_scenes",
        manifest_dir=manifest_dir,
        headers=headers,
        budget=Budget(max_bytes=args.max_total_size, root=output_root),
        keep_archives=False,
    )
    if not remote_zip_is_usable(args.image_url, context):
        raise RuntimeError(
            "Remote image zip does not support range access or content length is unavailable. "
            "Use --local-archive with a downloaded bdd100k_images_100k.zip."
        )
    rows: list[dict[str, str]] = []
    members = list_remote_zip_members(args.image_url, context)
    for member in members:
        if member.filename.endswith("/") or Path(member.filename).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if args.split and not should_use_split(member.filename, args.split):
            continue
        stem = scene_stem(member.filename)
        if stem not in scene_ids:
            continue
        destination = destination_for(output_root, args.split, member.filename, stem)
        if destination.exists() and not args.overwrite:
            logging.info("reuse: %s", destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = read_remote_zip_member_bytes(args.image_url, member, context)
            destination.write_bytes(data)
            logging.info("downloaded keyframe: %s", destination)
        rows.append(
            {
                "scene_id": stem,
                "split": args.split,
                "image_path": str(destination),
                "source_member": member.filename,
            }
        )
        if len(rows) >= len(scene_ids):
            break
    logging.info(
        "download bytes: %s, output size: %s",
        human_bytes(context.budget.network_bytes),
        human_bytes(context.budget.final_bytes()),
    )
    return rows


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    args.output_root.mkdir(parents=True, exist_ok=True)

    scene_ids = collect_target_scene_ids(args)
    if not scene_ids:
        raise ValueError("No scene ids. Use --scene-name, --scene-list, --stgru-scenes-csv, or --drivable-root.")
    logging.info("target scene count: %d", len(scene_ids))
    target_set = set(scene_ids)

    if args.local_archive is not None:
        rows = extract_from_local_archive(args, target_set)
    else:
        rows = extract_from_remote_archive(args, target_set)

    found = {row["scene_id"] for row in rows}
    missing = [scene_id for scene_id in scene_ids if scene_id not in found]
    if missing:
        logging.warning("missing keyframes: %d", len(missing))
        (args.output_root / "missing_scene_ids.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")
    write_manifest(args.output_root, rows)
    logging.info("done: %d/%d keyframes", len(rows), len(scene_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
