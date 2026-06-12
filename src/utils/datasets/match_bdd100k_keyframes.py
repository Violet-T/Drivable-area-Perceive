#!/usr/bin/env python3
"""Match BDD100K extracted video frames to image keyframes.

For each scene, compare frames around 10s with the corresponding BDD100K
image keyframe, choose the best matching frame as the supervised keyframe, and
copy the previous N frames plus the matched frame for STGRU training.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class MatchResult:
    scene_id: str
    keyframe_image: Path
    matched_frame: Path
    matched_index: int
    score: float
    absdiff_score: float
    hist_score: float
    orb_score: float
    sequence_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match BDD100K video frames to keyframe images")
    parser.add_argument("--scenes-root", type=Path, default=PROJECT_ROOT / "data" / "bdd100k_video_scenes" / "scenes")
    parser.add_argument("--keyframes-root", type=Path, default=PROJECT_ROOT / "data" / "bdd100k_keyframes" / "images")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "bdd100k_keyframe_matched")
    parser.add_argument("--scene-name", action="append", default=[], help="Only process this scene id; can repeat")
    parser.add_argument("--scene-list", type=Path, default=None)
    parser.add_argument("--frames-subdir", default="", help="Override relative frame dir under each scene")
    parser.add_argument("--previous-frames", type=int, default=5)
    parser.add_argument("--candidate-window", type=int, default=0, help="Only search center +/- window frames; 0 means all")
    parser.add_argument("--resize-width", type=int, default=320)
    parser.add_argument("--resize-height", type=int, default=180)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--copy-keyframe-image", action="store_true")
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


def scene_id_from_path(path: Path) -> str:
    stem = path.stem
    for suffix in ("_drivable_id", "_drivable_color", "_id", "_color"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def collect_scene_ids(args: argparse.Namespace) -> list[str]:
    scene_ids = list(args.scene_name)
    scene_ids.extend(read_scene_list(args.scene_list))
    if scene_ids:
        seen: set[str] = set()
        result: list[str] = []
        for scene_id in scene_ids:
            if scene_id not in seen:
                seen.add(scene_id)
                result.append(scene_id)
        return result
    if not args.scenes_root.exists():
        raise FileNotFoundError(args.scenes_root)
    return sorted(path.name for path in args.scenes_root.iterdir() if path.is_dir())


def find_keyframe(keyframes_root: Path, scene_id: str) -> Path | None:
    direct_candidates = []
    for suffix in IMAGE_EXTENSIONS:
        direct_candidates.extend(keyframes_root.glob(f"**/{scene_id}{suffix}"))
    if direct_candidates:
        return sorted(direct_candidates)[0]
    for path in keyframes_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and scene_id_from_path(path) == scene_id:
            return path
    return None


def find_frames_dir(scene_dir: Path, frames_subdir: str) -> Path | None:
    if frames_subdir:
        candidate = scene_dir / frames_subdir
        return candidate if candidate.exists() else None
    candidates = [
        scene_dir / "stgru" / "frames_9_12s",
        scene_dir / "frames_9_12s",
        scene_dir / "frames",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def list_frames(frames_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def frame_numeric_index(path: Path, fallback: int) -> int:
    digits = "".join(ch if ch.isdigit() else " " for ch in path.stem).split()
    if not digits:
        return fallback
    return int(digits[-1])


def read_resized_bgr(path: Path, width: int, height: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    if width > 0 and height > 0:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def absdiff_similarity(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    a_gray = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY)
    return float(1.0 - np.mean(cv2.absdiff(a_gray, b_gray)) / 255.0)


def hist_similarity(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    a_hsv = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2HSV)
    b_hsv = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2HSV)
    a_hist = cv2.calcHist([a_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    b_hist = cv2.calcHist([b_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(a_hist, a_hist, 0.0, 1.0, cv2.NORM_MINMAX)
    cv2.normalize(b_hist, b_hist, 0.0, 1.0, cv2.NORM_MINMAX)
    corr = cv2.compareHist(a_hist, b_hist, cv2.HISTCMP_CORREL)
    return float(np.clip((corr + 1.0) * 0.5, 0.0, 1.0))


def orb_similarity(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    orb = cv2.ORB_create(nfeatures=600)
    a_gray = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY)
    kp_a, des_a = orb.detectAndCompute(a_gray, None)
    kp_b, des_b = orb.detectAndCompute(b_gray, None)
    if des_a is None or des_b is None or not kp_a or not kp_b:
        return 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_a, des_b)
    if not matches:
        return 0.0
    good = [match for match in matches if match.distance <= 50]
    denom = max(min(len(kp_a), len(kp_b)), 1)
    return float(np.clip(len(good) / denom, 0.0, 1.0))


def compare_images(frame_bgr: np.ndarray, keyframe_bgr: np.ndarray) -> tuple[float, float, float, float]:
    diff = absdiff_similarity(frame_bgr, keyframe_bgr)
    hist = hist_similarity(frame_bgr, keyframe_bgr)
    orb = orb_similarity(frame_bgr, keyframe_bgr)
    score = 0.55 * diff + 0.30 * hist + 0.15 * orb
    return float(score), float(diff), float(hist), float(orb)


def candidate_frames(frames: list[Path], window: int) -> list[tuple[int, Path]]:
    indexed = list(enumerate(frames))
    if window <= 0 or not indexed:
        return indexed
    center = len(indexed) // 2
    start = max(0, center - window)
    end = min(len(indexed), center + window + 1)
    return indexed[start:end]


def copy_sequence(
    frames: list[Path],
    matched_pos: int,
    previous_frames: int,
    output_root: Path,
    scene_id: str,
    overwrite: bool,
) -> Path:
    if matched_pos < previous_frames:
        raise ValueError(f"{scene_id}: matched frame has fewer than {previous_frames} previous frames")
    sequence_dir = output_root / scene_id / "sequence"
    if sequence_dir.exists() and overwrite:
        shutil.rmtree(sequence_dir)
    sequence_dir.mkdir(parents=True, exist_ok=True)
    selected = frames[matched_pos - previous_frames : matched_pos + 1]
    for offset, source in zip(range(-previous_frames, 1), selected, strict=True):
        destination = sequence_dir / f"frame_{offset:+04d}{source.suffix.lower()}"
        if not destination.exists() or overwrite:
            shutil.copy2(source, destination)
    return sequence_dir


def match_scene(args: argparse.Namespace, scene_id: str) -> MatchResult | None:
    scene_dir = args.scenes_root / scene_id
    frames_dir = find_frames_dir(scene_dir, args.frames_subdir)
    keyframe_path = find_keyframe(args.keyframes_root, scene_id)
    if frames_dir is None:
        logging.warning("%s: frames dir not found", scene_id)
        return None
    if keyframe_path is None:
        logging.warning("%s: keyframe image not found", scene_id)
        return None
    frames = list_frames(frames_dir)
    if not frames:
        logging.warning("%s: no frames under %s", scene_id, frames_dir)
        return None

    keyframe_bgr = read_resized_bgr(keyframe_path, args.resize_width, args.resize_height)
    best: tuple[float, float, float, float, int, Path] | None = None
    for pos, frame_path in candidate_frames(frames, args.candidate_window):
        frame_bgr = read_resized_bgr(frame_path, args.resize_width, args.resize_height)
        score, diff, hist, orb = compare_images(frame_bgr, keyframe_bgr)
        if best is None or score > best[0]:
            best = (score, diff, hist, orb, pos, frame_path)
    if best is None:
        return None

    score, diff, hist, orb, matched_pos, matched_frame = best
    if score < args.min_score:
        logging.warning("%s: best score %.4f < min-score %.4f", scene_id, score, args.min_score)
        return None

    sequence_dir = copy_sequence(
        frames=frames,
        matched_pos=matched_pos,
        previous_frames=args.previous_frames,
        output_root=args.output_root,
        scene_id=scene_id,
        overwrite=args.overwrite,
    )
    if args.copy_keyframe_image:
        destination = args.output_root / scene_id / f"bdd_keyframe{keyframe_path.suffix.lower()}"
        if not destination.exists() or args.overwrite:
            shutil.copy2(keyframe_path, destination)

    return MatchResult(
        scene_id=scene_id,
        keyframe_image=keyframe_path,
        matched_frame=matched_frame,
        matched_index=frame_numeric_index(matched_frame, matched_pos),
        score=score,
        absdiff_score=diff,
        hist_score=hist,
        orb_score=orb,
        sequence_dir=sequence_dir,
    )


def write_manifest(output_root: Path, rows: list[MatchResult]) -> None:
    manifest_path = output_root / "matched_keyframes.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene_id",
        "keyframe_image",
        "matched_frame",
        "matched_index",
        "score",
        "absdiff_score",
        "hist_score",
        "orb_score",
        "sequence_dir",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "scene_id": row.scene_id,
                    "keyframe_image": row.keyframe_image,
                    "matched_frame": row.matched_frame,
                    "matched_index": row.matched_index,
                    "score": f"{row.score:.6f}",
                    "absdiff_score": f"{row.absdiff_score:.6f}",
                    "hist_score": f"{row.hist_score:.6f}",
                    "orb_score": f"{row.orb_score:.6f}",
                    "sequence_dir": row.sequence_dir,
                }
            )
    logging.info("manifest: %s", manifest_path)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    args.output_root.mkdir(parents=True, exist_ok=True)
    scene_ids = collect_scene_ids(args)
    logging.info("scene count: %d", len(scene_ids))

    rows: list[MatchResult] = []
    for scene_id in scene_ids:
        try:
            result = match_scene(args, scene_id)
        except Exception as exc:  # noqa: BLE001
            logging.warning("%s: failed: %s", scene_id, exc)
            continue
        if result is not None:
            rows.append(result)
            logging.info(
                "%s: matched %s score=%.4f",
                scene_id,
                result.matched_frame.name,
                result.score,
            )
    write_manifest(args.output_root, rows)
    logging.info("done: %d/%d scenes matched", len(rows), len(scene_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
