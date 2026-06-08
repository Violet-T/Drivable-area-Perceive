"""Train STGRU temporal fusion for binary free-space masks.

The training script supports two data modes:

1. Precomputed samples listed by CSV. This is the recommended mode for final
   experiments because the inputs match the real YOLOP + SEA-RAFT pipeline.
2. Cityscapes sequence + gtFine bootstrap mode. This uses annotated Cityscapes
   frames as targets and creates synthetic current/history masks for structure
   pretraining or smoke testing.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from STGRU.stgru_module import STGRUCell  # noqa: E402


@dataclass
class TrainStats:
    loss: float
    bce_loss: float
    dice_loss: float
    iou: float


@dataclass
class EvalStats:
    iou: float
    precision: float
    recall: float
    f1: float
    accuracy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train STGRU free-space temporal fusion")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--cityscapes-root", default=str(PROJECT_ROOT / "data" / "cityscapes"))
    parser.add_argument("--sample-list", default="", help="CSV for precomputed train samples")
    parser.add_argument("--val-sample-list", default="", help="CSV for precomputed val samples")
    parser.add_argument("--test-sample-list", default="", help="CSV for precomputed test samples")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "weights" / "STGRU"))
    parser.add_argument("--checkpoint", default="", help="Resume from STGRU checkpoint")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--free-label-ids", default="7", help="Cityscapes labelIds treated as free-space")
    parser.add_argument("--target-is-cityscapes-label", action="store_true")
    parser.add_argument("--previous-frame-offset", type=int, default=1)
    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--image-height", type=int, default=540)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-name", default="stgru_best.pth")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_label_ids(value: str) -> list[int]:
    ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not ids:
        raise ValueError("--free-label-ids must contain at least one label id")
    return ids


def resize_array(array: np.ndarray, width: int, height: int, interpolation: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        return array
    if array.shape[:2] == (height, width):
        return array
    return cv2.resize(array, (width, height), interpolation=interpolation)


def load_prob_mask(path: Path, width: int, height: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npy":
        mask = np.load(path).astype(np.float32)
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        mask = mask.astype(np.float32)
        if mask.max() > 1.0:
            mask /= 255.0
    mask = resize_array(mask, width, height, cv2.INTER_LINEAR)
    return np.clip(mask.astype(np.float32), 0.0, 1.0)


def load_cityscapes_target(
    path: Path,
    free_label_ids: list[int],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    label = resize_array(label, width, height, cv2.INTER_NEAREST)
    target = np.isin(label, np.asarray(free_label_ids, dtype=label.dtype)).astype(np.float32)
    valid = (label != 255).astype(np.float32)
    return target, valid


def load_binary_target(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    target = load_prob_mask(path, width, height)
    valid = np.ones_like(target, dtype=np.float32)
    return (target >= 0.5).astype(np.float32), valid


def two_class_probs(mask: np.ndarray) -> torch.Tensor:
    free = np.clip(mask.astype(np.float32), 0.0, 1.0)
    non_free = 1.0 - free
    return torch.from_numpy(np.stack([non_free, free], axis=0))


def image_photometric_error(current_path: Path, previous_path: Path, width: int, height: int) -> np.ndarray:
    current = cv2.imread(str(current_path), cv2.IMREAD_COLOR)
    previous = cv2.imread(str(previous_path), cv2.IMREAD_COLOR)
    if current is None or previous is None:
        return np.zeros((height, width), dtype=np.float32)
    current = resize_array(current, width, height, cv2.INTER_LINEAR)
    previous = resize_array(previous, width, height, cv2.INTER_LINEAR)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return np.abs(current_gray - previous_gray).astype(np.float32)


def random_shift(mask: np.ndarray, max_shift: int = 8) -> np.ndarray:
    dx = int(np.random.randint(-max_shift, max_shift + 1))
    dy = int(np.random.randint(-max_shift, max_shift + 1))
    height, width = mask.shape
    matrix = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    shifted = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=0.0)
    return np.clip(shifted, 0.0, 1.0).astype(np.float32)


def corrupt_mask(mask: np.ndarray) -> np.ndarray:
    """Create YOLOP-like mask noise for Cityscapes bootstrap training."""
    noisy = mask.astype(np.float32).copy()
    height, width = noisy.shape

    for _ in range(int(np.random.randint(1, 5))):
        rect_w = int(np.random.randint(max(8, width // 40), max(16, width // 8)))
        rect_h = int(np.random.randint(max(8, height // 40), max(16, height // 8)))
        x0 = int(np.random.randint(0, max(1, width - rect_w)))
        y0 = int(np.random.randint(height // 3, max(height // 3 + 1, height - rect_h)))
        value = float(np.random.choice([0.0, 1.0], p=[0.75, 0.25]))
        noisy[y0 : y0 + rect_h, x0 : x0 + rect_w] = value

    if np.random.rand() < 0.7:
        kernel = int(np.random.choice([3, 5, 7]))
        noisy = cv2.GaussianBlur(noisy, (kernel, kernel), 0)
    noisy += np.random.normal(0.0, 0.04, size=noisy.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


class PrecomputedSTGRUDataset(Dataset):
    """Dataset reading real YOLOP + SEA-RAFT training samples from CSV."""

    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path,
        width: int,
        height: int,
        free_label_ids: list[int],
        target_is_cityscapes_label: bool,
        max_samples: int = 0,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root)
        self.width = width
        self.height = height
        self.free_label_ids = free_label_ids
        self.target_is_cityscapes_label = target_is_cityscapes_label
        with self.csv_path.open("r", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if max_samples > 0:
            self.rows = self.rows[:max_samples]
        required = {"current_mask", "warped_mask", "target_mask"}
        missing = required - set(self.rows[0].keys()) if self.rows else required
        if missing:
            raise ValueError(f"{self.csv_path} missing required columns: {sorted(missing)}")

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.data_root / path

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        current_mask = load_prob_mask(self._resolve(row["current_mask"]), self.width, self.height)
        warped_mask = load_prob_mask(self._resolve(row["warped_mask"]), self.width, self.height)

        target_path = self._resolve(row["target_mask"])
        if self.target_is_cityscapes_label:
            target, valid = load_cityscapes_target(target_path, self.free_label_ids, self.width, self.height)
        else:
            target, valid = load_binary_target(target_path, self.width, self.height)

        if row.get("photometric_error"):
            photo_error = load_prob_mask(self._resolve(row["photometric_error"]), self.width, self.height)
        elif row.get("current_image") and row.get("warped_previous_image"):
            photo_error = image_photometric_error(
                self._resolve(row["current_image"]),
                self._resolve(row["warped_previous_image"]),
                self.width,
                self.height,
            )
        else:
            photo_error = np.abs(current_mask - warped_mask).astype(np.float32)

        return {
            "current_probs": two_class_probs(current_mask),
            "warped_history_probs": two_class_probs(warped_mask),
            "photometric_error": torch.from_numpy(photo_error[None, ...].astype(np.float32)),
            "target": torch.from_numpy(target[None, ...].astype(np.float32)),
            "valid": torch.from_numpy(valid[None, ...].astype(np.float32)),
        }


class CityscapesSequenceBootstrapDataset(Dataset):
    """Cityscapes gtFine supervised bootstrap data for STGRU structure training."""

    def __init__(
        self,
        cityscapes_root: str | Path,
        split: str,
        width: int,
        height: int,
        free_label_ids: list[int],
        previous_frame_offset: int = 1,
        max_samples: int = 0,
    ) -> None:
        self.root = Path(cityscapes_root)
        self.split = split
        self.width = width
        self.height = height
        self.free_label_ids = free_label_ids
        self.previous_frame_offset = previous_frame_offset
        self.labels = sorted((self.root / "gtFine" / split).glob("*/*_gtFine_labelIds.png"))
        if max_samples > 0:
            self.labels = self.labels[:max_samples]
        if not self.labels:
            raise FileNotFoundError(f"No gtFine labelIds found under {self.root / 'gtFine' / split}")

    def __len__(self) -> int:
        return len(self.labels)

    def _image_paths(self, label_path: Path) -> tuple[Path, Path]:
        city = label_path.parent.name
        base = label_path.name.replace("_gtFine_labelIds.png", "")
        parts = base.split("_")
        frame_id = int(parts[-1])
        previous_base = "_".join(parts[:-1] + [f"{max(0, frame_id - self.previous_frame_offset):06d}"])

        sequence_dir = self.root / "leftImg8bit_sequence" / self.split / city
        current = sequence_dir / f"{base}_leftImg8bit.png"
        previous = sequence_dir / f"{previous_base}_leftImg8bit.png"
        if not current.exists():
            current = self.root / "leftImg8bit" / self.split / city / f"{base}_leftImg8bit.png"
        if not previous.exists():
            previous = current
        return current, previous

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        label_path = self.labels[index]
        target, valid = load_cityscapes_target(label_path, self.free_label_ids, self.width, self.height)
        current_image, previous_image = self._image_paths(label_path)

        # 形状 HxW。这里是结构预训练/调试用合成输入，最终实验建议改用真实 YOLOP + SEA-RAFT 预计算 CSV。
        current_mask = corrupt_mask(target)
        warped_mask = corrupt_mask(random_shift(target))
        photo_error = image_photometric_error(current_image, previous_image, self.width, self.height)
        if photo_error.shape != target.shape:
            photo_error = np.zeros_like(target, dtype=np.float32)

        return {
            "current_probs": two_class_probs(current_mask),
            "warped_history_probs": two_class_probs(warped_mask),
            "photometric_error": torch.from_numpy(photo_error[None, ...].astype(np.float32)),
            "target": torch.from_numpy(target[None, ...].astype(np.float32)),
            "valid": torch.from_numpy(valid[None, ...].astype(np.float32)),
        }


def build_precomputed_dataset(
    csv_path: str,
    args: argparse.Namespace,
    free_label_ids: list[int],
    max_samples: int,
) -> PrecomputedSTGRUDataset | None:
    if not csv_path:
        return None
    return PrecomputedSTGRUDataset(
        csv_path=csv_path,
        data_root=args.data_root,
        width=args.image_width,
        height=args.image_height,
        free_label_ids=free_label_ids,
        target_is_cityscapes_label=args.target_is_cityscapes_label,
        max_samples=max_samples,
    )


def build_datasets(args: argparse.Namespace, free_label_ids: list[int]) -> tuple[Dataset, Dataset | None, Dataset | None]:
    if args.sample_list:
        train_dataset = build_precomputed_dataset(args.sample_list, args, free_label_ids, args.max_train_samples)
        val_dataset = build_precomputed_dataset(args.val_sample_list, args, free_label_ids, args.max_val_samples)
        test_dataset = build_precomputed_dataset(args.test_sample_list, args, free_label_ids, args.max_test_samples)
        if train_dataset is None:
            raise ValueError("--sample-list did not create a training dataset")
        return train_dataset, val_dataset, test_dataset

    train_dataset = CityscapesSequenceBootstrapDataset(
        cityscapes_root=args.cityscapes_root,
        split=args.train_split,
        width=args.image_width,
        height=args.image_height,
        free_label_ids=free_label_ids,
        previous_frame_offset=args.previous_frame_offset,
        max_samples=args.max_train_samples,
    )
    val_dataset = CityscapesSequenceBootstrapDataset(
        cityscapes_root=args.cityscapes_root,
        split=args.val_split,
        width=args.image_width,
        height=args.image_height,
        free_label_ids=free_label_ids,
        previous_frame_offset=args.previous_frame_offset,
        max_samples=args.max_val_samples,
    )
    return train_dataset, val_dataset, None


def masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return (value * valid).sum() / valid.sum().clamp_min(1.0)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    pred = pred * valid
    target = target * valid
    intersection = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1.0) / (denom + 1.0)
    return 1.0 - dice.mean()


def batch_iou(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    pred_bin = ((pred >= 0.5).float() * valid).bool()
    target_bin = ((target >= 0.5).float() * valid).bool()
    intersection = (pred_bin & target_bin).sum(dim=(1, 2, 3)).float()
    union = (pred_bin | target_bin).sum(dim=(1, 2, 3)).float()
    return ((intersection + 1.0) / (union + 1.0)).mean()


@torch.no_grad()
def evaluate_binary_metrics(
    model: STGRUCell,
    loader: DataLoader,
    device: torch.device,
    desc: str,
) -> EvalStats:
    model.eval()
    tp = fp = fn = tn = 0.0
    for batch in tqdm(loader, desc=desc):
        current_probs = batch["current_probs"].to(device, non_blocking=True)
        warped_history_probs = batch["warped_history_probs"].to(device, non_blocking=True)
        photometric_error = batch["photometric_error"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        fused_probs, _, _, _ = model(current_probs, warped_history_probs, photometric_error)
        pred = ((fused_probs[:, 1:2] >= 0.5).float() * valid).bool()
        gt = ((target >= 0.5).float() * valid).bool()
        valid_bool = valid.bool()
        tp += float((pred & gt & valid_bool).sum().cpu())
        fp += float((pred & ~gt & valid_bool).sum().cpu())
        fn += float((~pred & gt & valid_bool).sum().cpu())
        tn += float((~pred & ~gt & valid_bool).sum().cpu())

    eps = 1.0e-8
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    return EvalStats(iou=iou, precision=precision, recall=recall, f1=f1, accuracy=accuracy)


def run_epoch(
    model: STGRUCell,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    dice_weight: float,
    use_amp: bool,
    desc: str,
) -> TrainStats:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    total_iou = 0.0
    sample_count = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and is_train)

    progress = tqdm(loader, desc=desc)
    for batch in progress:
        current_probs = batch["current_probs"].to(device, non_blocking=True)
        warped_history_probs = batch["warped_history_probs"].to(device, non_blocking=True)
        photometric_error = batch["photometric_error"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            fused_probs, _, _, _ = model(current_probs, warped_history_probs, photometric_error)
            pred_free = fused_probs[:, 1:2]
        # BCE on probabilities is unsafe under autocast, so compute losses in FP32.
        pred_free_for_loss = pred_free.float()
        target_for_loss = target.float()
        valid_for_loss = valid.float()
        with torch.cuda.amp.autocast(enabled=False):
            bce = masked_mean(
                F.binary_cross_entropy(pred_free_for_loss, target_for_loss, reduction="none"),
                valid_for_loss,
            )
            dl = dice_loss(pred_free_for_loss, target_for_loss, valid_for_loss)
            loss = bce + dice_weight * dl

        if is_train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            iou = batch_iou(pred_free.detach(), target, valid)
        batch_size = current_probs.shape[0]
        sample_count += batch_size
        total_loss += float(loss.detach().cpu()) * batch_size
        total_bce += float(bce.detach().cpu()) * batch_size
        total_dice += float(dl.detach().cpu()) * batch_size
        total_iou += float(iou.detach().cpu()) * batch_size
        progress.set_postfix(loss=total_loss / sample_count, iou=total_iou / sample_count)

    denom = max(sample_count, 1)
    return TrainStats(
        loss=total_loss / denom,
        bce_loss=total_bce / denom,
        dice_loss=total_dice / denom,
        iou=total_iou / denom,
    )


def save_checkpoint(
    path: Path,
    model: STGRUCell,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    args: argparse.Namespace,
    free_label_ids: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_iou": best_metric,
            "free_label_ids": free_label_ids,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    free_label_ids = parse_label_ids(args.free_label_ids)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_dataset, val_dataset, test_dataset = build_datasets(args, free_label_ids)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

    model = STGRUCell(hidden_channels=args.hidden_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 1
    best_val_iou = -1.0

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=True)
        if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1 if isinstance(checkpoint, dict) else 1
        best_val_iou = float(checkpoint.get("best_val_iou", -1.0)) if isinstance(checkpoint, dict) else -1.0

    metrics_path = output_dir / "training_log.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "split", "loss", "bce_loss", "dice_loss", "iou"])
        for epoch in range(start_epoch, args.epochs + 1):
            train_stats = run_epoch(
                model,
                train_loader,
                optimizer=optimizer,
                device=device,
                dice_weight=args.dice_weight,
                use_amp=args.amp and device.type == "cuda",
                desc=f"train:{epoch}",
            )
            writer.writerow([epoch, "train", train_stats.loss, train_stats.bce_loss, train_stats.dice_loss, train_stats.iou])
            handle.flush()

            if val_loader is not None:
                with torch.no_grad():
                    val_stats = run_epoch(
                        model,
                        val_loader,
                        optimizer=None,
                        device=device,
                        dice_weight=args.dice_weight,
                        use_amp=False,
                        desc=f"val:{epoch}",
                    )
                writer.writerow([epoch, "val", val_stats.loss, val_stats.bce_loss, val_stats.dice_loss, val_stats.iou])
                handle.flush()
                metric = val_stats.iou
            else:
                metric = train_stats.iou

            latest_path = output_dir / "stgru_latest.pth"
            save_checkpoint(latest_path, model, optimizer, epoch, best_val_iou, args, free_label_ids)
            if metric > best_val_iou:
                best_val_iou = metric
                save_checkpoint(output_dir / args.save_name, model, optimizer, epoch, best_val_iou, args, free_label_ids)

    metadata = {
        "data_mode": "precomputed_csv" if args.sample_list else "cityscapes_sequence_bootstrap",
        "train_size": len(train_dataset),
        "val_size": len(val_dataset) if val_dataset is not None else 0,
        "test_size": len(test_dataset) if test_dataset is not None else 0,
        "free_label_ids": free_label_ids,
        "best_val_iou": best_val_iou,
        "args": vars(args),
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    eval_path = output_dir / "evaluation_summary.csv"
    best_checkpoint_path = output_dir / args.save_name
    if best_checkpoint_path.exists() and (val_loader is not None or test_loader is not None):
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        with eval_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["split", "iou", "precision", "recall", "f1", "accuracy"])
            if val_loader is not None:
                stats = evaluate_binary_metrics(model, val_loader, device, desc="eval:val")
                writer.writerow(["val", stats.iou, stats.precision, stats.recall, stats.f1, stats.accuracy])
            if test_loader is not None:
                stats = evaluate_binary_metrics(model, test_loader, device, desc="eval:test")
                writer.writerow(["test", stats.iou, stats.precision, stats.recall, stats.f1, stats.accuracy])
    print(f"best_checkpoint: {output_dir / args.save_name}")
    print(f"latest_checkpoint: {output_dir / 'stgru_latest.pth'}")
    print(f"training_log: {metrics_path}")
    if eval_path.exists():
        print(f"evaluation_summary: {eval_path}")


if __name__ == "__main__":
    main()
