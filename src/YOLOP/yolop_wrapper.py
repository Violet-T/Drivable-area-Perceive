"""YOLOP drivable-area inference wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def read_image_bgr(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise FileNotFoundError(path)
    return image


class YOLOPFreeSpaceWrapper:
    """Wrap YOLOP and expose only the drivable-area branch.

    Input image shape: OpenCV BGR HxWx3 uint8.
    Output mask shape: float32 HxW in [0, 1], where 1 means drivable/free-space.
    """

    def __init__(
        self,
        yolop_repo: str | Path,
        checkpoint_path: str | Path,
        img_size: int = 640,
        mask_mode: str = "probability",
        device: str = "cuda",
    ) -> None:
        self.yolop_repo = Path(yolop_repo).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.img_size = img_size
        self.mask_mode = mask_mode
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        if self.mask_mode not in {"binary", "probability"}:
            raise ValueError("mask_mode must be 'binary' or 'probability'")
        if not self.yolop_repo.exists():
            raise FileNotFoundError(self.yolop_repo)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(self.checkpoint_path)

        sys.path.insert(0, str(self.yolop_repo))
        from lib.config import cfg  # pylint: disable=import-error,import-outside-toplevel
        from lib.models import get_net  # pylint: disable=import-error,import-outside-toplevel
        from lib.utils import letterbox_for_img  # pylint: disable=import-error,import-outside-toplevel

        self.cfg = cfg
        self.letterbox_for_img = letterbox_for_img
        self.model = get_net(cfg)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device).eval()
        self.use_half = self.device.type == "cuda"
        if self.use_half:
            self.model.half()
        print(f"Loaded YOLOP checkpoint from {self.checkpoint_path}")

    def preprocess(self, image_bgr: np.ndarray) -> tuple[torch.Tensor, tuple[int, int], tuple[float, float], tuple[float, float]]:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must have shape HxWx3")
        original_hw = image_bgr.shape[:2]
        # YOLOP demo keeps BGR channel order and applies ImageNet normalization.
        letterboxed, ratio, pad = self.letterbox_for_img(image_bgr, new_shape=self.img_size, auto=True)
        tensor = transforms.ToTensor()(letterboxed)
        tensor = NORMALIZE(tensor).unsqueeze(0).to(self.device)
        tensor = tensor.half() if self.use_half else tensor.float()
        return tensor, original_hw, ratio, pad

    @torch.no_grad()
    def infer_drivable_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return YOLOP drivable-area mask without detection boxes or lane rendering."""
        tensor, original_hw, _ratio, pad = self.preprocess(image_bgr)
        _det_out, da_seg_out, _ll_seg_out = self.model(tensor)

        _, _, padded_h, padded_w = tensor.shape
        pad_w, pad_h = int(round(pad[0])), int(round(pad[1]))
        y0, y1 = pad_h, padded_h - pad_h
        x0, x1 = pad_w, padded_w - pad_w
        if y1 <= y0 or x1 <= x0:
            raise RuntimeError("Invalid YOLOP letterbox crop after inference.")

        da_predict = da_seg_out[:, :, y0:y1, x0:x1]
        da_predict = F.interpolate(da_predict.float(), size=original_hw, mode="bilinear", align_corners=False)
        if self.mask_mode == "probability":
            probs = torch.softmax(da_predict, dim=1)
            mask = probs[:, 1].squeeze(0)
            return mask.clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)

        pred = torch.argmax(da_predict, dim=1).squeeze(0)
        return pred.detach().cpu().numpy().astype(np.float32)


def mask_to_preview(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must have shape HxW")
    normalized = np.clip(mask.astype(np.float32), 0.0, 1.0)
    preview = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    preview[..., 1] = (normalized * 255.0).astype(np.uint8)
    return preview
