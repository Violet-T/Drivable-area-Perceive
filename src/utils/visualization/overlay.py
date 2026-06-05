"""OpenCV visualization helpers for free-space masks."""

from __future__ import annotations

import cv2
import numpy as np


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must have shape HxW")
    normalized = np.clip(mask.astype(np.float32), 0.0, 1.0)
    frame = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    frame[..., 1] = (normalized * 255.0).astype(np.uint8)
    return frame


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if image_bgr.shape[:2] != mask.shape:
        raise ValueError("image and mask spatial sizes must match")
    mask_color = mask_to_bgr(mask)
    return cv2.addWeighted(image_bgr, 1.0, mask_color, alpha, 0.0)


def make_temporal_panel(
    image_bgr: np.ndarray,
    raw_mask: np.ndarray,
    warped_mask: np.ndarray,
    fused_mask: np.ndarray,
) -> np.ndarray:
    raw_overlay = overlay_mask(image_bgr, raw_mask)
    fused_overlay = overlay_mask(image_bgr, fused_mask)
    warped_bgr = mask_to_bgr(warped_mask)
    raw_bgr = mask_to_bgr(raw_mask)
    top = cv2.hconcat([image_bgr, raw_overlay])
    bottom = cv2.hconcat([warped_bgr, fused_overlay])
    return cv2.vconcat([top, bottom])
