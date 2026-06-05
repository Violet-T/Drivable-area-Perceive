"""Mask warping utilities."""

from __future__ import annotations

import cv2
import numpy as np


def warp_mask_with_flow(previous_mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Align previous_mask to current image coordinates using Flow_{t-1->t}."""
    if previous_mask.ndim != 2:
        raise ValueError("previous_mask must have shape HxW")
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("flow must have shape HxWx2")
    if previous_mask.shape != flow.shape[:2]:
        raise ValueError("previous_mask and flow spatial sizes must match")

    height, width = previous_mask.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x - flow[..., 0].astype(np.float32)
    map_y = grid_y - flow[..., 1].astype(np.float32)
    warped = cv2.remap(
        previous_mask.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return np.clip(warped, 0.0, 1.0).astype(np.float32)
