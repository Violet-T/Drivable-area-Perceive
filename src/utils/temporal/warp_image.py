"""Image warping utilities."""

from __future__ import annotations

import cv2
import numpy as np


def warp_image_with_flow(previous_bgr: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Align previous BGR frame to current frame coordinates using Flow_{t-1->t}.

    previous_bgr: uint8 HxWx3.
    flow: float32 HxWx2, flow[...,0]=dx, flow[...,1]=dy.
    """
    if previous_bgr.ndim != 3 or previous_bgr.shape[2] != 3:
        raise ValueError("previous_bgr must have shape HxWx3")
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("flow must have shape HxWx2")
    if previous_bgr.shape[:2] != flow.shape[:2]:
        raise ValueError("previous_bgr and flow spatial sizes must match")

    height, width = previous_bgr.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x - flow[..., 0].astype(np.float32)
    map_y = grid_y - flow[..., 1].astype(np.float32)
    return cv2.remap(
        previous_bgr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
