from typing import Tuple

import cv2
import numpy as np


def warp_mask_with_flow(previous_mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Align previous free-space mask to current image coordinates.

    previous_mask: float32 HxW, values in [0, 1].
    flow: float32 HxWx2, flow[..., 0]=dx and flow[..., 1]=dy.
    """
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
    return cv2.remap(
        previous_mask.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


def fuse_masks(
    current_mask: np.ndarray,
    warped_mask: np.ndarray,
    alpha: float = 0.7,
) -> np.ndarray:
    """Fuse current PIDNet mask and warped historical mask."""
    if current_mask.shape != warped_mask.shape:
        raise ValueError("current_mask and warped_mask shapes must match")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    fused = alpha * current_mask.astype(np.float32)
    fused += (1.0 - alpha) * warped_mask.astype(np.float32)
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def occupancy_grid_size(
    forward_range_m: float = 20.0,
    side_range_m: float = 10.0,
    resolution_m: float = 0.1,
) -> Tuple[int, int]:
    """Return OccupancyGrid width and height in cells."""
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive")
    width = int(round((side_range_m * 2.0) / resolution_m))
    height = int(round(forward_range_m / resolution_m))
    return width, height
