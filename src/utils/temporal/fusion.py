"""Temporal free-space mask fusion."""

from __future__ import annotations

import numpy as np


def fuse_masks(
    current_mask: np.ndarray,
    warped_mask: np.ndarray,
    alpha: float = 0.7,
    non_free_threshold: float = 0.2,
) -> np.ndarray:
    """Fuse current YOLOP mask and warped historical mask.

    current_mask, warped_mask: float32 HxW, values in [0, 1].
    Historical free-space is not allowed to override current non-free pixels.
    """
    if current_mask.shape != warped_mask.shape:
        raise ValueError("current_mask and warped_mask shapes must match")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if not 0.0 <= non_free_threshold <= 1.0:
        raise ValueError("non_free_threshold must be in [0, 1]")

    current = current_mask.astype(np.float32)
    warped = warped_mask.astype(np.float32)
    fused = alpha * current + (1.0 - alpha) * warped
    fused = np.where(current < non_free_threshold, current, fused)
    return np.clip(fused, 0.0, 1.0).astype(np.float32)
