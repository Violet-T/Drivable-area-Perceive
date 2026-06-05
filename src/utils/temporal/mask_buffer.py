"""Multi-frame mask history buffer."""

from __future__ import annotations

import numpy as np

from utils.temporal.warp import warp_mask_with_flow


class MaskBuffer:
    """保存已经对齐到当前帧坐标系的历史 fused mask。"""

    def __init__(self, max_history: int) -> None:
        if max_history < 1:
            raise ValueError("max_history must be >= 1")
        self.max_history = max_history
        self._masks: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self._masks)

    @property
    def masks(self) -> list[np.ndarray]:
        return self._masks

    def clear(self) -> None:
        self._masks.clear()

    def warp_all(self, flow: np.ndarray) -> None:
        """用 Flow_{t-1->t} 将所有历史 mask 推进到当前帧坐标系。"""
        self._masks = [warp_mask_with_flow(mask, flow) for mask in self._masks]

    def append(self, mask: np.ndarray) -> None:
        """把当前 fused mask 放入 buffer；列表顺序为 newest -> oldest。"""
        if mask.ndim != 2:
            raise ValueError("mask must have shape HxW")
        self._masks.insert(0, np.clip(mask.astype(np.float32), 0.0, 1.0))
        if len(self._masks) > self.max_history:
            self._masks = self._masks[: self.max_history]


def default_history_weights(count: int, decay: float) -> np.ndarray:
    """生成 newest -> oldest 的指数衰减权重，并归一化。"""
    if count <= 0:
        return np.zeros((0,), dtype=np.float32)
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must be in (0, 1]")
    weights = np.array([decay**idx for idx in range(count)], dtype=np.float32)
    return weights / weights.sum()


def combine_history_masks(history_masks: list[np.ndarray], decay: float = 0.6) -> np.ndarray | None:
    """将多帧历史 mask 合成为单个历史参考 mask。"""
    if not history_masks:
        return None
    weights = default_history_weights(len(history_masks), decay)
    combined = np.zeros_like(history_masks[0], dtype=np.float32)
    for weight, mask in zip(weights, history_masks):
        combined += float(weight) * mask.astype(np.float32)
    return np.clip(combined, 0.0, 1.0).astype(np.float32)
