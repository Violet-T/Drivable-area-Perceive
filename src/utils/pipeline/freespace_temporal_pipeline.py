"""In-memory YOLOP + SEA-RAFT + temporal fusion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from SEA_RAFT.sea_raft_wrapper import SEARAFTWrapper
from STGRU.stgru_module import STGRUFusionModule, STGRUOutput
from utils.temporal.fusion import fuse_masks
from utils.temporal.mask_buffer import MaskBuffer, combine_history_masks
from utils.temporal.warp_image import warp_image_with_flow
from YOLOP.yolop_wrapper import YOLOPFreeSpaceWrapper


@dataclass
class PipelineResult:
    """All in-memory outputs produced for a single frame."""

    frame_id: int
    image_bgr: np.ndarray
    raw_mask: np.ndarray
    warped_mask: np.ndarray
    fused_mask: np.ndarray
    flow: np.ndarray | None
    update_gate: np.ndarray | None = None
    reset_gate: np.ndarray | None = None
    fusion_mode: str = "alpha"


class AlphaTemporalFusion:
    """State-light alpha fusion wrapper used as a no-training baseline."""

    def __init__(self, alpha: float = 0.7, non_free_threshold: float = 0.2) -> None:
        self.alpha = alpha
        self.non_free_threshold = non_free_threshold

    def fuse(self, current_mask: np.ndarray, warped_history_mask: np.ndarray) -> np.ndarray:
        return fuse_masks(
            current_mask,
            warped_history_mask,
            alpha=self.alpha,
            non_free_threshold=self.non_free_threshold,
        )


class FreeSpaceTemporalPipeline:
    """Compose YOLOP, SEA-RAFT and a temporal fusion module in one process."""

    def __init__(
        self,
        yolop_repo: str | Path,
        yolop_checkpoint: str | Path,
        sea_raft_repo: str | Path,
        sea_raft_config: str | Path,
        sea_raft_checkpoint: str | Path | None = None,
        sea_raft_url: str | None = "MemorySlices/Tartan-C-T-TSKH-spring540x960-S",
        stgru_checkpoint: str | Path | None = None,
        fusion_mode: str = "alpha",
        yolop_img_size: int = 640,
        mask_mode: str = "probability",
        alpha: float = 0.7,
        non_free_threshold: float = 0.2,
        history_size: int = 3,
        history_decay: float = 0.6,
        device: str = "cuda",
    ) -> None:
        if fusion_mode not in {"alpha", "stgru"}:
            raise ValueError("fusion_mode must be 'alpha' or 'stgru'")
        self.fusion_mode = fusion_mode
        self.history_decay = history_decay
        self.frame_id = 0
        self.previous_bgr: np.ndarray | None = None
        self.mask_buffer = MaskBuffer(max_history=history_size)

        self.yolop = YOLOPFreeSpaceWrapper(
            yolop_repo=yolop_repo,
            checkpoint_path=yolop_checkpoint,
            img_size=yolop_img_size,
            mask_mode=mask_mode,
            device=device,
        )
        self.sea_raft = SEARAFTWrapper(
            sea_raft_repo=sea_raft_repo,
            config_path=sea_raft_config,
            checkpoint_path=sea_raft_checkpoint,
            model_url=sea_raft_url if sea_raft_checkpoint is None else None,
            device=device,
        )
        self.alpha_fusion = AlphaTemporalFusion(alpha=alpha, non_free_threshold=non_free_threshold)
        self.stgru = (
            STGRUFusionModule(checkpoint_path=stgru_checkpoint, device=device)
            if fusion_mode == "stgru"
            else None
        )

    def reset(self) -> None:
        """Clear all recurrent state."""
        self.frame_id = 0
        self.previous_bgr = None
        self.mask_buffer.clear()

    def step(self, image_bgr: np.ndarray) -> PipelineResult:
        """Run one frame through YOLOP, SEA-RAFT and temporal fusion."""
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must have shape HxWx3")

        raw_mask = self.yolop.infer_drivable_mask(image_bgr)
        flow: np.ndarray | None = None
        update_gate: np.ndarray | None = None
        reset_gate: np.ndarray | None = None

        if self.previous_bgr is None or len(self.mask_buffer) == 0:
            warped_mask = np.zeros_like(raw_mask, dtype=np.float32)
            fused_mask = raw_mask.copy()
        else:
            flow = self.sea_raft.infer_flow(self.previous_bgr, image_bgr)
            self.mask_buffer.warp_all(flow)
            warped_mask = combine_history_masks(self.mask_buffer.masks, decay=self.history_decay)
            if warped_mask is None:
                warped_mask = np.zeros_like(raw_mask, dtype=np.float32)

            if self.fusion_mode == "stgru":
                if self.stgru is None:
                    raise RuntimeError("STGRU fusion mode selected but STGRU module is not initialized.")
                warped_previous_bgr = warp_image_with_flow(self.previous_bgr, flow)
                stgru_output: STGRUOutput = self.stgru.fuse(
                    current_mask=raw_mask,
                    warped_history_mask=warped_mask,
                    current_bgr=image_bgr,
                    warped_previous_bgr=warped_previous_bgr,
                )
                fused_mask = stgru_output.fused_mask
                update_gate = stgru_output.update_gate
                reset_gate = stgru_output.reset_gate
            else:
                fused_mask = self.alpha_fusion.fuse(raw_mask, warped_mask)

        result = PipelineResult(
            frame_id=self.frame_id,
            image_bgr=image_bgr,
            raw_mask=raw_mask.astype(np.float32),
            warped_mask=warped_mask.astype(np.float32),
            fused_mask=fused_mask.astype(np.float32),
            flow=flow,
            update_gate=update_gate,
            reset_gate=reset_gate,
            fusion_mode=self.fusion_mode,
        )
        self.previous_bgr = image_bgr
        self.mask_buffer.append(result.fused_mask)
        self.frame_id += 1
        return result


def resize_frame_if_needed(image_bgr: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize frame for a pipeline run when both target dimensions are set."""
    if target_width <= 0 and target_height <= 0:
        return image_bgr
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target-width and target-height must be set together.")
    if image_bgr.shape[1] == target_width and image_bgr.shape[0] == target_height:
        return image_bgr
    return cv2.resize(image_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)
