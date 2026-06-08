"""Spatio-temporal gated recurrent fusion for binary free-space masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


@dataclass
class STGRUOutput:
    """STGRU output tensors converted to numpy arrays."""

    fused_mask: np.ndarray
    update_gate: np.ndarray
    reset_gate: np.ndarray
    candidate_free: np.ndarray


class STGRUCell(nn.Module):
    """A compact STGRU cell adapted to binary free-space probability maps.

    Inputs are current two-class probabilities, warped historical two-class
    probabilities, and a one-channel photometric error map. The cell follows
    the paper's idea: estimate flow reliability with a reset gate, compute a
    candidate state from current and reliable history, then blend it with the
    warped history using an update gate.
    """

    def __init__(self, channels: int = 2, hidden_channels: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        if channels != 2:
            raise ValueError("This project STGRU implementation currently expects two classes.")
        padding = kernel_size // 2
        self.reset_from_error = nn.Conv2d(1, channels, kernel_size=kernel_size, padding=padding)
        self.current_proj = nn.Conv2d(channels, hidden_channels, kernel_size=kernel_size, padding=padding)
        self.history_proj = nn.Conv2d(channels, hidden_channels, kernel_size=kernel_size, padding=padding)
        self.candidate_head = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.update_gate = nn.Conv2d(channels * 2, channels, kernel_size=kernel_size, padding=padding)
        self.logit_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self._init_parameters()

    def _init_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        current_probs: torch.Tensor,
        warped_history_probs: torch.Tensor,
        photometric_error: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fused two-class probabilities and diagnostic gates.

        Shapes:
        - current_probs: Bx2xHxW
        - warped_history_probs: Bx2xHxW
        - photometric_error: Bx1xHxW, normalized to [0, 1]
        """
        if current_probs.shape != warped_history_probs.shape:
            raise ValueError("current_probs and warped_history_probs shapes must match")
        if current_probs.ndim != 4 or current_probs.shape[1] != 2:
            raise ValueError("current_probs must have shape Bx2xHxW")
        if photometric_error.shape[0] != current_probs.shape[0] or photometric_error.shape[1] != 1:
            raise ValueError("photometric_error must have shape Bx1xHxW")
        if photometric_error.shape[-2:] != current_probs.shape[-2:]:
            raise ValueError("photometric_error spatial size must match current_probs")

        reset_gate = 1.0 - torch.tanh(torch.abs(self.reset_from_error(photometric_error)))
        reset_gate = torch.clamp(reset_gate, 0.0, 1.0)
        reliable_history = reset_gate * warped_history_probs

        features = torch.relu(self.current_proj(current_probs) + self.history_proj(reliable_history))
        candidate_logits = self.candidate_head(features)
        candidate_probs = torch.softmax(candidate_logits, dim=1)

        update_gate = torch.sigmoid(self.update_gate(torch.cat([current_probs, warped_history_probs], dim=1)))
        # current_probs / warped_history_probs / candidate_probs are already probabilities.
        # Applying another softmax here would lift low-confidence pixels toward 0.5,
        # which can create large false free-space regions after thresholding.
        fused_probs = (1.0 - update_gate) * warped_history_probs + update_gate * candidate_probs
        fused_probs = fused_probs / fused_probs.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        return fused_probs, update_gate, reset_gate, candidate_probs


class STGRUFusionModule:
    """Numpy-facing wrapper around STGRUCell for the project pipeline."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cuda",
        hidden_channels: int = 16,
        non_free_threshold: float = 0.2,
        support_margin: float = 0.03,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = STGRUCell(hidden_channels=hidden_channels).to(self.device).eval()
        self.checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path else None
        self.non_free_threshold = non_free_threshold
        self.support_margin = support_margin
        self.is_trained = False
        if self.checkpoint_path is not None:
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(self.checkpoint_path)
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            self.model.load_state_dict(state_dict, strict=True)
            self.is_trained = True

    @staticmethod
    def _mask_to_two_class(mask: np.ndarray) -> torch.Tensor:
        if mask.ndim != 2:
            raise ValueError("mask must have shape HxW")
        free = np.clip(mask.astype(np.float32), 0.0, 1.0)
        non_free = 1.0 - free
        stacked = np.stack([non_free, free], axis=0)
        return torch.from_numpy(stacked).unsqueeze(0)

    @staticmethod
    def _photometric_error(current_bgr: np.ndarray, warped_previous_bgr: np.ndarray) -> torch.Tensor:
        if current_bgr.shape != warped_previous_bgr.shape:
            raise ValueError("current_bgr and warped_previous_bgr shapes must match")
        current_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        previous_gray = cv2.cvtColor(warped_previous_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        error = np.abs(current_gray - previous_gray)[None, None, ...]
        return torch.from_numpy(error.astype(np.float32))

    @torch.no_grad()
    def fuse(
        self,
        current_mask: np.ndarray,
        warped_history_mask: np.ndarray,
        current_bgr: np.ndarray,
        warped_previous_bgr: np.ndarray,
    ) -> STGRUOutput:
        """Fuse current and warped historical free-space masks.

        Returns HxW float32 arrays in [0, 1]. `fused_mask` is the free-space
        channel of the two-class STGRU output.
        """
        current = self._mask_to_two_class(current_mask).to(self.device)
        history = self._mask_to_two_class(warped_history_mask).to(self.device)
        error = self._photometric_error(current_bgr, warped_previous_bgr).to(self.device)
        fused, update_gate, reset_gate, candidate = self.model(current, history, error)
        fused_free = fused[0, 1].detach().cpu().numpy().astype(np.float32)
        # STGRU is a temporal fusion module, not a semantic generator. If neither
        # the current YOLOP mask nor the warped history supports free-space, the
        # fused result must not hallucinate a new free-space region.
        current_free = np.clip(current_mask.astype(np.float32), 0.0, 1.0)
        history_free = np.clip(warped_history_mask.astype(np.float32), 0.0, 1.0)
        # Give the current frame a small tolerance, but do not keep adding that
        # tolerance to recurrent history; otherwise a low background score can
        # drift upward frame by frame until it crosses the visualization threshold.
        support_upper_bound = np.clip(np.maximum(current_free + self.support_margin, history_free), 0.0, 1.0)
        fused_free = np.minimum(fused_free, support_upper_bound)
        fused_free = np.where(current_free < self.non_free_threshold, current_free, fused_free)
        return STGRUOutput(
            fused_mask=fused_free.astype(np.float32),
            update_gate=update_gate[0, 1].detach().cpu().numpy().astype(np.float32),
            reset_gate=reset_gate[0, 1].detach().cpu().numpy().astype(np.float32),
            candidate_free=candidate[0, 1].detach().cpu().numpy().astype(np.float32),
        )
