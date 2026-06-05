"""SEA-RAFT optical-flow inference wrapper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class SEARAFTWrapper:
    """Wrap SEA-RAFT and return flow as float32 HxWx2."""

    def __init__(
        self,
        sea_raft_repo: str | Path,
        config_path: str | Path,
        checkpoint_path: str | Path | None = None,
        model_url: str | None = None,
        device: str = "cuda",
    ) -> None:
        self.sea_raft_repo = Path(sea_raft_repo).resolve()
        self.config_path = Path(config_path).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path else None
        self.model_url = model_url
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        if not self.sea_raft_repo.exists():
            raise FileNotFoundError(self.sea_raft_repo)
        if not self.config_path.exists():
            raise FileNotFoundError(self.config_path)
        if self.checkpoint_path is not None and not self.checkpoint_path.exists():
            raise FileNotFoundError(self.checkpoint_path)
        if self.checkpoint_path is None and not self.model_url:
            raise ValueError("Either checkpoint_path or model_url must be provided.")
        self.args = self._load_args()
        self.model = self._load_model()

    def _load_args(self) -> argparse.Namespace:
        with self.config_path.open() as config_file:
            config = json.load(config_file)
        return argparse.Namespace(**config)

    def _load_model(self) -> torch.nn.Module:
        core_path = self.sea_raft_repo / "core"
        sys.path.insert(0, str(self.sea_raft_repo))
        sys.path.insert(0, str(core_path))
        from raft import RAFT  # pylint: disable=import-error,import-outside-toplevel
        from utils.utils import load_ckpt  # pylint: disable=import-error,import-outside-toplevel

        if self.checkpoint_path is not None:
            model = RAFT(self.args)
            load_ckpt(model, str(self.checkpoint_path))
            print(f"Loaded SEA-RAFT checkpoint from {self.checkpoint_path}")
        else:
            model = RAFT.from_pretrained(self.model_url, args=self.args)
            print(f"Loaded SEA-RAFT model from {self.model_url}")
        return model.to(self.device).eval()

    def _preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must have shape HxWx3")
        # SEA-RAFT expects RGB CHW float tensor in [0, 255].
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def infer_flow(self, previous_bgr: np.ndarray, current_bgr: np.ndarray) -> np.ndarray:
        """Return Flow_{t-1->t} as float32 HxWx2, flow[...,0]=dx, flow[...,1]=dy."""
        if previous_bgr.shape != current_bgr.shape:
            raise ValueError("previous_bgr and current_bgr must have the same shape")
        image1 = self._preprocess(previous_bgr)
        image2 = self._preprocess(current_bgr)
        scale = 2 ** self.args.scale
        image1_scaled = F.interpolate(image1, scale_factor=scale, mode="bilinear", align_corners=False)
        image2_scaled = F.interpolate(image2, scale_factor=scale, mode="bilinear", align_corners=False)
        output = self.model(image1_scaled, image2_scaled, iters=self.args.iters, test_mode=True)
        flow = output["flow"][-1]
        flow = F.interpolate(flow, size=image1.shape[-2:], mode="bilinear", align_corners=False)
        flow = flow * (image1.shape[-1] / image1_scaled.shape[-1])
        return flow[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
