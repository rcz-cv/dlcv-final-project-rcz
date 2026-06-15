"""
nanodet.py

Preprocessor adapter to create new detections based on NanoDet / NanoDet-Plus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import os
import sys
from pathlib import Path

import numpy as np
import torch

from .base import Detection


COCO_PERSON_CLASS_ID = 0

NANODET_CONFIGS = {                                         # available models and their configs
    "nanodet-plus-m-416":
        "config/nanodet-plus-m_416.yml"
}


@dataclass
class NanoDetDetector:
    """NanoDet detector adapter.

    Returns pedestrian detections as Detection(x, y, w, h, confidence),
    where x/y/w/h are in MOT format.

    Expected external dependency:
        external/nanodet/

    This code inspired by https://github.com/RangiLyu/nanodet/blob/main/demo/demo.py
    """

    model_name: str = "nanodet-plus-m_416"
    repo_path: str = "external/nanodet"
    model_path: str = "resources/networks/nanodet/nanodet-plus-m_416.pth"
    confidence_threshold: float = 0.35
    device: str | None = None

    def __post_init__(self) -> None:
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        self.repo_path = os.path.abspath(self.repo_path)
        self.model_path = os.path.abspath(self.model_path)

        if self.model_name not in NANODET_CONFIGS:
            raise ValueError(
                f"Unknown NanoDet model: {self.model_name}"
            )

        self.config_path = os.path.join(
            self.repo_path,
            NANODET_CONFIGS[self.model_name],
        )

        if not os.path.isdir(self.repo_path):
            raise ValueError(f"NanoDet repo does not exist: {self.repo_path}")
        if not os.path.exists(self.config_path):
            raise ValueError(f"NanoDet config does not exist: {self.config_path}")
        if not os.path.exists(self.model_path):
            raise ValueError(f"NanoDet model does not exist: {self.model_path}")

        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)

        from nanodet.data.transform import Pipeline
        from nanodet.model.arch import build_model
        from nanodet.util import Logger, cfg, load_config, load_model_weight

        load_config(cfg, self.config_path)

        self.cfg = cfg
        self.logger = Logger(local_rank=-1, use_tensorboard=False)

        model = build_model(cfg.model)
    
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)

        class QuietLogger:
            def log(self, *args, **kwargs):
                pass
            def info(self, *args, **kwargs):
                pass            
            def warning(self, *args, **kwargs):
                print(*args)
            def error(self, *args, **kwargs):
                print(*args)

        load_model_weight(model, checkpoint, QuietLogger())
        self.model = model.to(self.device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    @property
    def name(self) -> str:
        base = os.path.basename(self.config_path)
        return os.path.splitext(base)[0]

    def detect(self, bgr_image: np.ndarray) -> list[Detection]:
        if bgr_image is None:
            raise ValueError("bgr_image is None")

        height, width = bgr_image.shape[:2]

        meta = {
            "img_info": {
                "id": 0,
                "file_name": None,
                "height": height,
                "width": width,
            },
            "raw_img": bgr_image,
            "img": bgr_image,
        }

        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta["img"] = meta["img"].unsqueeze(0)

        # wrap the batch to nanodet standards, derived empirically...
        meta["img_info"]["id"] = [meta["img_info"]["id"]]
        meta["img_info"]["file_name"] = [meta["img_info"]["file_name"]]
        meta["img_info"]["height"] = [meta["img_info"]["height"]]
        meta["img_info"]["width"] = [meta["img_info"]["width"]]
        if "warp_matrix" in meta:
            meta["warp_matrix"] = [meta["warp_matrix"]]

        with torch.no_grad():
            results = self.model.inference(meta)

        return self._parse_results(results, width, height)


    def _parse_results(self, results, image_width: int, image_height: int):
        detections = []

        if not isinstance(results, dict):
            raise TypeError(f"Expected NanoDet results dict, got {type(results)}")

        # NanoDet inference returns:
        #   {image_id: {class_id: [[x1, y1, x2, y2, score], ...]}}
        # For batch size 1, take the first image result.
        image_result = next(iter(results.values()))

        if not isinstance(image_result, dict):
            raise TypeError(
                f"Expected image result dict, got {type(image_result)}: {image_result}"
            )

        person_results = image_result.get(COCO_PERSON_CLASS_ID, [])

        for row in person_results:
            row = np.asarray(row).reshape(-1)

            if row.shape[0] < 5:
                continue

            x1, y1, x2, y2, score = row[:5]
            score = float(score)

            if score < self.confidence_threshold:
                continue

            x1 = max(0.0, min(float(x1), image_width - 1.0))
            y1 = max(0.0, min(float(y1), image_height - 1.0))
            x2 = max(0.0, min(float(x2), image_width - 1.0))
            y2 = max(0.0, min(float(y2), image_height - 1.0))

            w = x2 - x1
            h = y2 - y1

            if w <= 0 or h <= 0:
                continue

            detections.append(
                Detection(
                    x=x1,
                    y=y1,
                    w=w,
                    h=h,
                    confidence=score,
                    class_id=COCO_PERSON_CLASS_ID,
                )
            )

        return detections
