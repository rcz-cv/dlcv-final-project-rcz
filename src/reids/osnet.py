"""
osnet.py

Create ReID detections based on an OSNet model through torchreid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np

from common.types import Detection


def _default_device() -> str:
    """Pick the best available torch device for OSNet inference."""
    from torch import backends, cuda

    if cuda.is_available():
        return "cuda"
    if backends.mps.is_available():
        return "mps"
    return "cpu"


def extract_image_patch(
    image: np.ndarray,
    bbox: np.ndarray,
    patch_shape: tuple[int, int],
    mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """Extract image patch from bbox in TLWH format: x, y, w, h."""
    bbox = np.array(bbox)

    target_aspect = float(patch_shape[1]) / patch_shape[0]
    new_width = target_aspect * bbox[3]
    bbox[0] -= (new_width - bbox[2]) / 2.0
    bbox[2] = new_width

    # TLWH -> XYXY
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int64)

    # Clip to image bounds.
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])

    if np.any(bbox[:2] >= bbox[2:]):
        return None

    sx, sy, ex, ey = bbox

    patch = image[sy:ey, sx:ex]
    patch = cv2.resize(patch, tuple(patch_shape[::-1]))

    if mask is not None:
        mask_patch = mask[sy:ey, sx:ex]
        mask_patch = cv2.resize(
            mask_patch.astype(np.float32),
            tuple(patch_shape[::-1]),
            interpolation=cv2.INTER_LINEAR,
        )

        patch = patch.copy()
        patch[mask_patch < 0.5] = 0

    return patch


class OSNetImageEncoder:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: str = "cpu",
        image_shape: tuple[int, int] = (256, 128),
        feature_dim: int = 512,
        torchreid_root: str | Path | None = None,
    ) -> None:
        if torchreid_root is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            torchreid_root = project_root / "external" / "deep-person-reid"

        torchreid_root = Path(torchreid_root)
        if str(torchreid_root) not in sys.path:
            sys.path.insert(0, str(torchreid_root))
        print("torchreid_root:", str(torchreid_root))

        from torchreid.utils import FeatureExtractor

        self.feature_dim = feature_dim
        self.image_shape = image_shape
        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            device=device,
        )

    def __call__(self, images: np.ndarray, batch_size: int = 32) -> np.ndarray:
        if len(images) == 0:
            return np.zeros((0, self.feature_dim), np.float32)

        out = np.zeros((len(images), self.feature_dim), np.float32)

        for start in range(0, len(images), batch_size):
            end = min(start + batch_size, len(images))

            batch: list[np.ndarray] = []
            for patch in images[start:end]:
                if patch.ndim != 3:
                    raise ValueError(
                        f"OSNetImageEncoder: expected HxWxC patch, got {patch.shape}"
                    )

                # OpenCV images enter the tracker as BGR, but torchreid's
                # FeatureExtractor expects RGB image arrays.
                batch.append(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))

            features = self.extractor(batch)
            if hasattr(features, "detach"):
                features = features.detach().cpu().numpy()

            out[start:end] = features.astype(np.float32)

        return out


@dataclass
class OsnetReid:
    model_filename: str = "resources/networks/osnet_x1_0_market1501.pth.tar"
    model_name: str = "osnet_x1_0"
    batch_size: int = 32
    device: str | None = None
    image_shape: tuple[int, int] = (256, 128)
    feature_dim: int = 512
    use_detection_mask: bool = False
    torchreid_root: str | Path | None = None

    def __post_init__(self) -> None:
        self.encoder = OSNetImageEncoder(
            model_name=self.model_name,
            model_path=self.model_filename,
            device=self.device or _default_device(),
            image_shape=self.image_shape,
            feature_dim=self.feature_dim,
            torchreid_root=self.torchreid_root,
        )
        self.image_shape = self.encoder.image_shape

    @property
    def name(self) -> str:
        return self.model_name

    def reid(
        self,
        image: np.ndarray,
        detections: list[Detection],
    ) -> list[Detection]:
        """Populate Detection.feature for each detection and return detections."""
        if image is None:
            raise ValueError("image is None")

        if not detections:
            return detections

        patches: list[np.ndarray] = []
        valid_detections: list[Detection] = []

        for detection in detections:
            patch = extract_image_patch(
                image,
                detection.tlwh,
                tuple(self.image_shape[:2]),
                mask=detection.mask if self.use_detection_mask else None,
            )

            if patch is None:
                print(f"WARNING: Failed to extract image patch: {detection.tlwh}")
                patch = np.random.uniform(
                    0.0,
                    255.0,
                    (*self.image_shape, 3),
                ).astype(np.uint8)

            patches.append(patch)
            valid_detections.append(detection)

        if not patches:
            return []

        features = self.encoder(
            np.asarray(patches),
            batch_size=self.batch_size,
        )

        for detection, feature in zip(valid_detections, features):
            detection.feature = feature.astype(np.float32)

        return valid_detections
