"""
mars.py

Create ReID detections based on the original DeepSORT MARS model.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import tensorflow as tf

from common.types import Detection


def _run_in_batches(f, data_dict, out, batch_size: int) -> None:
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)

    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


def extract_image_patch(
    image: np.ndarray,
    bbox: np.ndarray,
    patch_shape: tuple[int, int],
) -> np.ndarray | None:
    """Extract image patch from bbox in TLWH format: x, y, w, h."""
    bbox = np.asarray(bbox, dtype=np.float32).copy()

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
    return cv2.resize(patch, tuple(patch_shape[::-1]))


class ImageEncoder:
    def __init__(
        self,
        checkpoint_filename: str,
        input_name: str = "images",
        output_name: str = "features",
    ) -> None:
        self.session = tf.compat.v1.Session()

        with tf.compat.v1.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(file_handle.read())

        tf.import_graph_def(graph_def, name="net")

        graph = tf.compat.v1.get_default_graph()
        self.input_var = graph.get_tensor_by_name(f"{input_name}:0")
        self.output_var = graph.get_tensor_by_name(f"{output_name}:0")

        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, images: np.ndarray, batch_size: int = 32) -> np.ndarray:
        out = np.zeros((len(images), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: images},
            out,
            batch_size,
        )
        return out


@dataclass
class MarsReid:
    model_filename: str = "resources/networks/mars-small128/mars-small128.pb"
    batch_size: int = 32
    input_name: str = "images"
    output_name: str = "features"

    def __post_init__(self) -> None:
        self.encoder = ImageEncoder(
            self.model_filename,
            input_name=self.input_name,
            output_name=self.output_name,
        )
        self.image_shape = self.encoder.image_shape

    @property
    def name(self) -> str:
        return "mars"

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
            )

            if patch is None:
                continue

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
