
"""
utils.py

Utilities to support our various applications.
"""

from __future__ import division, print_function, absolute_import

import os
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import shutil
import yaml

import cv2
import numpy as np

from .types import Detection

def update_metadata(
    metadata_path,
    *,
    parameters,
    sequence_name,
    sequence_stats,
):
    """
    Create or update metadata.yaml for the current experiment/sequence.

    If metadata exists but parameters/git differ from the current run,
    then the existing file is backed up and replaced.

    Args:
        metadata_path: Path to metadata.yaml
        parameters: Dict of run parameters
        sequence_name: video name
        sequence_stats: Dict containing fps, etc.
    """

    def utc_now_iso():
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    def git_info():
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                text=True
            ).strip()
            dirty = subprocess.call(
                ["git", "diff", "--quiet"]
            ) != 0
            return {
                "commit": commit,
                "dirty": dirty
            }

        except Exception:
            return {
                "commit": None,
                "dirty": None
            }

    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    now_string = utc_now_iso()

    current = {
        "created": now_string,
        "git": git_info(),
        "parameters": parameters,
        "sequences": {}
    }

    metadata = None
    if metadata_path.exists():
        with metadata_path.open("r") as f:
            metadata = yaml.safe_load(f) or {}

        old_parameters = metadata.get("parameters")
        old_git = metadata.get("git", {})

        if old_parameters != parameters or old_git != current["git"]:
            backup_path = metadata_path.with_suffix(
                metadata_path.suffix + f".bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            shutil.copy2(metadata_path, backup_path)
            print(f"overwriting existing metadata:", metadata_path)

            metadata = current
    else:
        metadata = current

    metadata.setdefault("sequences", {})
    metadata["sequences"][sequence_name] = {
        **sequence_stats,
        "updated": now_string,
    }

    with metadata_path.open("w") as f:
        yaml.safe_dump(
            metadata,
            f,
            sort_keys=False,
            default_flow_style=False,
        )

    return metadata


def gather_sequence_info(sequence_dir):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.
    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir) if not f.startswith('.')}
    if len(image_filenames) == 0:
        raise ValueError(f"No image files found in {image_dir}")

    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')
        groundtruth = np.atleast_2d(groundtruth)
    else:
        raise ValueError(f"Ground truth file not found: {groundtruth_file}")

    det_file = os.path.join(sequence_dir, "det/det.txt")
    if os.path.exists(det_file):
        detections = np.loadtxt(det_file, delimiter=',')
        detections = np.atleast_2d(detections)

    image = cv2.imread(next(iter(image_filenames.values())),
                        cv2.IMREAD_GRAYSCALE)
    image_size = image.shape
    min_frame_idx = min(image_filenames.keys())
    max_frame_idx = max(image_filenames.keys())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)
        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        raise ValueError(f"Sequence info file not found: {info_filename}")

    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "update_ms": update_ms
    }
    if detections is not None:
       seq_info["detections"] = detections

    return seq_info


def mot_gt_detections_for_frame(seq_info, frame_idx):
    groundtruth = seq_info["groundtruth"]
    frame_rows = groundtruth[groundtruth[:, 0].astype(int) == frame_idx]

    detections = []
    for row in frame_rows:
        _, track_id, x, y, w, h, mark, class_id, visibility = row[:9]

        if int(mark) != 1:
            continue

        det = Detection(
            x=float(x),
            y=float(y),
            w=float(w),
            h=float(h),
            confidence=1.0,
            class_id=int(class_id)
        )
        det.gt_track_id = int(track_id)
        detections.append(det)

    return detections


def make_output_dir(output_dir):
    """
    Validate path to "eval/trackers/DLCV/DLCV-train/<tracker-name>/data/"
    and then ensure the directory exists.
    """
    OUTPUT_DIR_ROOT = Path("eval/trackers/DLCV/DLCV-train").resolve()
    candidate = Path(output_dir).resolve()
    try:
        rel = candidate.relative_to(OUTPUT_DIR_ROOT)
    except ValueError:
        print("specified output directory:", output_dir)
        raise ValueError(
            f"Output directory must be under {OUTPUT_DIR_ROOT}"
        )
    parts = rel.parts
    if len(parts) != 2 or parts[1] != "data":
        print("specified output directory:", output_dir)
        raise ValueError(
            "Output directory must have form "
            "'eval/trackers/DLCV/DLCV-train/<tracker-name>/data'"
        )
    candidate.mkdir(parents=True, exist_ok=True)


def score_frame(gt_detections, pred_detections, iou_threshold):
    """
    Generate TP/FP/FN scoring for GT vs our detector predictions
    """
    def tlwh_iou(a, b):
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)

        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    gt_boxes = [d.tlwh for d in gt_detections]
    pred_boxes = [d.tlwh for d in pred_detections]

    matched_gt = set()
    matched_pred = set()

    pairs = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            pairs.append((tlwh_iou(gt, pred), gi, pi))

    pairs.sort(reverse=True)

    for iou, gi, pi in pairs:
        if iou < iou_threshold:
            break
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)

    tp = len(matched_gt)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def detector_quality(detector_stats):
    tp = detector_stats["tp"]
    fp = detector_stats["fp"]
    fn = detector_stats["fn"]

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )
    return {
        "iou_threshold": detector_stats["gt_iou_threshold"],
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "f1": f1
    }
