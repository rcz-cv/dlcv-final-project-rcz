# vim: expandtab:ts=4:sw=4
"""
run_tracker.py

Run our version of the DeepSORT tracker.
"""

from __future__ import division, print_function, absolute_import

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import argparse
import shutil
import yaml

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from common.types import Detection
import detectors
import reids

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.tracker import Tracker


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
        raise ValueError(
            f"Output directory must be under {OUTPUT_DIR_ROOT}"
        )
    parts = rel.parts
    if len(parts) != 2 or parts[1] != "data":
        raise ValueError(
            "Output directory must have form "
            "'eval/trackers/DLCV/DLCV-train/<tracker-name>/data'"
        )
    candidate.mkdir(parents=True, exist_ok=True)


def create_pipeline(parameters):
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine",
        parameters["max_cosine_distance"],
        parameters["nn_budget"],
    )
    tracker = Tracker(metric, max_age=parameters["max_age"])

    detector = detectors.create_detector(
        parameters["detector"],
        min_confidence=parameters["min_confidence"],
        min_detection_height=parameters["min_detection_height"],
    )

    reid = reids.create_reid_detector(
        parameters["reid"],
        use_detection_mask=parameters["mask"],
    )

    return detector, reid, tracker


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


class EvalTrack:
    """
    Trivial tracker to use GT as our results
    """
    def __init__(self, track_id, tlwh):
        self.track_id = int(track_id)
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.time_since_update = 0

    def is_confirmed(self):
        return True

    def to_tlwh(self):
        return self._tlwh


class ReidIdentityAssigner:
    def __init__(self, max_cosine_distance=0.2, nn_budget=100):
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.next_id = 1
        self.galleries = {}  # track_id -> list of feature vectors

    def update(self, detections):
        assigned_ids = []
        used_ids = set()

        for detection in detections:
            feature = np.asarray(detection.feature, dtype=np.float32)

            best_id = None
            best_distance = float("inf")

            for track_id, gallery in self.galleries.items():
                if track_id in used_ids:
                    continue

                distance = self._nearest_cosine_distance(feature, gallery)

                if distance < best_distance:
                    best_distance = distance
                    best_id = track_id

            if best_id is None or best_distance > self.max_cosine_distance:
                best_id = self.next_id
                self.next_id += 1
                self.galleries[best_id] = []

            self.galleries[best_id].append(feature)

            if self.nn_budget is not None:
                self.galleries[best_id] = self.galleries[best_id][-self.nn_budget:]

            assigned_ids.append(best_id)
            used_ids.add(best_id)

        return assigned_ids

    def _nearest_cosine_distance(self, feature, gallery):
        feature = self._normalize(feature)

        distances = []
        for gallery_feature in gallery:
            gallery_feature = self._normalize(gallery_feature)
            cosine_similarity = np.dot(feature, gallery_feature)
            cosine_distance = 1.0 - cosine_similarity
            distances.append(cosine_distance)

        return min(distances) if distances else float("inf")

    def _normalize(self, feature):
        norm = np.linalg.norm(feature)
        if norm == 0:
            return feature
        return feature / norm


def run(sequence_dir, output_dir, parameters):
    """
    Run multi-target tracker on a particular sequence.

    Arguments
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory, i.e. the videos
    output_dir : str
        Path to the output directory
    parameters : Dict
        See next

    Parameters
    ----------
    detector : str
        Name of the detector model
    reid : str
        Name of the reid model
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    max_age : int
        Gating threshold for cosine distance metric (object appearance).
    mask : boolean
        Specify to apply the segmentation mask (if any) before ReID.
    display : bool
        If True, show visualization of intermediate tracking results.

    Legacy Parameters
    -----------------
    nms_max_overlap: float
        Maximum detection overlap (non-maximum suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.

    """

    results = []

    seq_info = gather_sequence_info(sequence_dir)
    sequence_name = Path(sequence_dir).name

    make_output_dir(output_dir)
    output_file = str(Path(output_dir) / sequence_name) + ".txt"
    metadata_file = str(Path(output_dir) / "metadata.yaml")

    detector, reid, tracker = create_pipeline(parameters)

    prev_time = time.perf_counter()
    fps = 0.0
    detector_stats = {"tp": 0, "fp": 0, "fn": 0,
        "gt_iou_threshold": parameters["gt_iou_threshold"]}

    if parameters["gt_eval"]:
        reid_identity_assigner = ReidIdentityAssigner(
            max_cosine_distance=parameters["max_cosine_distance"],
            nn_budget=parameters["nn_budget"],
        )

    def frame_callback(vis, frame_idx):
        nonlocal prev_time,fps
        print("Processing frame %05d\r" % frame_idx, end="")
        # get our image
        image_filename = seq_info["image_filenames"][frame_idx]
        frame = cv2.imread(image_filename, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"WARNING: could not read frame: {image_filename}")
            return

        our_detections = detector.detect(frame)

        if parameters["gt_eval"]:                                   # evaluating detector/reid independently
            gt_detections = mot_gt_detections_for_frame(seq_info, frame_idx)

            tp, fp, fn = score_frame(
                gt_detections,
                our_detections,
                parameters["gt_iou_threshold"],
            )
            detector_stats["tp"] += tp
            detector_stats["fp"] += fp
            detector_stats["fn"] += fn

            detections = gt_detections
            detections = reid.reid(frame, detections)
            assigned_ids = reid_identity_assigner.update(gt_detections)

            active_tracks = [
                EvalTrack(track_id, detection.tlwh)
                for detection, track_id in zip(gt_detections, assigned_ids)
            ]
        else:
            detections = our_detections                             # running actual tracker...

            # Run non-maximum suppression on our detections
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(
                boxes, parameters["nms_max_overlap"], scores)
            detections = [detections[i] for i in indices]
            detections = reid.reid(frame, detections)

            tracker.predict()
            tracker.update(detections)
            active_tracks = tracker.tracks

        # Update speed metric.
        now = time.perf_counter()
        instantaneous_fps = 1.0 / (now - prev_time)
        prev_time = now
        if fps == 0.0:
            fps = instantaneous_fps
        else:
            fps = 0.9 * fps + 0.1 * instantaneous_fps

        # Update visualization.
        if parameters["display"]:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            cv2.putText(
                image,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(active_tracks)

        # add results
        for track in active_tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    if parameters["display"]:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)

    print("    fps:", fps)
    sequence_stats= {
        "fps": fps
    }
    if parameters["gt_eval"]:
        sequence_stats["det_stats"] = detector_quality(detector_stats)
    
    update_metadata(metadata_file,
        parameters=parameters,
        sequence_name=sequence_name,
        sequence_stats=sequence_stats
    )

def args_parser(description):
    """ Return parser for command line arguments.
    """
    if len(sys.argv) == 1:
        usage=argparse.SUPPRESS
    else:
        usage=None
    parser = argparse.ArgumentParser(description=description, usage=usage)
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        required=True)
    parser.add_argument(
        "--output_dir", help="Path to output directory",
        required=True)
    parser.add_argument(
        "--detector", help="Detector model to use.",
        type=str)
    parser.add_argument(
        "--reid", help="ReID model to use.",
        type=str)
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int)
    parser.add_argument(
        "--max_age", help="How many frames a track can go unmatched before "
        "it is deleted. If None, the default is 30.", type=int, default=30)
    parser.add_argument(
        "--mask", help="Apply mask before ReID for segmentation detectors",
        action=argparse.BooleanOptionalAction,
        default=None)
    parser.add_argument(
        "--gt_eval",
        help="Evaluate Detector and ReID against ground truth",
        action=argparse.BooleanOptionalAction,
        default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        action=argparse.BooleanOptionalAction,
        default=None)

    # legacy parameters
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=None, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maximum suppression threshold: Maximum "
        "detection overlap.", default=None, type=float)
    return parser

def get_parameters(args, parameters):
    if args.detector is not None:
        parameters["detector"] = args.detector
    if args.reid is not None:
        parameters["reid"] = args.reid
    if args.min_confidence is not None:
        parameters["min_confidence"] = args.min_confidence
    if args.max_cosine_distance is not None:
        parameters["max_cosine_distance"] = args.max_cosine_distance
    if args.nn_budget is not None:
        parameters["nn_budget"] = args.nn_budget
    if args.max_age is not None:
        parameters["max_age"] = args.max_age
    if args.mask is not None:
        parameters["mask"] = args.mask
    if args.gt_eval is not None:
        parameters["gt_eval"] = args.gt_eval
    if args.display is not None:
        parameters["display"] = args.display
    # legacy:
    if args.min_detection_height is not None:
        parameters["min_detection_height"] = args.min_detection_height
    if args.nms_max_overlap is not None:
        parameters["nms_max_overlap"] = args.nms_max_overlap
    return parameters


if __name__ == "__main__":
    DEFAULT_PARAMETERS = {
        "detector": "yolo26m",
        "reid": "mars",
        "min_confidence": 0.30,
        "max_cosine_distance": 0.2,
        "nn_budget": 100,
        "max_age": 30,
        "mask": False,
        "gt_eval": False,
        "gt_iou_threshold": 0.5,
        "display": True,
    # legacy:
        "min_detection_height": 0,
        "nms_max_overlap": 1.0
    }
    parameters = DEFAULT_PARAMETERS.copy()
    parser = args_parser("Deep SORT")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()

    parameters = get_parameters(args, parameters)
    run(args.sequence_dir, args.output_dir, parameters)
