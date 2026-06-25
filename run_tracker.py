# vim: expandtab:ts=4:sw=4
"""
run_tracker.py

Run our version of the DeepSORT tracker.
"""

from __future__ import division, print_function, absolute_import

import sys
import time
from pathlib import Path
import argparse

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from common import (
    update_metadata,
    gather_sequence_info,
    make_output_dir,
    mot_gt_detections_for_frame,
    score_frame,
    detector_quality
)

import detectors
from detectors import Detector
import reids

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.tracker import Tracker


def create_pipeline(parameters):
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine",
        parameters["max_cosine_distance"],
        parameters["nn_budget"],
    )
    tracker = Tracker(metric, max_age=parameters["max_age"])

    detector: Detector = detectors.create_detector(
        parameters["detector"],
        min_confidence=parameters["min_confidence"],
        min_detection_height=parameters["min_detection_height"],
    )

    reid = reids.create_reid_detector(
        parameters["reid"],
        use_detection_mask=parameters["mask"],
    )

    return detector, reid, tracker


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
    def __init__(self, max_cosine_distance, nn_budget):
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
    if hasattr(detector, "sequence"):
        detector.sequence(seq_info)                                 # for MOT16 detector, or ignored

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
        eol = "\n" if frame_idx % 10 == 0 else ""
        print("Processing frame %05d\r" % frame_idx, end=eol)
        # get our image
        image_filename = seq_info["image_filenames"][frame_idx]
        frame = cv2.imread(image_filename, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"WARNING: could not read frame: {image_filename}")
            return

        our_detections = detector.detect(frame, frame_idx)

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
        help="Evaluate operation against ground truth",
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
        "min_confidence": 0.3,
        "max_cosine_distance": 0.2,
        "nn_budget": None,
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
