# vim: expandtab:ts=4:sw=4
"""
run_tracker.py

Run our version of the DeepSORT tracker.
"""

from __future__ import division, print_function, absolute_import

import argparse
import os
import sys
import time

import cv2
import numpy as np

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from common.types import Detection
import detectors
import reids

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

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


def run(sequence_dir, output_dir, detector_name, reid_name,
        min_confidence, max_cosine_distance, nn_budget, max_age, 
        mask, display, nms_max_overlap, min_detection_height):
    """
    Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory, i.e. the videos
    output_dir : str
        Path to the output directory
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

    seq_info = gather_sequence_info(sequence_dir)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=max_age)
    results = []

    detector = detectors.create_detector(detector_name,
        min_confidence=min_confidence,
        min_detection_height=min_detection_height)
    reid = reids.create_reid_detector(reid_name, use_detection_mask=mask)

    sequence_name = Path(sequence_dir).name

    make_output_dir(output_dir)
    output_file = str(Path(output_dir) / sequence_name) + ".txt"

    prev_time = time.perf_counter()
    fps = 0.0

    def frame_callback(vis, frame_idx):
        nonlocal prev_time,fps
        print("Processing frame %05d\r" % frame_idx, end="")
        # get our image
        image_filename = seq_info["image_filenames"][frame_idx]
        frame = cv2.imread(image_filename, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"WARNING: could not read frame: {image_filename}")
            return

        detections = detector.detect(frame)
        detections = reid.reid(frame, detections)

        # Run non-maximum suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update speed metric.
        now = time.perf_counter()
        instantaneous_fps = 1.0 / (now - prev_time)
        prev_time = now
        if fps == 0.0:
            fps = instantaneous_fps
        else:
            fps = 0.9 * fps + 0.1 * instantaneous_fps

        # Update visualization.
        if display:
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
            vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    if display:
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

def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    if len(sys.argv) == 1:
        usage=argparse.SUPPRESS
    else:
        usage=None
    parser = argparse.ArgumentParser(description="Deep SORT", usage=usage)
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=False, required=True)
    parser.add_argument(
        "--output_dir", help="Path to output directory",
        default=False, required=True)
    parser.add_argument(
        "--detector", help="Detector model to use.",
        default="yolo26m")
    parser.add_argument(
        "--reid", help="ReID model to use.",
        default="mars")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--max_age", help="How many frames a track can go unmatched before "
        "it is deleted. If None, the default is 30.", type=int, default=30)
    parser.add_argument(
        "--mask", help="Apply mask before ReID for segmentation detectors",
        default=False, type=bool_string)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)

    # legacy parameters
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maximum suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.output_dir, args.detector, args.reid,
        args.min_confidence, args.max_cosine_distance, args.nn_budget, args.max_age,
        args.mask, args.display, args.nms_max_overlap, args.min_detection_height)
