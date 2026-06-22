# vim: expandtab:ts=4:sw=4
"""
run_identity.py

Run our version of the DeepSORT tracker with body ReID
"""

from __future__ import division, print_function, absolute_import

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from identity import IdentityManager

from run_tracker import (
    gather_sequence_info,
    make_output_dir,
    update_metadata,
    get_parameters,
    create_pipeline,
    args_parser
)


def run(sequence_dir, output_dir, parameters):
    """
    Run Body ReID on a particular sequence.

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
    identity_manager = IdentityManager(
        knn_k=parameters["knn_k"],
        knn_min_votes=parameters["knn_min_votes"],
        id_window=parameters["id_window"],
        identity_max_distance=parameters["identity_max_distance"],
        min_majority_count=parameters["min_majority_count"],
        conflict_policy=parameters["conflict_policy"]
    )

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
            boxes, parameters["nms_max_overlap"], scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        identity_manager.update(frame_idx, tracker.tracks,)
        identity_manager.resolve_active_track_identities(tracker.tracks)
        identity_manager.resolve_conflicts(tracker.tracks)

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
            vis.draw_identities(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            identity_id = getattr(track, "identity_id", None)
            if identity_id is None:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, identity_id, bbox[0], bbox[1], bbox[2], bbox[3],
            ])

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
    update_metadata(metadata_file,
        parameters=parameters,
        sequence_name=sequence_name,
        sequence_stats=sequence_stats
    )


def add_parser(parser):
    parser.add_argument(
        "--knn_k", help="Number of nearest gallery descriptors used for KNN identity voting.",
        type=str)
    parser.add_argument(
        "--knn_min_votes", help="KNN votes required to assign an existing identity.",
        type=str)
    parser.add_argument(
        "--id_window", help="Number of recent frames used for identity majority voting.",
        type=str)
    parser.add_argument(
        "--identity_max_distance", help="Maximum cosine distance for matching a descriptor to an existing identity.",
        type=str)
    parser.add_argument(
        "--min_majority_count", help="Minimum votes required before an identity becomes active for a track.",
        type=str)
    parser.add_argument(
        "--conflict_policy", help="Resolve identity assignments with mark|reset|competitive policy.",
        type=str)


def add_parameters(args, parameters):
    if args.knn_k is not None:
        parameters["knn_k"] = args.knn_k
    if args.knn_min_votes is not None:
        parameters["knn_min_votes"] = args.knn_min_votes
    if args.id_window is not None:
        parameters["id_window"] = args.id_window
    if args.identity_max_distance is not None:
        parameters["identity_max_distance"] = args.identity_max_distance
    if args.min_majority_count is not None:
        parameters["min_majority_count"] = args.min_majority_count
    if args.conflict_policy is not None:
        parameters["conflict_policy"] = args.conflict_policy
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
        "display": True,
    # legacy:
        "min_detection_height": 0,
        "nms_max_overlap": 1.0,
    # identity:
        "knn_k": 5,
        "knn_min_votes": 1,
        "id_window": 30,
        "identity_max_distance": 0.25,
        "min_majority_count": 1,
        "conflict_policy": "competitive"
    }
    parameters = DEFAULT_PARAMETERS.copy()
    parser = args_parser("Body ReID")
    add_parser(parser)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    parameters = get_parameters(args, parameters)
    add_parameters(args, parameters)

    run(args.sequence_dir, args.output_dir, parameters)
