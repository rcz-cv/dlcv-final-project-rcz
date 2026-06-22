# vim: expandtab:ts=4:sw=4
import argparse
import os
import sys
import run_tracker


def parse_args():
    """ Parse command line arguments.
    """
    if len(sys.argv) == 1:
        usage=argparse.SUPPRESS
    else:
        usage=None
    parser = argparse.ArgumentParser(description="Deep SORT", usage=usage)
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

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    return parser.parse_args()


def get_parameters(args):
    DEFAULT_PARAMETERS = {
        "detector": "yolo26m",
        "reid": "mars",
        "min_confidence": 0.30,
        "max_cosine_distance": 0.2,
        "nn_budget": 100,
        "max_age": 30,
        "mask": False,
        "display": False,
    # legacy:
        "min_detection_height": 0,
        "nms_max_overlap": 1.0
    }
    parameters = DEFAULT_PARAMETERS.copy()
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
    if args.display is not None:
        parameters["display"] = args.display
    # legacy:
    if args.min_detection_height is not None:
        parameters["min_detection_height"] = args.min_detection_height
    if args.nms_max_overlap is not None:
        parameters["nms_max_overlap"] = args.nms_max_overlap
    return parameters


if __name__ == "__main__":
    args = parse_args()
    parameters = get_parameters(args)

    mot_dir = "videos"
    sequences = sorted(os.listdir(mot_dir))
    for sequence in sequences:
        if sequence.startswith('.'):
            continue
        print("Running sequence %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)
        run_tracker.run(sequence_dir, args.output_dir, parameters)
