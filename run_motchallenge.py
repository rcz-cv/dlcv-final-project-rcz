# vim: expandtab:ts=4:sw=4
import argparse
import os
import sys
import run_tracker


def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

# parse_args
#
# This front end is copied from run_tracker.py
#
def parse_args():
    """ Parse command line arguments.
    """
    if len(sys.argv) == 1:
        usage=argparse.SUPPRESS
    else:
        usage=None
    parser = argparse.ArgumentParser(description="Deep SORT on all videos", usage=usage)
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
        "--display", help="Show visual tracking during execution",
        default=False, type=bool_string)

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

    mot_dir = "videos"
    sequences = sorted(os.listdir(mot_dir))
    for sequence in sequences:
        if sequence.startswith('.'):
            continue
        print("Running sequence %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)
        run_tracker.run(
            sequence_dir, args.output_dir, args.detector, args.reid,
            args.min_confidence, args.max_cosine_distance, args.nn_budget, args.max_age,
            args.mask, args.display, args.nms_max_overlap, args.min_detection_height)
