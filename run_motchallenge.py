# vim: expandtab:ts=4:sw=4
import argparse
import os
import run_tracker


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="MOTChallenge evaluation")
    parser.add_argument(
        "--suffix", help="Added to tracker results directory name.",
        default="")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value. Set to "
        "0.3 to reproduce results in the paper.",
        default=0.3, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maximum suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    mot_dir = "videos"
    sequences = os.listdir(mot_dir)
    for sequence in sequences:
        if sequence.startswith('.'):
            continue
        print("Running sequence %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)
        run_tracker.run(
            sequence_dir, args.suffix, args.min_confidence,
            args.nms_max_overlap, args.min_detection_height,
            args.max_cosine_distance, args.nn_budget, display=False)
